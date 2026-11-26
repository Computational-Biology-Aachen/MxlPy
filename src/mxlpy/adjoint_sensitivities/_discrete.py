# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Discrete adjoint gradients in compiled code, for the `Native` integrator.

`mxn_da_forward` (mxlpy-native or solvers.wasm, from mxl-solvers'
discrete_adjoint.c) integrates the model with Native's solver and records every
step on a tape; `mxn_da_backward` differentiates that recorded solution
exactly (step sizes frozen, Newton iterations by the implicit function
theorem), for any number of cotangents. So a loss evaluated on the forward
predictions and its gradient belong to the same computed solution.

The model, its Jacobian, its adjoint function and derived targets with their
gradients are compiled by Native's backend (see `_native`). Problems that
can't be compiled raise `NativeUnsupported`.
"""

from __future__ import annotations

import atexit
import ctypes as ct
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mxlpy.adjoint_sensitivities._native import (
    NativeUnsupported,
    _Compiled,
    _try_compile,
    native_options,
)
from mxlpy.symbolic.jacobian import _generated, all_parameter_values
from mxlpy.types import IntegrationFailure

if TYPE_CHECKING:
    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.types import Array

    type _Model = KineticModelBuilder | OdeModelBuilder

__all__ = ["NEWTON_SCALE", "Tape", "TapeTooLarge", "forward", "supports"]

_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_METHODS = {"Radau": 0, "LSODA": 1, "CVODE_BDF": 2, "CVODE_Adams": 3}
_REUSE_JACOBIAN, _FORWARD_JAC = 1, 2
# discrete_adjoint.h's codes
_NO_MEMORY, _NOT_FINITE, _TAPE_TOO_LARGE, _BAD_INPUT, _SINGULAR = (
    -1001,
    -1002,
    -1003,
    -1004,
    -1005,
)

# Factor on the solvers' Newton / corrector convergence tolerance while
# recording: the gradient is exact for converged iterations, so a tighter one
# makes it agree with the predictions (e.g. ~1e-4 vs ~1e-9 relative at
# rtol 1e-6, see mxl-solvers' tests/test_discrete_adjoint.c)
NEWTON_SCALE = 1e-3


class TapeTooLarge(NativeUnsupported):
    """A discrete adjoint's recorded solve would exceed `Native.max_tape_bytes`."""


@dataclass(frozen=True)
class _Settings:
    backend: str
    method: int
    rtol: float
    atol: float
    nmax: int
    options: int
    max_tape_bytes: float


def _settings(integrator: IntegratorType) -> _Settings:
    """Backend, method, tolerances and options of a Native integrator."""
    backend, get = native_options(integrator)
    method = get("method")
    if method not in _METHODS:
        msg = f"Native has no method {method!r}"
        raise NativeUnsupported(msg)
    options = (_REUSE_JACOBIAN if get("reuse_jacobian") else 0) | (
        _FORWARD_JAC if get("jacobian_method") == "analytic" else 0
    )
    return _Settings(
        backend=backend,
        method=_METHODS[str(method)],
        rtol=float(get("rtol")),
        atol=float(get("atol")),
        nmax=int(get("nmax")),
        options=options,
        max_tape_bytes=float(get("max_tape_bytes")),
    )


def supports(integrator: IntegratorType) -> bool:
    """Whether the integrator can run discrete adjoints (models may still fail)."""
    try:
        _settings(integrator)
    except NativeUnsupported:
        return False
    return True


@dataclass
class _Functions:
    """`_native`'s compiled functions plus the Jacobian, kept alive with the tape."""

    adjoint: _Compiled
    jacobian: Any

    @property
    def model(self) -> int:
        return self.adjoint.pointers[0]

    @property
    def adjoint_fn(self) -> int:
        return self.adjoint.pointers[1]

    @property
    def derived(self) -> int:
        return self.adjoint.pointers[2]

    @property
    def derived_grad(self) -> int:
        return self.adjoint.pointers[3]


def _compile(
    model: _Model, wrt: list[str], outputs: list[str], backend: str
) -> _Functions | NativeUnsupported:
    from mxlpy.integrators._native import differentiate  # noqa: PLC0415
    from mxlpy.integrators.int_native import (  # noqa: PLC0415
        _backend_module,
        _check_jacobian,
    )

    compiled = _try_compile(model, wrt, outputs, backend)
    if isinstance(compiled, NativeUnsupported):
        return compiled
    try:
        jacobian = _backend_module(backend).compile_model(
            differentiate.jacobian(compiled.low)
        )
        y0 = np.array(list(model.get_initial_conditions().values()), dtype=float)
        _check_jacobian(
            jacobian, compiled.functions[0], model, compiled.low.par_names, t=0.0, y0=y0
        )
    except Exception as e:  # noqa: BLE001
        return NativeUnsupported(f"the Jacobian can't be compiled ({e})")
    return _Functions(adjoint=compiled, jacobian=jacobian)


###############################################################################
# Backends
###############################################################################


class _Llvm:
    def __init__(self) -> None:
        import mxlpy_native  # noqa: PLC0415

        lib = mxlpy_native.load()
        d, i, v = ct.c_double, ct.c_int, ct.c_void_p
        lib.mxn_da_forward.argtypes = [
            *[v, v, v, i, i, _DP, i, i, _DP, _DP, i, _DP, i, _IP, _IP, i],
            *[d, d, i, i, d, d, _DP, ct.POINTER(v)],
        ]
        lib.mxn_da_forward.restype = i
        lib.mxn_da_backward.argtypes = [v, v, v, i, _DP, _DP, _DP]
        lib.mxn_da_backward.restype = i
        lib.mxn_da_replay.argtypes = [v, _DP, _DP, _DP]
        lib.mxn_da_replay.restype = i
        lib.mxn_da_free.argtypes = [v]
        lib.mxn_da_free.restype = None
        lib.mxn_da_tape_steps.argtypes = [v]
        lib.mxn_da_tape_steps.restype = i
        lib.mxn_da_tape_bytes.argtypes = [v]
        lib.mxn_da_tape_bytes.restype = d
        self.lib = lib

    def forward(
        self, pointers: tuple[int, int, int], ints: list[Any], out: Array
    ) -> tuple[int, int]:
        """Run mxn_da_forward, returning its code and the tape handle."""
        tape = ct.c_void_p()
        args = [
            a.ctypes.data_as(_DP)
            if isinstance(a, np.ndarray) and a.dtype == float
            else (ct.c_int * max(len(a), 1))(*a)
            if isinstance(a, list)
            else a
            for a in ints
        ]
        rc = self.lib.mxn_da_forward(
            *pointers, *args, out.ctypes.data_as(_DP), ct.byref(tape)
        )
        return int(rc), int(tape.value or 0)

    def backward(
        self,
        tape: int,
        pointers: tuple[int, int],
        n_theta: int,
        *,
        cot: Array,
        grad: Array,
        lam0: Array,
    ) -> int:
        cot_p, grad_p = cot.ctypes.data_as(_DP), grad.ctypes.data_as(_DP)
        rc = self.lib.mxn_da_backward(
            tape, *pointers, n_theta, cot_p, grad_p, lam0.ctypes.data_as(_DP)
        )
        return int(rc)

    def replay(self, tape: int, seg_pars: Array, x0: Array, out: Array) -> int:
        pars_p, x0_p = seg_pars.ctypes.data_as(_DP), x0.ctypes.data_as(_DP)
        return int(self.lib.mxn_da_replay(tape, pars_p, x0_p, out.ctypes.data_as(_DP)))

    def steps(self, tape: int) -> tuple[int, float]:
        return int(self.lib.mxn_da_tape_steps(tape)), float(
            self.lib.mxn_da_tape_bytes(tape)
        )

    def free(self, tape: int) -> None:
        self.lib.mxn_da_free(tape)


class _Wasm:
    def __init__(self) -> None:
        from mxlpy.integrators._native.wasm_backend import _runtime  # noqa: PLC0415

        self.rt = _runtime()

    def forward(
        self, pointers: tuple[int, int, int], ints: list[Any], out: Array
    ) -> tuple[int, int]:
        import wasmtime as wt  # noqa: PLC0415

        rt = self.rt
        buffers: list[int] = []
        args: list[Any] = []
        for a in ints:
            if isinstance(a, np.ndarray):
                ptr = rt.write(np.ascontiguousarray(a, dtype=float))
            elif isinstance(a, list):
                ptr = rt.write_ints(a)
            else:
                args.append(a)
                continue
            buffers.append(ptr)
            args.append(ptr)
        p_out = rt.malloc(len(out))
        p_tape = rt.write_ints([0])
        buffers += [p_out, p_tape]
        try:
            rc = int(
                rt.solver["mxn_da_forward"](rt.store, *pointers, *args, p_out, p_tape)
            )
            out[:] = rt.view(p_out, len(out))
            base = ct.addressof(rt.memory.data_ptr(rt.store).contents)
            tape = int(ct.c_uint32.from_address(base + p_tape).value)
        except wt.Trap:
            rc, tape = -9, 0
        finally:
            rt.free(*buffers)
        return rc, tape

    def backward(
        self,
        tape: int,
        pointers: tuple[int, int],
        n_theta: int,
        *,
        cot: Array,
        grad: Array,
        lam0: Array,
    ) -> int:
        import wasmtime as wt  # noqa: PLC0415

        rt = self.rt
        p_cot, p_grad, p_lam = rt.write(cot), rt.malloc(len(grad)), rt.malloc(len(lam0))
        try:
            rc = int(
                rt.solver["mxn_da_backward"](
                    rt.store, tape, *pointers, n_theta, p_cot, p_grad, p_lam
                )
            )
            grad[:] = rt.view(p_grad, len(grad))
            lam0[:] = rt.view(p_lam, len(lam0))
        except wt.Trap:
            rc = -9
        finally:
            rt.free(p_cot, p_grad, p_lam)
        return rc

    def replay(self, tape: int, seg_pars: Array, x0: Array, out: Array) -> int:
        rt = self.rt
        p_pars, p_x0, p_out = rt.write(seg_pars), rt.write(x0), rt.malloc(len(out))
        try:
            rc = int(rt.solver["mxn_da_replay"](rt.store, tape, p_pars, p_x0, p_out))
            out[:] = rt.view(p_out, len(out))
        finally:
            rt.free(p_pars, p_x0, p_out)
        return rc

    def steps(self, tape: int) -> tuple[int, float]:
        rt = self.rt
        return int(rt.solver["mxn_da_tape_steps"](rt.store, tape)), float(
            rt.solver["mxn_da_tape_bytes"](rt.store, tape)
        )

    def free(self, tape: int) -> None:
        self.rt.solver["mxn_da_free"](self.rt.store, tape)


_BACKENDS: dict[str, Any] = {}
# wasmtime objects can't be freed once the interpreter has started shutting down.
atexit.register(_BACKENDS.clear)


def _backend(name: str) -> Any:
    if name not in _BACKENDS:
        _BACKENDS[name] = _Wasm() if name == "wasm" else _Llvm()
    return _BACKENDS[name]


def _free(backend: Any, tape: int) -> None:
    from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

    try:
        with _LOCK:
            backend.free(tape)
    except Exception:  # noqa: BLE001  wasmtime is gone once the interpreter shuts down
        pass


###############################################################################
# Tape
###############################################################################


class Tape:
    """A recorded forward solve, differentiated by `backward` for any cotangents.

    Holds solver memory (freed with the object) and the compiled functions.
    """

    def __init__(
        self,
        backend: Any,
        handle: int,
        functions: _Functions,
        *,
        n_y: int,
        n_theta: int,
        shape: tuple[int, int],
    ) -> None:
        self._backend = backend
        self._handle = handle
        self._functions = functions
        self.n_y = n_y
        self.n_theta = n_theta
        self.shape = shape
        self._finalizer = weakref.finalize(self, _free, backend, handle)
        # At exit the process' memory goes anyway, and wasmtime may be gone
        self._finalizer.atexit = False

    @property
    def steps(self) -> int:
        """Accepted solver steps on the tape."""
        return self._backend.steps(self._handle)[0]

    @property
    def nbytes(self) -> float:
        """Memory the tape holds."""
        return self._backend.steps(self._handle)[1]

    def replay(self, segment_parameters: Array, x0: Array) -> Array | Exception:
        """Re-solve the recorded steps with other parameters and initial state.

        For gradient checks: `backward` is the exact derivative of this function
        (the recorded step sizes, Newton iterations converged).
        `segment_parameters` has one row per segment, in the compiled model's
        parameter order (`parameter_names`).
        """
        from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

        out = np.zeros(max(self.shape[0] * self.shape[1], 1))
        with _LOCK:
            rc = self._backend.replay(
                self._handle,
                np.ascontiguousarray(segment_parameters, dtype=float).ravel(),
                np.ascontiguousarray(x0, dtype=float),
                out,
            )
        if rc != 0:
            return ArithmeticError(f"the replay failed ({rc})")
        return out[: self.shape[0] * self.shape[1]].reshape(self.shape)

    @property
    def parameter_names(self) -> list[str]:
        """Parameters of the compiled model, the columns of `replay`'s parameters."""
        return list(self._functions.adjoint.low.par_names)

    def backward(self, cotangent: Array) -> tuple[Array, Array] | Exception:
        """Gradients w.r.t. the parameters and the initial state.

        `cotangent` is laid out like the predictions, time points x targets.
        """
        from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

        cot = np.ascontiguousarray(cotangent, dtype=float).ravel()
        grad = np.zeros(max(self.n_theta, 1))
        lam0 = np.zeros(max(self.n_y, 1))
        f = self._functions
        with _LOCK:
            rc = self._backend.backward(
                self._handle,
                (f.adjoint_fn, f.derived_grad),
                self.n_theta,
                cot=cot,
                grad=grad,
                lam0=lam0,
            )
        if rc != 0:
            return ArithmeticError(f"the discrete adjoint failed ({rc})")
        return grad[: self.n_theta], lam0[: self.n_y]


def forward(
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    x0: Array,
    integrator: IntegratorType,
    segments: list[tuple[float, dict[str, float]]] | None = None,
    newton_scale: float = NEWTON_SCALE,
) -> tuple[Array, Tape] | Exception:
    """Record a forward solve, returning the predictions and the tape.

    `wrt` are model parameters (the caller applies initial assignments' chain
    rule), `columns` variables and derived values. `segments` are protocol
    steps (end time, parameter overrides); a time point on a step's end belongs
    to it. Returns the predictions (time points x columns) and the tape, or the
    solver's failure. Raises `NativeUnsupported` (`TapeTooLarge`) for problems
    it can't run.
    """
    from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

    from ._core import _overridden  # noqa: PLC0415

    settings = _settings(integrator)
    with _LOCK:
        f = _generated(
            model,
            ("native_discrete_adjoint", settings.backend, tuple(wrt), tuple(columns)),
            lambda: _compile(model, wrt, columns, settings.backend),
        )
    if isinstance(f, NativeUnsupported):
        raise f
    c = f.adjoint
    if segments is None:
        segments = [(float(times[-1]), {})]
    names = c.low.par_names
    seg_pars = []
    for _, overrides in segments:
        if unknown := [k for k in overrides if k not in names]:
            msg = f"protocol columns {unknown} aren't model parameters"
            raise NativeUnsupported(msg)
        with _overridden(model, overrides):
            values = all_parameter_values(model)
        seg_pars.append([values[k] for k in names] or [0.0])
    n_pars = len(seg_pars[0])
    n_y, n_points, n_targets = len(x0), len(times), len(columns)
    pred = np.empty(max(n_points * n_targets, 1))
    args: list[Any] = [
        settings.method, n_y, np.ascontiguousarray(x0, dtype=float), n_pars,
        len(segments), np.array([s[0] for s in segments], dtype=float),
        np.ascontiguousarray(seg_pars, dtype=float).ravel(),
        n_points, np.ascontiguousarray(times, dtype=float), n_targets,
        list(c.target_kind), list(c.target_index), c.n_derived,
        settings.rtol, settings.atol, settings.nmax, settings.options,
        float(newton_scale), settings.max_tape_bytes,
    ]  # fmt: skip
    backend = _backend(settings.backend)
    with _LOCK:
        rc, handle = backend.forward(
            (f.model, f.jacobian.address, f.derived), args, pred
        )
    if rc == _TAPE_TOO_LARGE:
        msg = f"the tape would exceed max_tape_bytes={settings.max_tape_bytes:g}"
        raise TapeTooLarge(msg)
    if rc in (_NO_MEMORY, _BAD_INPUT):
        msg = f"the discrete adjoint couldn't start ({rc})"
        raise NativeUnsupported(msg)
    if rc != 0 or handle == 0:
        return IntegrationFailure()
    tape = Tape(
        backend, handle, f, n_y=n_y, n_theta=len(c.theta), shape=(n_points, n_targets)
    )
    return pred[: n_points * n_targets].reshape(n_points, n_targets), tape
