# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint gradients in compiled code, for the `Native` integrator.

The model, its adjoint function (reverse-mode differentiation of the lowered
model) and the derived targets with their gradients are compiled by Native's
LLVM or wasm backend. `mxn_adjoint_vjp` (mxlpy-native or solvers.wasm) then
runs the forward and the backward solve, with the parameter quadrature, in C.
Problems it can't express raise `NativeUnsupported`, and the caller falls back
to the Python adjoint.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy.forward_sensitivities._core import _unwrap
from mxlpy.symbolic.jacobian import _generated, all_parameter_values
from mxlpy.types import IntegrationFailure

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.integrators._native.lower import Lowered
    from mxlpy.types import Array

    type _Model = KineticModelBuilder | OdeModelBuilder

__all__ = ["NativeUnsupported", "supports", "vjp"]

_DP = ct.POINTER(ct.c_double)
_TARGET_STATE, _TARGET_DERIVED = 0, 1
_TIME_COURSE = 0
_CHECK_RTOL = 1e-5
# mxn_adjoint_vjp's codes for problems it can't run; it returns failed solves as
# -1000 plus the solver's code, so they can't be mistaken for these
_SETUP_ERRORS = {
    -1: "out of memory",
    -3: "inconsistent experiments",
    -4: "overridden parameters",
    -5: "missing derived functions",
    -7: "missing functions",
}


class NativeUnsupported(Exception):
    """The compiled adjoint can't run this problem."""


@dataclass
class _Compiled:
    """The compiled functions of a model, kept alive while their addresses are used."""

    low: Lowered
    theta: list[int]
    target_kind: list[int]
    target_index: list[int]
    n_derived: int
    functions: list[Any] = field(default_factory=list)
    pointers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _Settings:
    backend: str
    solver: int
    rtol: float
    atol: float


def native_options(integrator: IntegratorType) -> tuple[str, Callable[[str], Any]]:
    """Backend of a Native integrator and a getter of its other options.

    The getter returns the integrator's keyword argument, else Native's
    default. Raises `NativeUnsupported` for other integrators and backends
    that aren't installed.
    """
    from mxlpy.integrators._native.backends import problem  # noqa: PLC0415
    from mxlpy.integrators.int_native import Native, _pick_backend  # noqa: PLC0415

    base, kwargs = _unwrap(integrator)
    if not (isinstance(base, type) and issubclass(base, Native)):
        msg = "the integrator isn't Native"
        raise NativeUnsupported(msg)

    def get(name: str) -> Any:
        return kwargs.get(name, getattr(base, name))

    backend = str(get("backend"))
    try:
        backend = _pick_backend() if backend == "auto" else backend
    except ImportError as e:
        raise NativeUnsupported(str(e)) from e
    if (reason := problem(backend)) is not None:
        raise NativeUnsupported(reason)
    return backend, get


def _settings(integrator: IntegratorType) -> _Settings:
    """Backend, solver and tolerances of a Native integrator."""
    from mxlpy.fit._native import _SOLVERS  # noqa: PLC0415

    backend, get = native_options(integrator)
    method = get("method")
    if method not in _SOLVERS:
        msg = f"the compiled adjoint supports the methods {list(_SOLVERS)}, not {method!r}"
        raise NativeUnsupported(msg)
    return _Settings(
        backend=backend,
        solver=_SOLVERS[str(method)],
        rtol=float(get("rtol")),
        atol=float(get("atol")),
    )


def _backend(name: str) -> Any:
    if name == "wasm":
        from mxlpy.integrators._native import wasm_backend  # noqa: PLC0415

        return wasm_backend
    from mxlpy.integrators._native import llvm_backend  # noqa: PLC0415

    return llvm_backend


def supports(integrator: IntegratorType) -> bool:
    """Whether the integrator can run compiled adjoints (models may still fail)."""
    try:
        _settings(integrator)
    except NativeUnsupported:
        return False
    return True


def _check_adjoint(
    adjoint: Any,
    plain: Any,
    low: Lowered,
    theta: list[int],
    *,
    pars: Array,
    y0: Array,
) -> None:
    """Compare the compiled adjoint function with central differences of the model.

    At the model's parameters, where its values are finite (random parameters
    can make real models NaN everywhere). At y0 every entry the differences
    can evaluate must agree. At random states around y0 non-finite adjoint
    entries are skipped, as symbolic derivatives can overflow where the model
    is finite.
    """
    from mxlpy.integrators.int_native import (  # noqa: PLC0415
        check_states,
        check_step,
        checked_entries,
    )

    rng = np.random.default_rng(0)
    n = low.n
    lam = rng.uniform(-1, 1, n)
    for i, y in enumerate(check_states(y0, 2, seed=0)):
        dlam, dtheta = adjoint.adjoint(0.0, y, lam, pars, len(theta))

        def partial(j: int, y: Array = y) -> float:
            """-lam^T df/dz_j by central differences, z = (y, pars)."""
            z = np.concatenate([y, pars])
            h = check_step(float(z[j]))
            dz = np.zeros_like(z)
            dz[j] = h
            up, down = z + dz, z - dz
            df = plain.rhs(0.0, up[:n], up[n:]) - plain.rhs(0.0, down[:n], down[n:])
            return -float(lam @ df) / (2 * h)

        expected = np.array(
            [partial(j) for j in range(n)] + [partial(n + k) for k in theta]
        )
        got = np.concatenate([dlam, dtheta[: len(theta)]])
        mask = checked_entries(got, expected, strict=i == 0)
        if not mask.any():
            if i == 0:
                msg = "the compiled adjoint can't be checked, the model isn't finite at y0"
                raise NativeUnsupported(msg)
            continue
        expected, got = expected[mask], got[mask]
        scale = _CHECK_RTOL * max(1.0, float(np.abs(expected).max()))
        if not np.allclose(got, expected, rtol=_CHECK_RTOL, atol=scale):
            msg = "the compiled adjoint disagrees with the model"
            raise NativeUnsupported(msg)


def _compile(
    model: _Model, wrt: list[str], outputs: list[str], backend_name: str
) -> _Compiled:
    from mxlpy.fit._native import _compile_derived, _Unsupported  # noqa: PLC0415
    from mxlpy.integrators._native import differentiate  # noqa: PLC0415
    from mxlpy.integrators._native.lower import (  # noqa: PLC0415
        NativeLoweringError,
        lower,
    )

    if getattr(model, "_events", None):
        msg = "the model has events"
        raise NativeUnsupported(msg)
    try:
        low = lower(model)
    except NativeLoweringError as e:
        raise NativeUnsupported(str(e)) from e
    if low.arrays or low.data:
        msg = "the model uses arrays or data"
        raise NativeUnsupported(msg)
    if unknown := [k for k in wrt if k not in low.par_names]:
        msg = f"{unknown} aren't model parameters"
        raise NativeUnsupported(msg)
    theta = [low.par_names.index(k) for k in wrt]
    var_names = list(model.get_variable_names())
    derived = [c for c in outputs if c not in var_names]

    backend = _backend(backend_name)
    try:
        plain = backend.compile_model(low)
        adjoint = backend.compile_model(
            differentiate.adjoint(low, theta), adjoint=low.n
        )
    except Exception as e:
        msg = f"the adjoint can't be compiled ({e})"
        raise NativeUnsupported(msg) from e
    values = all_parameter_values(model)
    pars = np.array([values[k] for k in low.par_names] or [0.0], dtype=float)
    y0 = np.array(list(model.get_initial_conditions().values()), dtype=float)
    _check_adjoint(adjoint, plain, low, theta, pars=pars, y0=y0)
    compiled = _Compiled(
        low=low,
        theta=theta,
        target_kind=[
            _TARGET_STATE if c in var_names else _TARGET_DERIVED for c in outputs
        ],
        target_index=[
            var_names.index(c) if c in var_names else derived.index(c) for c in outputs
        ],
        n_derived=len(derived),
        functions=[plain, adjoint],
        pointers=[plain.address, adjoint.address, 0, 0],
    )
    if derived:
        try:
            derived_fn, gradient_fn = _compile_derived(
                model, derived, theta, n=low.n, pars=pars, y0=y0, backend=backend
            )
        except _Unsupported as e:
            raise NativeUnsupported(str(e)) from e
        compiled.functions += [derived_fn, gradient_fn]
        compiled.pointers[2:] = [derived_fn.address, gradient_fn.address]
    return compiled


def _try_compile(
    model: _Model, wrt: list[str], outputs: list[str], backend: str
) -> _Compiled | NativeUnsupported:
    try:
        return _compile(model, wrt, outputs, backend)
    except NativeUnsupported as e:
        return e


# mxn_adjoint_vjp's arguments: ("i", int), ("d", float), ("ints", list[int]),
# ("doubles", array) or ("out", array) for outputs written by the solver
type _Arg = tuple[str, Any]


def _run_llvm(pointers: list[int], args: list[_Arg]) -> int:
    import mxlpy_native  # noqa: PLC0415

    lib = mxlpy_native.load()
    lib.mxn_adjoint_set_fns.argtypes = [ct.c_void_p] * 4
    lib.mxn_adjoint_set_fns.restype = None
    lib.mxn_adjoint_vjp.restype = ct.c_int
    values: list[Any] = []
    for kind, value in args:
        if kind == "i":
            values.append(ct.c_int(value))
        elif kind == "d":
            values.append(ct.c_double(value))
        elif kind == "ints":
            values.append((ct.c_int * max(len(value), 1))(*value))
        else:
            values.append(value.ctypes.data_as(_DP))
    lib.mxn_adjoint_set_fns(*pointers)
    return int(lib.mxn_adjoint_vjp(*values))


def _run_wasm(pointers: list[int], args: list[_Arg]) -> int:
    import wasmtime as wt  # noqa: PLC0415

    from mxlpy.integrators._native.wasm_backend import _runtime  # noqa: PLC0415

    rt = _runtime()
    values: list[Any] = []
    buffers: list[int] = []
    outputs: list[tuple[int, Any]] = []
    for kind, value in args:
        if kind in ("i", "d"):
            values.append(value)
            continue
        ptr = rt.write_ints(value) if kind == "ints" else rt.write(value)
        buffers.append(ptr)
        values.append(ptr)
        if kind == "out":
            outputs.append((ptr, value))
    try:
        rt.solver["mxn_adjoint_set_fns"](rt.store, *pointers)
        rc = int(rt.solver["mxn_adjoint_vjp"](rt.store, *values))
        for ptr, out in outputs:
            out[:] = rt.view(ptr, len(out))
    except wt.Trap:
        # e.g. a trap in a compiled function
        rc = -9
    finally:
        rt.free(*buffers)
    return rc


def vjp(
    model: _Model,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    x0: Array,
    integrator: IntegratorType,
    protocol: pd.DataFrame | None = None,
) -> tuple[Array, Array, Array] | Exception:
    """Compiled gradient of `sum(cotangents * y(t))` w.r.t. parameters `wrt` and x0.

    `wrt` are parameters (including those set by initial assignments, whose
    chain rule the caller applies). A protocol (TimedeltaIndex of step ends)
    becomes mxl-solvers' segments with parameter overrides. Returns the
    simulated values (time x cotangent columns), the gradient w.r.t. `wrt` and
    w.r.t. `x0`, or the solver's failure. Raises `NativeUnsupported` for
    problems it can't run.
    """
    from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

    settings = _settings(integrator)
    outputs = [str(c) for c in cotangents.columns]
    # Failures are cached too, so a fit doesn't retry them for every gradient.
    # Compiling for wasm uses the process-wide store, like Native's own calls
    with _LOCK:
        c = _generated(
            model,
            ("native_adjoint", settings.backend, tuple(wrt), tuple(outputs)),
            lambda: _try_compile(model, wrt, outputs, settings.backend),
        )
    if isinstance(c, NativeUnsupported):
        raise c
    values = all_parameter_values(model)
    pars = np.array([values[k] for k in c.low.par_names] or [0.0], dtype=float)
    times = cotangents.index.to_numpy(dtype=float)
    n_targets, n_points, n_y = len(outputs), len(times), len(x0)
    # Target-major: all time points of the first target, then the second, ...
    cot = np.ascontiguousarray(cotangents.to_numpy(dtype=float).T.ravel())
    pred = np.empty(max(n_targets * n_points, 1))
    grad = np.empty(max(len(wrt), 1))
    lam0 = np.empty(max(n_y, 1))
    if protocol is None:
        seg_t_end, ovr_idx, seg_ovr = np.array([times[-1]]), [], np.zeros(1)
    else:
        if protocol.isna().to_numpy().any():
            # The solvers keep the previous value for NaN, simulations set it
            msg = "the protocol contains NaN"
            raise NativeUnsupported(msg)
        if unknown := [k for k in protocol.columns if k not in c.low.par_names]:
            msg = f"protocol columns {unknown} aren't model parameters"
            raise NativeUnsupported(msg)
        seg_t_end = np.asarray(
            cast(pd.TimedeltaIndex, protocol.index).total_seconds(), dtype=float
        )
        ovr_idx = [c.low.par_names.index(k) for k in protocol.columns]
        seg_ovr = protocol.to_numpy(dtype=float).ravel()
    args: list[_Arg] = [
        ("i", n_y), ("doubles", np.ascontiguousarray(x0, dtype=float)),
        ("i", len(pars)), ("doubles", pars), ("i", len(c.theta)), ("ints", c.theta),
        ("i", n_targets), ("ints", c.target_kind), ("ints", c.target_index),
        ("i", len(ovr_idx)), ("ints", ovr_idx),
        ("i", 1), ("ints", [_TIME_COURSE]), ("ints", [len(seg_t_end)]),
        ("ints", [n_points]), ("i", len(seg_t_end)), ("doubles", seg_t_end),
        ("doubles", np.ascontiguousarray(seg_ovr, dtype=float)),
        ("i", n_points), ("doubles", np.ascontiguousarray(times)), ("doubles", cot),
        ("i", c.n_derived), ("i", settings.solver), ("d", settings.rtol),
        ("d", settings.atol), ("d", 1e-6), ("d", 1e6),
        ("out", pred), ("out", grad), ("out", lam0),
    ]  # fmt: skip
    run = _run_wasm if settings.backend == "wasm" else _run_llvm
    with _LOCK:
        rc = run(c.pointers, args)
    if rc in _SETUP_ERRORS:
        msg = f"the compiled adjoint couldn't start ({_SETUP_ERRORS[rc]})"
        raise NativeUnsupported(msg)
    if rc != 0:
        return IntegrationFailure()
    simulated = pred[: n_targets * n_points].reshape(n_targets, n_points).T
    return simulated, grad[: len(wrt)], lam0[:n_y]
