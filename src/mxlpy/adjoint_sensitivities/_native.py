# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint gradients in compiled code, for the `Native` integrator.

The model, its adjoint function (reverse-mode differentiation of the lowered
model) and the derived targets with their gradients are compiled by Native's
LLVM backend. mxlpy-native's `mxn_adjoint_vjp` then runs the forward and the
backward solve, with the parameter quadrature, in C. Problems it can't express
raise `NativeUnsupported`, and the caller falls back to the Python adjoint.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mxlpy.forward_sensitivities._core import _unwrap
from mxlpy.symbolic.jacobian import _generated, all_parameter_values
from mxlpy.types import IntegrationFailure

if TYPE_CHECKING:
    import pandas as pd

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.integrators._native.lower import Lowered
    from mxlpy.types import Array

    type _Model = KineticModelBuilder | OdeModelBuilder

__all__ = ["NativeUnsupported", "supports", "vjp"]

_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_ADJOINT_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP, _DP, _DP)
_TARGET_STATE, _TARGET_DERIVED = 0, 1
_TIME_COURSE = 0
_CHECK_RTOL = 1e-5
# mxn_adjoint_vjp's codes for problems it can't run (rather than failed solves)
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


def _settings(integrator: IntegratorType) -> tuple[int, float, float]:
    """Solver and tolerances of a Native integrator using the LLVM backend."""
    from mxlpy.fit._native import _SOLVERS  # noqa: PLC0415
    from mxlpy.integrators._native.backends import problem  # noqa: PLC0415
    from mxlpy.integrators.int_native import Native  # noqa: PLC0415

    base, kwargs = _unwrap(integrator)
    if not (isinstance(base, type) and issubclass(base, Native)):
        msg = "the integrator isn't Native"
        raise NativeUnsupported(msg)
    if kwargs.get("backend", getattr(base, "backend", "auto")) == "wasm":
        msg = "the wasm backend has no compiled adjoint"
        raise NativeUnsupported(msg)
    if (reason := problem("llvm")) is not None:
        raise NativeUnsupported(reason)
    method = kwargs.get("method", getattr(base, "method", None))
    if method not in _SOLVERS:
        msg = f"the compiled adjoint supports the methods {list(_SOLVERS)}, not {method!r}"
        raise NativeUnsupported(msg)
    return (
        _SOLVERS[str(method)],
        float(cast(float, kwargs.get("rtol", getattr(base, "rtol", 1e-8)))),
        float(cast(float, kwargs.get("atol", getattr(base, "atol", 1e-8)))),
    )


def supports(integrator: IntegratorType) -> bool:
    """Whether the integrator can run compiled adjoints (models may still fail)."""
    try:
        _settings(integrator)
    except NativeUnsupported:
        return False
    return True


def _check_adjoint(adjoint: Any, plain: Any, low: Lowered, theta: list[int]) -> None:
    """Compare the compiled adjoint function with central differences of the model."""
    rng = np.random.default_rng(0)
    n = low.n
    pars = rng.uniform(0.5, 1.5, max(len(low.par_names), 1))
    y = rng.uniform(0.5, 1.5, n)
    lam = rng.uniform(-1, 1, n)
    dlam, dtheta = np.empty(n), np.empty(max(len(theta), 1))
    fn = _ADJOINT_FN(adjoint.address)
    fn(
        n, 0.0, y.ctypes.data_as(_DP), lam.ctypes.data_as(_DP), pars.ctypes.data_as(_DP),
        dlam.ctypes.data_as(_DP), dtheta.ctypes.data_as(_DP),
    )  # fmt: skip

    def directional(dy: Array, dp: Array) -> float:
        h = 1e-6 * max(1.0, float(np.abs(y @ dy) + np.abs(pars @ dp)))
        up = plain.rhs(0.0, y + h * dy, pars + h * dp)
        down = plain.rhs(0.0, y - h * dy, pars - h * dp)
        return -float(lam @ (up - down)) / (2 * h)

    expected = [directional(np.eye(n)[i], np.zeros_like(pars)) for i in range(n)]
    expected += [directional(np.zeros(n), np.eye(len(pars))[k]) for k in theta]
    got = np.concatenate([dlam, dtheta[: len(theta)]])
    scale = _CHECK_RTOL * max(1.0, float(np.abs(expected).max(initial=0.0)))
    if not np.allclose(got, expected, rtol=_CHECK_RTOL, atol=scale):
        msg = "the compiled adjoint disagrees with the model"
        raise NativeUnsupported(msg)


def _compile(model: _Model, wrt: list[str], outputs: list[str]) -> _Compiled:
    from mxlpy.fit._native import _compile_derived, _Unsupported  # noqa: PLC0415
    from mxlpy.integrators._native import differentiate, llvm_backend  # noqa: PLC0415
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

    try:
        plain = llvm_backend.compile_model(low)
        adjoint = llvm_backend.compile_model(
            differentiate.adjoint(low, theta), adjoint=low.n
        )
    except Exception as e:
        msg = f"the adjoint can't be compiled ({e})"
        raise NativeUnsupported(msg) from e
    _check_adjoint(adjoint, plain, low, theta)
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
        rng = np.random.default_rng(2)
        try:
            derived_fn, gradient_fn = _compile_derived(
                model,
                derived,
                theta,
                n=low.n,
                pars=rng.uniform(0.5, 1.5, max(len(low.par_names), 1)),
                y0=rng.uniform(0.5, 1.5, low.n),
            )
        except _Unsupported as e:
            raise NativeUnsupported(str(e)) from e
        compiled.functions += [derived_fn, gradient_fn]
        compiled.pointers[2:] = [derived_fn.address, gradient_fn.address]
    return compiled


def _try_compile(
    model: _Model, wrt: list[str], outputs: list[str]
) -> _Compiled | NativeUnsupported:
    try:
        return _compile(model, wrt, outputs)
    except NativeUnsupported as e:
        return e


def _library() -> ct.CDLL:
    import mxlpy_native  # noqa: PLC0415

    lib = mxlpy_native.load()
    lib.mxn_adjoint_set_fns.argtypes = [ct.c_void_p] * 4
    lib.mxn_adjoint_set_fns.restype = None
    d = ct.c_double
    lib.mxn_adjoint_vjp.restype = ct.c_int
    lib.mxn_adjoint_vjp.argtypes = [
        ct.c_int, _DP, ct.c_int, _DP, ct.c_int, _IP,
        ct.c_int, _IP, _IP, ct.c_int, _IP,
        ct.c_int, _IP, _IP, _IP, ct.c_int, _DP, _DP,
        ct.c_int, _DP, _DP, ct.c_int, ct.c_int, d, d, d, d,
        _DP, _DP, _DP,
    ]  # fmt: skip
    return lib


def _ints(values: list[int]) -> Any:
    return (ct.c_int * max(len(values), 1))(*values)


def vjp(
    model: _Model,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    x0: Array,
    integrator: IntegratorType,
) -> tuple[Array, Array, Array] | Exception:
    """Compiled gradient of `sum(cotangents * y(t))` w.r.t. parameters `wrt` and x0.

    `wrt` are parameters (including those set by initial assignments, whose
    chain rule the caller applies). Returns the simulated values (time x
    cotangent columns), the gradient w.r.t. `wrt` and w.r.t. `x0`, or the
    solver's failure. Raises `NativeUnsupported` for problems it can't run.
    """
    from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

    solver, rtol, atol = _settings(integrator)
    outputs = [str(c) for c in cotangents.columns]
    # Failures are cached too, so a fit doesn't retry them for every gradient
    c = _generated(
        model,
        ("native_adjoint", tuple(wrt), tuple(outputs)),
        lambda: _try_compile(model, wrt, outputs),
    )
    if isinstance(c, NativeUnsupported):
        raise c
    values = all_parameter_values(model)
    pars = np.array([values[k] for k in c.low.par_names] or [0.0], dtype=float)
    times = cotangents.index.to_numpy(dtype=float)
    n_targets, n_points, n_y = len(outputs), len(times), len(x0)
    # Target-major: all time points of the first target, then the second, ...
    cot = np.ascontiguousarray(cotangents.to_numpy(dtype=float).T.ravel())
    y0 = np.ascontiguousarray(x0, dtype=float)
    pred = np.empty(max(n_targets * n_points, 1))
    grad = np.empty(max(len(wrt), 1))
    lam0 = np.empty(max(n_y, 1))
    seg_t_end = np.array([times[-1]])
    lib = _library()
    with _LOCK:
        lib.mxn_adjoint_set_fns(*c.pointers)
        rc = lib.mxn_adjoint_vjp(
            n_y, y0.ctypes.data_as(_DP), len(pars), pars.ctypes.data_as(_DP),
            len(c.theta), _ints(c.theta),
            n_targets, _ints(c.target_kind), _ints(c.target_index), 0, _ints([]),
            1, _ints([_TIME_COURSE]), _ints([1]), _ints([n_points]), 1,
            seg_t_end.ctypes.data_as(_DP), np.zeros(1).ctypes.data_as(_DP),
            n_points, np.ascontiguousarray(times).ctypes.data_as(_DP),
            cot.ctypes.data_as(_DP), c.n_derived, solver, rtol, atol, 1e-6, 1e6,
            pred.ctypes.data_as(_DP), grad.ctypes.data_as(_DP), lam0.ctypes.data_as(_DP),
        )  # fmt: skip
    if rc in _SETUP_ERRORS:
        msg = f"the compiled adjoint couldn't start ({_SETUP_ERRORS[rc]})"
        raise NativeUnsupported(msg)
    if rc != 0:
        return IntegrationFailure()
    simulated = pred[: n_targets * n_points].reshape(n_targets, n_points).T
    return simulated, grad[: len(wrt)], lam0[:n_y]
