# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Levenberg-Marquardt fits running entirely in compiled code.

The model, its forward sensitivities (`dS/dt = J S + df/dtheta`, by forward-mode
differentiation of the lowered model) and the derived targets with their gradients
are compiled by the Native integrator's LLVM backend. mxlpy-native's `mxn_fit_*`
then runs cminpack's `lmder` over them, with one augmented solve per Jacobian.
Problems the compiled loop can't express fall back to the scipy backend.

The backend of `LeastSquaresMinimizer(method="lm", backend="native")`.
"""

from __future__ import annotations

import ctypes as ct
import functools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy.fit._gradient import GradientResidual, _TransformedGradient
from mxlpy.fit.callbacks import untracked
from mxlpy.fit.penalties import _Penalised
from mxlpy.fit.residuals import _apply_updates
from mxlpy.integrators._native import differentiate
from mxlpy.minimizers._scipy import LeastSquaresMinimizer
from mxlpy.minimizers.abstract import OptimisationState
from mxlpy.symbolic.jacobian import all_parameter_values
from mxlpy.types import FitFailure, Result

if TYPE_CHECKING:
    from mxlpy.integrators._native.lower import Lowered
    from mxlpy.minimizers.abstract import Bounds, Residual
    from mxlpy.types import Array

__all__ = ["least_squares"]

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()

_DP = ct.POINTER(ct.c_double)
_IP = ct.POINTER(ct.c_int)
_SOLVERS = {"Radau": 0, "LSODA": 3}
_TARGET_STATE, _TARGET_DERIVED = 0, 1
_TIME_COURSE, _STEADY_STATE = 0, 1
_MAXFEV_CHUNK = 100
# MINPACK info codes: 1-4 converged, 5 out of evaluations
_CONVERGED = range(1, 5)
_OUT_OF_EVALUATIONS = 5
_CHECK_RTOL = 1e-5


class _Unsupported(Exception):
    """The compiled loop can't run this fit, so it falls back to scipy."""


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOGGER.warning(msg)


###############################################################################
# The fit problem
###############################################################################


@dataclass
class _Problem:
    """Everything mxn_fit_init needs, plus the compiled functions to keep alive."""

    y0: Array
    pars: Array
    theta: list[int]
    log: list[int]
    target_kind: list[int]
    target_index: list[int]
    target_scale: list[float]
    ovr_idx: list[int]
    exp_kind: int
    seg_t_end: list[float]
    seg_ovr: Array
    data_t: Array
    data_y: Array
    n_derived: int
    compiled: list[Any] = field(default_factory=list)
    pointers: list[int] = field(default_factory=list)


def _unwrap(residual_fn: Residual) -> tuple[GradientResidual, set[str]]:
    if isinstance(residual_fn, _TransformedGradient) and isinstance(
        residual_fn.inner, GradientResidual
    ):
        return residual_fn.inner, set(residual_fn.log)
    if isinstance(residual_fn, GradientResidual):
        return residual_fn, set()
    msg = "The native backend needs residual vectors, fit with one of the residual functions of mxlpy.fit.residuals"
    raise TypeError(msg)


def _check_tangents(
    compiled: Any, plain: Any, low: Lowered, *, theta: list[int], pars: Array, y0: Array
) -> None:
    """Compare the compiled sensitivities with central differences of the model."""
    rng = np.random.default_rng(0)
    n, m = low.n, len(theta)
    y = np.where(y0 != 0, y0, 1.0) * rng.uniform(0.5, 1.5, n)
    s = rng.uniform(-1, 1, n * m)
    out = compiled.rhs(0.0, np.concatenate([y, s]), pars)
    for j, k in enumerate(theta):
        h = 1e-6 * max(1.0, float(np.abs(y).max()), abs(pars[k]))
        e = np.zeros_like(pars)
        e[k] = h
        sj = s[j * n : (j + 1) * n]
        up = plain.rhs(0.0, y + h * sj, pars + e)
        down = plain.rhs(0.0, y - h * sj, pars - e)
        fd = (up - down) / (2 * h)
        got = out[n + j * n : n + (j + 1) * n]
        scale = _CHECK_RTOL * max(1.0, float(np.abs(fd).max()))
        if not np.allclose(got, fd, rtol=_CHECK_RTOL, atol=scale):
            msg = "compiled sensitivities disagree with the model"
            raise _Unsupported(msg)


def _check_gradients(
    gradient_fn: Any,
    derived_fn: Any,
    *,
    n: int,
    n_derived: int,
    theta: list[int],
    pars: Array,
    y0: Array,
) -> None:
    """Compare the compiled derived-target gradients with central differences.

    At y0 every entry the differences can evaluate must agree, at a random
    state around it the finite ones (see `checked_entries`). Uses `eval_rows`
    with explicit output sizes: `rhs` sizes its output like the state, but
    these functions have other numbers of outputs.
    """
    from mxlpy.integrators.int_native import (  # noqa: PLC0415
        check_states,
        check_step,
        checked_entries,
    )

    m = len(theta)
    t = np.zeros(1)

    def derived(state: Array, parameters: Array) -> Array:
        return derived_fn.eval_rows(t, state[None, :], parameters, n_derived)[0]

    for i, y in enumerate(check_states(np.asarray(y0, dtype=float), 1, seed=1)):
        grads = gradient_fn.eval_rows(t, y[None, :], pars, n_derived * (n + m))[0]
        for col in range(n + m):
            dy, dp = np.zeros(n), np.zeros_like(pars)
            if col < n:
                h = check_step(float(y[col]))
                dy[col] = h
            else:
                k = theta[col - n]
                h = check_step(float(pars[k]))
                dp[k] = h
            fd = (derived(y + dy, pars + dp) - derived(y - dy, pars - dp)) / (2 * h)
            # Per derived target: its derivatives along all n + m directions
            got = grads[col :: n + m]
            mask = checked_entries(got, fd, strict=i == 0)
            if mask.any() and not np.allclose(
                got[mask],
                fd[mask],
                rtol=_CHECK_RTOL,
                atol=_CHECK_RTOL * max(1.0, float(np.abs(fd[mask]).max())),
            ):
                msg = "compiled derived-target gradients disagree with the model"
                raise _Unsupported(msg)


def _compile_derived(
    model: Any,
    derived: list[str],
    theta: list[int],
    *,
    n: int,
    pars: Array,
    y0: Array,
    backend: Any = None,
) -> tuple[Any, Any]:
    """Compile derived targets and their gradients [d/dy, d/dtheta], checked numerically.

    Shared by the compiled least-squares loop and the compiled adjoint.
    `backend` is the backend module, the LLVM one by default.
    """
    from mxlpy.integrators._native import llvm_backend  # noqa: PLC0415

    backend = llvm_backend if backend is None else backend
    from mxlpy.integrators._native.lower import (  # noqa: PLC0415
        NativeLoweringError,
        lower_outputs,
    )

    try:
        outputs = lower_outputs(model, derived)
    except NativeLoweringError as e:
        raise _Unsupported(str(e)) from e
    if outputs.arrays or outputs.data:
        msg = "derived targets use arrays or data"
        raise _Unsupported(msg)
    try:
        derived_fn = backend.compile_model(outputs)
        gradient_fn = backend.compile_model(differentiate.gradients(outputs, theta))
    except Exception as e:
        msg = f"the derived targets can't be compiled ({e})"
        raise _Unsupported(msg) from e
    _check_gradients(
        gradient_fn,
        derived_fn,
        n=n,
        n_derived=len(derived),
        theta=theta,
        pars=pars,
        y0=y0,
    )
    return derived_fn, gradient_fn


def _build(
    residual: GradientResidual, values: dict[str, float], log: set[str]
) -> _Problem:
    from mxlpy.integrators._native.backends import problem  # noqa: PLC0415

    if (reason := problem("llvm")) is not None:
        raise _Unsupported(reason)
    from mxlpy.integrators._native import llvm_backend  # noqa: PLC0415
    from mxlpy.integrators._native.lower import (  # noqa: PLC0415
        NativeLoweringError,
        lower,
    )

    s = residual.settings
    model = s.model
    if not isinstance(model, (KineticModelBuilder, OdeModelBuilder)):
        msg = "the model has no dynamics"
        raise _Unsupported(msg)
    if s.v_names:
        msg = "initial values are fitted"
        raise _Unsupported(msg)
    if s.postprocess_fn is not None:
        msg = "a postprocess_fn can't be compiled"
        raise _Unsupported(msg)
    if getattr(model, "_events", None):
        msg = "the model has events"
        raise _Unsupported(msg)

    _apply_updates(values, s)
    try:
        low = lower(model)
    except NativeLoweringError as e:
        raise _Unsupported(str(e)) from e
    if low.arrays or low.data:
        msg = "the model uses arrays or data"
        raise _Unsupported(msg)

    var_names = list(model.get_variable_names())
    y0 = np.array([model.get_initial_conditions()[v] for v in var_names], dtype=float)

    def parameter_vector() -> Array:
        # Includes parameters set by initial assignments
        all_values = all_parameter_values(model)
        return np.array([all_values[p] for p in low.par_names] or [0.0], dtype=float)

    pars = parameter_vector()
    if unknown := [k for k in values if k not in low.par_names]:
        msg = f"fitted parameters {unknown} aren't model parameters"
        raise _Unsupported(msg)
    theta = [low.par_names.index(k) for k in values]

    # Sensitivities start at zero, so initial values mustn't depend on fitted
    # parameters. The compiled loop only changes the fitted entries of the
    # parameter vector, so no other parameter may be assigned from them either.
    others = np.ones(len(pars), dtype=bool)
    others[theta] = False
    try:
        for k, v in values.items():
            _apply_updates(values | {k: v * 1.01 + 1e-3}, s)
            moved = np.array(
                [model.get_initial_conditions()[x] for x in var_names], dtype=float
            )
            if not np.allclose(moved, y0):
                msg = f"initial values depend on the fitted parameter {k}"
                raise _Unsupported(msg)
            if not np.allclose(parameter_vector()[others], pars[others]):
                msg = f"initial assignments of parameters depend on the fitted parameter {k}"
                raise _Unsupported(msg)
    finally:
        _apply_updates(values, s)

    data = s.data
    names = residual.names
    derived = [n for n in names if n not in var_names]
    target_kind = [_TARGET_STATE if n in var_names else _TARGET_DERIVED for n in names]
    target_index = [
        var_names.index(n) if n in var_names else derived.index(n) for n in names
    ]

    raw = data.to_frame().T if isinstance(data, pd.Series) else data
    values_y = raw.to_numpy(dtype=float).copy()
    if residual.scaled:
        # A Series' scale is one number, a DataFrame's one per column
        scale = (
            np.full(len(names), float(s.scale))
            if np.isscalar(s.scale)
            else np.asarray(pd.Series(s.scale).reindex(names), dtype=float)
        )
        # Constant columns are skipped by the losses, so they are missing data here
        values_y[:, ~(np.isfinite(scale) & (scale > 0))] = np.nan
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    elif isinstance(sigma := getattr(s.loss_fn, "sigma", None), pd.DataFrame):
        msg = "sigma per data point can't be compiled"
        raise _Unsupported(msg)
    elif residual.weights is not None:
        # GaussianNll: residuals weighted by 1 / sigma per observable
        scale = (
            np.asarray(sigma.reindex(names), dtype=float)
            if isinstance(sigma, pd.Series)
            else np.full(len(names), float(cast(float, sigma)))
        )
    else:
        scale = np.ones(len(names))

    ovr_idx: list[int] = []
    seg_ovr = np.zeros(0)
    if residual.kind == "steady_state":
        exp_kind, data_t, seg_t_end = _STEADY_STATE, np.zeros(1), [0.0]
    else:
        exp_kind = _TIME_COURSE
        data_t = np.asarray(raw.index, dtype=float)
        if (data_t < 0).any() or (np.diff(data_t) < 0).any():
            msg = "data time points must be ascending and non-negative"
            raise _Unsupported(msg)
        if residual.kind == "time_course":
            seg_t_end = [float(data_t.max())]
        else:
            from mxlpy.simulator import _normalise_protocol_index  # noqa: PLC0415

            protocol = _normalise_protocol_index(cast(pd.DataFrame, s.protocol))
            ends = cast(pd.TimedeltaIndex, protocol.index).total_seconds()
            seg_t_end = [float(t) for t in ends]
            if unknown := [c for c in protocol.columns if c not in low.par_names]:
                msg = f"protocol columns {unknown} aren't model parameters"
                raise _Unsupported(msg)
            if set(protocol.columns) & set(values):
                msg = "a fitted parameter is part of the protocol"
                raise _Unsupported(msg)
            ovr_idx = [low.par_names.index(c) for c in protocol.columns]
            seg_ovr = protocol.to_numpy(dtype=float).ravel()
            # The simulation ignores time points after the protocol
            keep = data_t <= seg_t_end[-1] + 1e-9 * max(1.0, seg_t_end[-1])
            data_t, values_y = data_t[keep], values_y[keep]

    try:
        plain = llvm_backend.compile_model(low)
        augmented = llvm_backend.compile_model(differentiate.augmented(low, theta))
    except Exception as e:
        msg = f"the sensitivities can't be compiled ({e})"
        raise _Unsupported(msg) from e
    _check_tangents(augmented, plain, low, theta=theta, pars=pars, y0=y0)
    compiled = [plain, augmented]
    pointers = [plain.address, augmented.address, 0, 0]
    if derived:
        derived_fn, gradient_fn = _compile_derived(
            model, derived, theta, n=low.n, pars=pars, y0=y0
        )
        compiled += [derived_fn, gradient_fn]
        pointers[2:] = [derived_fn.address, gradient_fn.address]

    return _Problem(
        y0=y0,
        pars=pars,
        theta=theta,
        log=[int(k in log) for k in values],
        target_kind=target_kind,
        target_index=target_index,
        target_scale=[float(x) for x in scale],
        ovr_idx=ovr_idx,
        exp_kind=exp_kind,
        seg_t_end=seg_t_end,
        seg_ovr=seg_ovr,
        data_t=data_t,
        # Target-major: all points of the first target, then the second, ...
        data_y=values_y.T.ravel(),
        n_derived=len(derived),
        compiled=compiled,
        pointers=pointers,
    )


def _library() -> ct.CDLL:
    import mxlpy_native  # noqa: PLC0415

    lib = mxlpy_native.load()
    lib.mxn_fit_set_fns.argtypes = [ct.c_void_p] * 4
    lib.mxn_fit_set_fns.restype = None
    lib.mxn_fit_init.restype = ct.c_int
    lib.mxn_fit_init.argtypes = [
        ct.c_int, _DP, ct.c_int, _DP, ct.c_int, _IP, _IP,
        ct.c_int, _IP, _IP, _DP,
        ct.c_int, _IP, ct.c_int, _IP,
        _IP, _IP, ct.c_int, _DP, _DP,
        ct.c_int, _DP, _DP, ct.c_int, ct.c_int,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    ]  # fmt: skip
    lib.mxn_fit_chunk.restype = ct.c_int
    lib.mxn_fit_chunk.argtypes = [ct.c_int]
    lib.mxn_fit_params.restype = None
    lib.mxn_fit_params.argtypes = [_DP]
    lib.mxn_fit_free.restype = None
    return lib


def _ints(values: list[int]) -> Any:
    return (ct.c_int * max(len(values), 1))(*values)


def _doubles(values: Array | list[float]) -> Any:
    values = np.ascontiguousarray(values, dtype=float)
    return (ct.c_double * max(len(values), 1))(*values)


def _run(
    p: _Problem, *, solver: int, rtol: float, atol: float, ss_tol: float, max_nfev: int
) -> tuple[int, Array]:
    from mxlpy.integrators.int_native import _LOCK  # noqa: PLC0415

    lib = _library()
    n_targets, n_points = len(p.target_kind), len(p.data_t)
    with _LOCK:
        lib.mxn_fit_set_fns(*p.pointers)
        rc = lib.mxn_fit_init(
            len(p.y0), _doubles(p.y0), len(p.pars), _doubles(p.pars),
            len(p.theta), _ints(p.theta), _ints(p.log),
            n_targets, _ints(p.target_kind), _ints(p.target_index), _doubles(p.target_scale),
            len(p.ovr_idx), _ints(p.ovr_idx), 1, _ints([p.exp_kind]),
            _ints([len(p.seg_t_end)]), _ints([n_points]), len(p.seg_t_end),
            _doubles(p.seg_t_end), _doubles(p.seg_ovr),
            n_points, _doubles(p.data_t), _doubles(p.data_y), p.n_derived, solver,
            rtol, atol, ss_tol, 1e6,
        )  # fmt: skip
        if rc != 0:
            lib.mxn_fit_free()
            msg = f"the compiled fit couldn't start (code {rc})"
            raise _Unsupported(msg)
        try:
            info, used = _OUT_OF_EVALUATIONS, 0
            while info == _OUT_OF_EVALUATIONS and used < max_nfev:
                info = lib.mxn_fit_chunk(min(_MAXFEV_CHUNK, max_nfev - used))
                used += _MAXFEV_CHUNK
            out = (ct.c_double * len(p.pars))()
            lib.mxn_fit_params(out)
        finally:
            lib.mxn_fit_free()
    return int(info), np.array(out[:], dtype=float)


###############################################################################
# Entry point
###############################################################################


def _solver_settings(runner: object) -> tuple[int, float, float]:
    """Solver and tolerances of the runner's Native integrator, else its defaults."""
    from mxlpy.integrators.int_native import Native  # noqa: PLC0415

    integrator = getattr(runner, "integrator", None)
    kwargs: dict[str, Any] = {}
    while isinstance(integrator, functools.partial):
        kwargs = integrator.keywords | kwargs
        integrator = integrator.func
    method = kwargs.get("method", Native.method)
    if method not in _SOLVERS:
        msg = f"the compiled loop supports the methods {list(_SOLVERS)}, not {method!r}"
        raise _Unsupported(msg)
    return (
        _SOLVERS[method],
        float(kwargs.get("rtol", Native.rtol)),
        float(kwargs.get("atol", Native.atol)),
    )


def least_squares(
    residual_fn: Residual,
    p0: dict[str, float],
    bounds: Bounds,
    *,
    max_nfev: int,
) -> Result[OptimisationState]:
    """Levenberg-Marquardt least squares in compiled code.

    The backend of `LeastSquaresMinimizer(method="lm", backend="native")`.
    Integrates with the method and tolerances of the runner's `Native`
    integrator (Radau or LSODA), or Native's defaults for other integrators.
    Fits the compiled loop can't express fall back to the scipy backend with a
    one-time warning: models that don't lower or use arrays, events, a
    `postprocess_fn`, fitted initial values or initial values depending on
    fitted parameters, the wasm backend.
    """
    # Progress tracking can't see into the compiled loop, only the fallbacks
    tracked_fn, residual_fn = residual_fn, untracked(residual_fn)
    if isinstance(residual_fn, _Penalised) or isinstance(
        getattr(residual_fn, "inner", None), _Penalised
    ):
        _warn_once(
            "Fitting with scipy instead of the compiled loop, as priors and penalties can't be compiled"
        )
        return LeastSquaresMinimizer(method="lm", max_nfev=max_nfev)(
            tracked_fn, p0, bounds
        )
    try:
        residual, log = _unwrap(residual_fn)
    except TypeError:
        if not callable(getattr(residual_fn, "residuals", None)):
            raise
        # e.g. joint fits, whose residual vectors span several models
        _warn_once(
            "Fitting with scipy instead of the compiled loop, as it only runs single fits"
        )
        return LeastSquaresMinimizer(method="lm", max_nfev=max_nfev)(
            tracked_fn, p0, bounds
        )
    to_inner = getattr(residual_fn, "to_inner", lambda x: x)
    values = cast(dict[str, float], to_inner(p0))
    runner = residual.settings.runner
    try:
        solver, rtol, atol = _solver_settings(runner)
        problem = _build(residual, values, log)
        info, pars = _run(
            problem,
            solver=solver,
            rtol=rtol,
            atol=atol,
            ss_tol=float(getattr(runner, "tolerance", 1e-6)),
            max_nfev=max_nfev,
        )
    except _Unsupported as e:
        _apply_updates(values, residual.settings)
        _warn_once(f"Fitting with scipy instead of the compiled loop, as {e}")
        return LeastSquaresMinimizer(method="lm", max_nfev=max_nfev)(
            tracked_fn, p0, bounds
        )

    fitted = {k: float(pars[i]) for k, i in zip(values, problem.theta, strict=True)}
    # Back to the minimizer's (possibly log10) parameters
    parameters = {k: float(np.log10(v)) if k in log else v for k, v in fitted.items()}
    # Through the tracker, so the callback sees the optimum
    loss = tracked_fn(parameters)
    if info not in _CONVERGED or not np.isfinite(loss):
        msg = f"Compiled least squares stopped with MINPACK info {info}"
        _LOGGER.warning(msg)
        return Result(FitFailure(extra_info=[msg]))
    return Result(OptimisationState(parameters=parameters, residual=float(loss)))
