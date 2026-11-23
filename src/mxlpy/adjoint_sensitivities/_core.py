# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint sensitivities, see the package docstring."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from mxlpy import forward_sensitivities
from mxlpy.adjoint_sensitivities import _native
from mxlpy.forward_sensitivities._core import (
    _chain_factors,
    _check_names,
    _left_null_space,
    _output_names,
    _unwrap,
    _values,
)
from mxlpy.integrators import DefaultIntegrator
from mxlpy.simulator import Simulator
from mxlpy.symbolic.jacobian import (
    all_parameter_values,
    compile_rhs,
    compile_vjp,
    constant_values,
    jacobian_fn,
)
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.symbolic.jacobian import CompiledVjp
    from mxlpy.types import Array, DerivativeMethod, Rhs

    type _Model = KineticModelBuilder | OdeModelBuilder

__all__ = [
    "ADJOINT_THRESHOLD",
    "AdjointSensitivities",
    "steady_state",
    "time_course",
    "uses_adjoint",
]

_LOGGER = logging.getLogger(__name__)

# From this size of the forward-sensitivity system, variables x parameters,
# on `auto` uses adjoints for time courses. Forward sensitivities integrate
# n (1 + m) states with dense n (1 + m) Jacobians, adjoints cost about the same
# for any m. Measured with Sundials on chains with a tanh layer: the crossover
# is at n m ~ 1000 for n = 3, 10 and 30 (e.g. n = 10: 115 parameters 0.17 s vs
# 0.21 s, 430 parameters 0.49 s vs 4.6 s)
ADJOINT_THRESHOLD = 1000

# Refinement rounds of the interpolation grids
_MAX_REFINEMENTS = 8
# Most pieces an interval is split into per round
_MAX_SPLIT = 16
# Nodes between t0 and the end of the time course / before each data point,
# at these fractions of the interval, for fast transients
_GEOMETRIC = 10.0 ** -np.arange(1, 7)
_N_UNIFORM = 20

# Gauss-Legendre nodes on [0, 1]
_GL2 = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
_GL3 = 0.5 + np.array([-1, 0, 1]) * np.sqrt(3 / 5) / 2
_GL3_WEIGHTS = np.array([5, 8, 5]) / 18


@dataclass(slots=True)
class AdjointSensitivities:
    """Gradient of a weighted sum of simulated values.

    Attributes
    ----------
    simulated
        Simulated values at the time points and for the columns of the
        cotangents
    gradient
        For every parameter / variable initial value `theta`, the derivative
        of `sum(cotangents * simulated)` w.r.t. it
    method
        How the gradient was computed
    adjoint
        Adjoint state `λ(t)` (time x variables) on the grid of the backward
        solve, only for `method="adjoint"` (only `λ(0)` from the compiled
        adjoint of `Native`). It jumps by the cotangents at
        their time points, where it holds the value after the jump (the
        limit from earlier times), so `λ(0)` is the gradient w.r.t. the
        initial values

    """

    simulated: pd.DataFrame | pd.Series
    gradient: pd.Series
    method: Literal["adjoint", "forward", "finite_difference"]
    adjoint: pd.DataFrame | None = None


###############################################################################
# Interpolation
###############################################################################


@dataclass
class _Hermite:
    """Cubic Hermite interpolation of a solution through its nodes and slopes."""

    t: Array
    y: Array
    dy: Array
    _i: int = 0

    def _locate(self, t: float) -> int:
        i = self._i
        # Integrators query neighbouring times, so check the last interval first
        if not (self.t[i] <= t <= self.t[i + 1]):
            i = int(np.searchsorted(self.t, t, side="right")) - 1
            i = min(max(i, 0), len(self.t) - 2)
            self._i = i
        return i

    def _basis(self, i: int, s: Array) -> tuple[Array, Array]:
        """Values and slopes at local coordinates `s` (column vector) of interval i."""
        h = self.t[i + 1] - self.t[i]
        y0, y1 = self.y[i], self.y[i + 1]
        d0, d1 = h * self.dy[i], h * self.dy[i + 1]
        s2 = s * s
        s3 = s2 * s
        value = (
            (2 * s3 - 3 * s2 + 1) * y0
            + (s3 - 2 * s2 + s) * d0
            + (-2 * s3 + 3 * s2) * y1
            + (s3 - s2) * d1
        )
        slope = (
            (6 * s2 - 6 * s) * y0
            + (3 * s2 - 4 * s + 1) * d0
            + (-6 * s2 + 6 * s) * y1
            + (3 * s2 - 2 * s) * d1
        ) / h
        return value, slope

    def __call__(self, t: float) -> Array:
        i = self._locate(t)
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        s2 = s * s
        s3 = s2 * s
        return (
            (2 * s3 - 3 * s2 + 1) * self.y[i]
            + (s3 - 2 * s2 + s) * h * self.dy[i]
            + (-2 * s3 + 3 * s2) * self.y[i + 1]
            + (s3 - s2) * h * self.dy[i + 1]
        )

    def at(self, i: int, s: Array) -> Array:
        """Values at local coordinates `s` of interval i, (len(s), n)."""
        return self._basis(i, s[:, None])[0]

    def refinement(
        self,
        f: Callable[[float, Array], Array],
        rtol: float,
        atol: float | Array,
        accepted: set[tuple[float, float]],
    ) -> Array:
        """Nodes to add so that the interpolation error stays within tolerance.

        The error of the cubic is estimated from its defect `y' - f(t, y)` at
        the two points where it's largest: the error is ~ h |defect| / 3.
        Intervals within tolerance are added to `accepted` and not checked
        again in later rounds.
        """
        new: list[Array] = []
        for i in range(len(self.t) - 1):
            a, b = float(self.t[i]), float(self.t[i + 1])
            h = b - a
            if h <= 0 or (a, b) in accepted:
                continue
            scale = atol + rtol * np.maximum(np.abs(self.y[i]), np.abs(self.y[i + 1]))
            values, slopes = self._basis(i, _GL2[:, None])
            with np.errstate(all="ignore"):
                ratio = max(
                    float(np.max(np.abs(slopes[k] - f(a + s * h, values[k])) / scale))
                    for k, s in enumerate(_GL2)
                ) * (h / 3)
            if np.isfinite(ratio) and ratio <= 1:
                accepted.add((a, b))
                continue
            k = int(np.ceil(ratio**0.25)) + 1 if np.isfinite(ratio) else _MAX_SPLIT
            k = min(max(k, 2), _MAX_SPLIT)
            new.append(a + h * np.arange(1, k) / k)
        return np.concatenate(new) if new else np.empty(0)


def _tolerances(integrator: IntegratorType) -> tuple[float, float]:
    """Relative and absolute tolerance of the integrator, used for interpolation."""
    base, kwargs = _unwrap(integrator)
    rtol = kwargs.get("rtol", getattr(base, "rtol", 1e-6))
    atol = kwargs.get("atol", getattr(base, "atol", 1e-8))
    return float(cast(float, rtol)), float(cast(float, atol))


def _merge(grid: Array, new: Array) -> Array:
    out = np.unique(np.concatenate([grid, new]))
    # Drop nodes the integrators can't distinguish
    keep = np.concatenate([[True], np.diff(out) > 1e-12 * max(1.0, abs(out[-1]))])
    return out[keep]


###############################################################################
# Time courses
###############################################################################


@dataclass
class _Problem:
    model: _Model
    integrator: IntegratorType
    backward_integrator: IntegratorType
    p: Array
    c: Array
    vjp_x: CompiledVjp
    vjp_p: CompiledVjp
    vjp_out: CompiledVjp | None
    jac: Callable[[float, Array], Array]
    rtol: float
    atol: float
    compiled_integrator: bool
    f: Callable[[float, Array, Array, Array], Array]
    f_constants: Array

    def rhs(self, t: float, x: Array) -> Array:
        return self.f(t, x, self.p, self.f_constants)

    def x_bar(self, t: float, x: Array, lam: Array) -> Array:
        return self.vjp_x.fn(t, x, self.p, self.c, lam, np.empty(0))[0]

    def p_bar(self, t: float, x: Array, lam: Array) -> Array:
        return self.vjp_p.fn(t, x, self.p, self.c, lam, np.empty(0))[1]


def _forward(prob: _Problem, x0: Array, times: Array) -> _Hermite | Exception:
    """Forward solve on a grid fine enough for Hermite interpolation."""
    end = float(times[-1])
    grid = _merge(
        np.linspace(0.0, end, _N_UNIFORM + 1),
        np.concatenate([times, end * _GEOMETRIC]),
    )
    traj: _Hermite | None = None
    accepted: set[tuple[float, float]] = set()
    for _ in range(_MAX_REFINEMENTS):
        # The compiled rhs is faster than the model, unless the integrator
        # compiles the model itself
        rhs = prob.model if prob.compiled_integrator else cast("Rhs", prob.rhs)
        res = prob.integrator(rhs, tuple(x0), None).integrate_time_course(
            time_points=grid
        )
        if isinstance(tc := res.value, Exception):
            return tc
        t = np.asarray(tc.time, dtype=float)
        x = np.asarray(tc.values, dtype=float)
        if len(t) != len(grid) or not np.isfinite(x).all():
            return ValueError("Forward solve didn't reach all time points")
        traj = _Hermite(
            t, x, np.array([prob.rhs(ti, xi) for ti, xi in zip(t, x, strict=True)])
        )
        new = traj.refinement(prob.rhs, prob.rtol, prob.atol, accepted)
        if len(new) == 0:
            return traj
        grid = _merge(grid, new)
    _LOGGER.debug("Forward interpolation grid not converged with %d nodes", len(grid))
    return cast(_Hermite, traj)


def _backward(
    prob: _Problem, x: _Hermite, t_lo: float, t_hi: float, lam_hi: Array
) -> tuple[_Hermite, Array] | Exception:
    """Adjoint state from t_hi down to t_lo and the parameter quadrature.

    Integrates `dλ/ds = (∂f/∂x)ᵀ λ` in `s = t_hi - t`, so every integrator
    can solve it, on a grid refined until the Hermite interpolation of λ is
    accurate. The quadrature `∫ λᵀ ∂f/∂θ dt` then runs over that grid,
    rather than as part of the solve, so implicit solvers only factor n x n
    matrices, however many parameters there are.
    """
    inside = x.t[(x.t > t_lo) & (x.t < t_hi)]
    nodes = _merge(
        np.array([t_lo, t_hi]),
        np.concatenate([inside, t_hi - (t_hi - t_lo) * _GEOMETRIC]),
    )

    def rhs(s: float, lam: Array) -> Array:
        t = t_hi - s
        return prob.x_bar(t, x(t), np.asarray(lam, dtype=float))

    def jac(s: float, _: Array) -> Array:
        t = t_hi - s
        return prob.jac(t, x(t)).T

    def dlam_dt(t: float, lam: Array) -> Array:
        return -prob.x_bar(t, x(t), lam)

    lam: _Hermite | None = None
    accepted: set[tuple[float, float]] = set()
    for _ in range(_MAX_REFINEMENTS):
        s_nodes = t_hi - nodes[::-1]
        res = prob.backward_integrator(
            cast("Rhs", rhs), tuple(lam_hi), jac
        ).integrate_time_course(time_points=s_nodes)
        if isinstance(tc := res.value, Exception):
            return tc
        values = np.asarray(tc.values, dtype=float)
        if len(values) != len(nodes) or not np.isfinite(values).all():
            return ValueError("Adjoint solve didn't reach all time points")
        y = values[::-1]
        lam = _Hermite(
            nodes, y, np.array([dlam_dt(t, v) for t, v in zip(nodes, y, strict=True)])
        )
        atol = prob.atol + prob.rtol * np.abs(y).max(axis=0, initial=0.0)
        new = lam.refinement(dlam_dt, prob.rtol, atol, accepted)
        if len(new) == 0:
            break
        nodes = _merge(nodes, new)
    lam = cast(_Hermite, lam)

    total = np.zeros(len(prob.vjp_p.wrt))
    for i in range(len(nodes) - 1):
        a, h = nodes[i], nodes[i + 1] - nodes[i]
        lams = lam.at(i, _GL3)
        for k, (s, w) in enumerate(zip(_GL3, _GL3_WEIGHTS, strict=True)):
            t = a + s * h
            total += (w * h) * prob.p_bar(t, x(t), lams[k])
    return lam, total


def _adjoint_time_course(
    model: _Model,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[AdjointSensitivities]:
    """Adjoint gradient, with the cotangents scaled to max |w| = 1.

    λ is linear in the cotangents, but integrated with the absolute tolerance
    of the states, so small cotangents (close to a fit's optimum) would lose
    accuracy.
    """
    scale = float(np.abs(cotangents.to_numpy(dtype=float)).max(initial=0.0))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    res = _scaled_adjoint_time_course(
        model,
        cotangents / scale,
        wrt,
        y0=y0,
        integrator=integrator,
        displacement=displacement,
    )
    if isinstance(val := res.value, Exception):
        return res
    val.gradient = val.gradient * scale
    if val.adjoint is not None:
        val.adjoint = val.adjoint * scale
    return res


def _scaled_adjoint_time_course(
    model: _Model,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[AdjointSensitivities]:
    var_names = model.get_variable_names()
    outputs = [c for c in cotangents.columns if c not in var_names]
    compiled_wrt, chain, s0 = _chain_factors(
        model, wrt, y0_from_model=y0 is None, displacement=displacement
    )
    start = model.get_initial_conditions() if y0 is None else y0
    x0 = np.array([start[k] for k in var_names], dtype=float)
    times = cotangents.index.to_numpy(dtype=float)

    compiled_integrator = bool(getattr(_unwrap(integrator)[0], "compiles_model", False))
    if compiled_integrator and times[-1] > 0:
        try:
            res = _native.vjp(
                model, cotangents, compiled_wrt, x0=x0, integrator=integrator
            )
        except _native.NativeUnsupported as e:
            _LOGGER.debug("Using the Python adjoint, as %s", e)
        else:
            if isinstance(res, Exception):
                return Result(res)
            simulated, theta_bar, lam0 = res
            return Result(
                AdjointSensitivities(
                    simulated=pd.DataFrame(
                        simulated, index=times, columns=list(cotangents.columns)
                    ),
                    gradient=pd.Series(chain.T @ theta_bar + s0.T @ lam0, index=wrt),
                    method="adjoint",
                    adjoint=pd.DataFrame(lam0[None, :], index=[0.0], columns=var_names),
                )
            )

    vjp_out = compile_vjp(model, wrt=compiled_wrt, outputs=outputs) if outputs else None
    vjp_p = compile_vjp(model, wrt=compiled_wrt)
    values = all_parameter_values(model)
    rtol, atol = _tolerances(integrator)
    backward_integrator = (
        functools.partial(DefaultIntegrator, rtol=rtol, atol=atol)
        if compiled_integrator
        else integrator
    )
    f, f_constant_names = compile_rhs(model)
    prob = _Problem(
        model=model,
        integrator=integrator,
        backward_integrator=backward_integrator,
        p=np.array([values[k] for k in model.get_parameter_names()], dtype=float),
        c=constant_values(model, vjp_p.constant_names),
        vjp_x=compile_vjp(model, wrt=[]),
        vjp_p=vjp_p,
        vjp_out=vjp_out,
        jac=jacobian_fn(model),
        rtol=rtol,
        atol=atol,
        compiled_integrator=compiled_integrator,
        f=f,
        f_constants=constant_values(model, f_constant_names),
    )

    if times[-1] > 0:
        if isinstance(x := _forward(prob, x0, times), Exception):
            return Result(x)
    else:
        # Only the initial values: nothing to integrate
        x = _Hermite(np.array([0.0, 1.0]), np.vstack([x0, x0]), np.zeros((2, len(x0))))

    w = cotangents.reindex(columns=var_names, fill_value=0.0).to_numpy(dtype=float)
    w_out = cotangents.loc[:, outputs].to_numpy(dtype=float)
    n = len(var_names)
    lam = np.zeros(n)
    theta_bar = np.zeros(len(compiled_wrt))
    pieces: list[pd.DataFrame] = []
    t_hi = float(times[-1])
    for i in range(len(times), -1, -1):
        t_lo = float(times[i - 1]) if i > 0 else 0.0
        if t_hi > t_lo:
            if isinstance(res := _backward(prob, x, t_lo, t_hi, lam), Exception):
                return Result(res)
            traj, quadrature = res
            theta_bar += quadrature
            lam = traj.y[0].copy()
            pieces.append(pd.DataFrame(traj.y, index=traj.t, columns=var_names))
        if i > 0:
            lam = lam + w[i - 1]
            if vjp_out is not None:
                x_bar, p_bar = vjp_out.fn(
                    t_lo, x(t_lo), prob.p, prob.c, np.zeros(n), w_out[i - 1]
                )
                lam = lam + x_bar
                theta_bar += p_bar
        t_hi = t_lo

    gradient = chain.T @ theta_bar + s0.T @ lam
    # At data times the value after their jump, i.e. the limit from below
    adjoint = (
        pd.concat(pieces[::-1]).groupby(level=0).first()
        if pieces
        else pd.DataFrame(columns=var_names, dtype=float)
    )
    adjoint.loc[0.0] = lam
    adjoint = adjoint.sort_index()
    states = pd.DataFrame(
        np.array([x(t) for t in times]), index=times, columns=var_names
    )
    simulated = _values(model, states).loc[:, list(cotangents.columns)]
    return Result(
        AdjointSensitivities(
            simulated=simulated,
            gradient=pd.Series(gradient, index=wrt),
            method="adjoint",
            adjoint=adjoint,
        )
    )


###############################################################################
# Shared helpers
###############################################################################


def _check_columns(model: _Model, columns: list[str]) -> None:
    available = model.get_variable_names() + _output_names(model)
    if unknown := [c for c in columns if c not in available]:
        msg = f"Unknown cotangent columns {unknown}, choose from {available}"
        raise ValueError(msg)


def _contract(
    sens: forward_sensitivities.ForwardSensitivities,
    cotangents: pd.DataFrame | pd.Series,
    wrt: list[str],
) -> AdjointSensitivities:
    """Gradient from forward sensitivities: Σ cotangent · dy/dθ."""
    if isinstance(cotangents, pd.Series):
        columns = list(cotangents.index)
        simulated: pd.DataFrame | pd.Series = sens.simulated.iloc[-1].loc[columns]
        gradient = [
            float((sens.derivatives[k].iloc[-1].loc[columns] * cotangents).sum())
            for k in wrt
        ]
    else:
        columns = list(cotangents.columns)
        simulated = sens.simulated.loc[:, columns]
        w = cotangents.to_numpy(dtype=float)
        gradient = [
            float(np.sum(sens.derivatives[k].loc[:, columns].to_numpy(dtype=float) * w))
            for k in wrt
        ]
    return AdjointSensitivities(
        simulated=simulated,
        gradient=pd.Series(gradient, index=wrt, dtype=float),
        method=sens.method,
    )


###############################################################################
# Public API
###############################################################################


def uses_adjoint(
    model: _Model, n_wrt: int, integrator: IntegratorType | None = None
) -> bool:
    """Whether `method="auto"` uses adjoints for a time course.

    Always with compiled adjoints (`Native`, LLVM backend, Radau or LSODA),
    which can't integrate forward sensitivities either, so these would fall
    back to finite differences. Otherwise from `ADJOINT_THRESHOLD` variables x
    parameters / initial values on.
    """
    if integrator is not None and _native.supports(integrator):
        return True
    return len(model.get_variable_names()) * n_wrt >= ADJOINT_THRESHOLD


def time_course(
    model: _Model,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
) -> Result[AdjointSensitivities]:
    """Gradient of `sum(cotangents * y(t))` w.r.t. parameters and initial values.

    This is the vector-Jacobian product of a time course, e.g. the gradient
    of a loss with `cotangents = d loss / d y(t)`. With adjoints it costs one
    forward and one backward solve, however many parameters there are.

    Examples
    --------
        >>> w = pd.DataFrame({"x": [1.0, -1.0]}, index=[1.0, 2.0])
        >>> adjoint_sensitivities.time_course(model, w, ["k1", "k2"]).unwrap_or_err().gradient

    Parameters
    ----------
    model
        Model to simulate
    cotangents
        Weight of every simulated value (time points x variables, derived
        quantities and reactions), NaN for none. The time points must be
        ascending, distinct and non-negative
    wrt
        Parameters and variables (for their initial values) to differentiate w.r.t.
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator. `Native` (LLVM
        backend, Radau or LSODA) runs both solves in compiled code; for models
        it can't compile, or the wasm backend, it solves the forward pass and
        the default integrator the backward one
    method
        `adjoint` solves the adjoint equations backwards in time, `forward`
        and `finite_difference` contract forward sensitivities (see
        `forward_sensitivities`), `auto` uses adjoints for large problems
        and `Native` (see `uses_adjoint`) and forward sensitivities otherwise
        or if adjoints aren't possible (models with events)
    displacement
        Relative step of the finite differences of initial assignments

    Returns
    -------
    Result[AdjointSensitivities]
        Gradient and the simulated values at the cotangents' time points

    """
    integrator = DefaultIntegrator if integrator is None else integrator
    _check_names(model, wrt)
    _check_columns(model, list(cotangents.columns))
    cotangents = cotangents.fillna(0.0)
    times = cotangents.index.to_numpy(dtype=float)
    if len(times) == 0 or (times < 0).any() or (np.diff(times) <= 0).any():
        msg = "Cotangent time points must be ascending, distinct and non-negative"
        raise ValueError(msg)

    adjoint = method == "adjoint" or (
        method == "auto" and uses_adjoint(model, len(wrt), integrator)
    )
    if adjoint and getattr(model, "_events", None):
        if method == "adjoint":
            msg = "Adjoint sensitivities aren't supported for models with events"
            raise ValueError(msg)
        adjoint = False
    if adjoint:
        try:
            return _adjoint_time_course(
                model,
                cotangents,
                wrt,
                y0=y0,
                integrator=integrator,
                displacement=displacement,
            )
        except Exception as e:
            if method == "adjoint":
                raise
            _LOGGER.debug("Using forward sensitivities instead of adjoints: %s", e)

    outputs = [c for c in cotangents.columns if c not in model.get_variable_names()]
    res = forward_sensitivities.time_course(
        model,
        times,
        wrt,
        y0=y0,
        integrator=integrator,
        outputs=outputs,
        method="auto" if method in ("auto", "adjoint") else method,
        displacement=displacement,
    )
    if isinstance(sens := res.value, Exception):
        return Result(sens)
    return Result(_contract(sens, cotangents, wrt))


def steady_state(
    model: _Model,
    cotangents: pd.Series,
    wrt: list[str],
    *,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
    tolerance: float = 1e-6,
    rel_norm: bool = False,
) -> Result[AdjointSensitivities]:
    """Gradient of `sum(cotangents * y*)` of the steady state `y*`.

    With adjoints, one linear solve `Jᵀλ = cotangents` replaces one per
    parameter (implicit function theorem). Conserved quantities, which make
    `J` singular, are fixed by the initial values instead.

    Examples
    --------
        >>> w = pd.Series({"x": 1.0})
        >>> adjoint_sensitivities.steady_state(model, w, ["k1", "k2"]).unwrap_or_err().gradient

    Parameters
    ----------
    model
        Model to simulate
    cotangents
        Weight of every steady-state value (variables, derived quantities and
        reactions), NaN for none
    wrt
        Parameters and variables (for their initial values) to differentiate
        w.r.t. Initial values only matter for conserved quantities
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator
    method
        `adjoint` solves the adjoint system, `forward` and `finite_difference`
        contract forward sensitivities, `auto` uses adjoints unless the model
        can't be differentiated symbolically
    displacement
        Relative step of the finite differences of initial assignments
    tolerance
        Steady-state tolerance
    rel_norm
        Whether to use the relative norm for the steady-state check

    Returns
    -------
    Result[AdjointSensitivities]
        Gradient and the steady-state values

    """
    integrator = DefaultIntegrator if integrator is None else integrator
    _check_names(model, wrt)
    _check_columns(model, list(cotangents.index))
    cotangents = cotangents.fillna(0.0)
    if method in ("auto", "adjoint"):
        try:
            return _adjoint_steady_state(
                model,
                cotangents,
                wrt,
                y0=y0,
                integrator=integrator,
                displacement=displacement,
                tolerance=tolerance,
                rel_norm=rel_norm,
            )
        except Exception as e:
            if method == "adjoint":
                raise
            _LOGGER.debug("Using forward sensitivities instead of adjoints: %s", e)

    outputs = [str(c) for c in cotangents.index if c not in model.get_variable_names()]
    res = forward_sensitivities.steady_state(
        model,
        wrt,
        y0=y0,
        integrator=integrator,
        outputs=outputs,
        method="auto" if method in ("auto", "adjoint") else method,
        displacement=displacement,
        tolerance=tolerance,
        rel_norm=rel_norm,
    )
    if isinstance(sens := res.value, Exception):
        return Result(sens)
    return Result(_contract(sens, cotangents, wrt))


def _adjoint_steady_state(
    model: _Model,
    cotangents: pd.Series,
    wrt: list[str],
    *,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
    tolerance: float,
    rel_norm: bool,
) -> Result[AdjointSensitivities]:
    """Adjoint of the implicit function theorem.

    The forward sensitivities solve `A s = b` with `A = [J; N]`, `b =
    [-∂f/∂θ; N s0]` for the left null space `N` of `J`. So `wᵀ s = μᵀ b`
    with `Aᵀ μ = w` (minimum-norm least squares, as `A` has more rows than
    columns), one solve for all parameters.
    """
    if getattr(model, "_events", None):
        msg = "Adjoint sensitivities aren't supported for models with events"
        raise ValueError(msg)
    var_names = model.get_variable_names()
    outputs = [str(c) for c in cotangents.index if c not in var_names]
    compiled_wrt, chain, s0 = _chain_factors(
        model, wrt, y0_from_model=y0 is None, displacement=displacement
    )
    vjp = compile_vjp(model, wrt=compiled_wrt, outputs=outputs)
    jac = jacobian_fn(model)

    start = model.get_initial_conditions() if y0 is None else y0
    res = (
        Simulator(model, y0=start, integrator=integrator)
        .simulate_to_steady_state(tolerance=tolerance, rel_norm=rel_norm)
        .get_result()
    )
    if isinstance(sim := res.value, Exception):
        return Result(sim)
    variables = sim.get_variables(
        include_derived_variables=False,
        include_surrogate_variables=False,
        include_readouts=False,
    )
    variables = variables.loc[:, var_names].iloc[[-1]]
    t = float(variables.index[0])
    x = variables.to_numpy(dtype=float)[0]

    values = all_parameter_values(model)
    p = np.array([values[k] for k in model.get_parameter_names()], dtype=float)
    c = constant_values(model, vjp.constant_names)
    n = len(var_names)
    w = cotangents.reindex(var_names, fill_value=0.0).to_numpy(dtype=float)
    w_out = cotangents.loc[outputs].to_numpy(dtype=float)
    x_out, p_out = vjp.fn(t, x, p, c, np.zeros(n), w_out)
    w = w + x_out

    j = jac(t, x)
    left_null = _left_null_space(j)
    a = np.vstack([j, left_null])
    mu = np.linalg.lstsq(a.T, w, rcond=None)[0]
    mu_j, mu_n = mu[:n], mu[n:]
    _, p_bar = vjp.fn(t, x, p, c, mu_j, np.zeros(len(outputs)))
    theta_bar = p_out - p_bar
    gradient = chain.T @ theta_bar + s0.T @ (left_null.T @ mu_n)

    simulated = _values(model, variables).iloc[-1].loc[list(cotangents.index)]
    return Result(
        AdjointSensitivities(
            simulated=simulated,
            gradient=pd.Series(gradient, index=wrt, dtype=float),
            method="adjoint",
            adjoint=pd.DataFrame(mu_j[None, :], index=[t], columns=var_names),
        )
    )
