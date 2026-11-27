# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint sensitivities, see the package docstring."""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from mxlpy import forward_sensitivities
from mxlpy.adjoint_sensitivities import _discrete, _native
from mxlpy.forward_sensitivities._core import (
    _assigned_parameters,
    _chain_factors,
    _check_names,
    _computed_from,
    _fd,
    _left_null_space,
    _output_names,
    _raw_variables,
    _select_times,
    _unwrap,
    _values,
)
from mxlpy.integrators import DefaultIntegrator
from mxlpy.simulator import Simulator, _normalise_protocol_index
from mxlpy.symbolic.jacobian import (
    all_parameter_values,
    compile_rhs,
    compile_vjp,
    constant_values,
    jacobian_fn,
)
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.symbolic.jacobian import CompiledVjp
    from mxlpy.types import Array, ArrayLike, DerivativeMethod, Rhs

    type _Model = KineticModelBuilder | OdeModelBuilder

__all__ = [
    "ADJOINT_THRESHOLD",
    "PROTOCOL_ADJOINT_THRESHOLD",
    "AdjointSensitivities",
    "Pullback",
    "protocol_time_course",
    "protocol_time_course_pullback",
    "steady_state",
    "steady_state_pullback",
    "time_course",
    "time_course_pullback",
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
# From this many parameters / initial values on, `auto` uses adjoints for
# protocol time courses instead of finite differences (2 simulations per
# parameter). Measured on chains with 3 and 10 states over a 6-step protocol:
# the adjoint costs 65-110 ms for any number of parameters, finite differences
# 23-48 ms for one parameter and 85-142 ms for four
PROTOCOL_ADJOINT_THRESHOLD = 3

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
        How the gradient was computed, see `Pullback.method`
    adjoint
        Adjoint state `λ(t)` (time x variables) on the grid of the backward
        solve of the Python continuous adjoint, only `λ(0)` from the compiled
        adjoints of `Native` and at the steady state for steady states; None
        for forward sensitivities. It jumps by the cotangents at their time
        points, where it holds the value after the jump (the limit from
        earlier times), so `λ(0)` is the gradient w.r.t. the initial values

    """

    simulated: pd.DataFrame | pd.Series
    gradient: pd.Series
    method: Literal[
        "discrete_adjoint",
        "continuous_adjoint",
        "adjoint",
        "forward",
        "finite_difference",
    ]
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
class _Segment:
    """A stretch (start, end] of constant parameters, e.g. a protocol step."""

    start: float
    end: float
    overrides: dict[str, float]
    p: Array
    c: Array
    f_constants: Array


@contextlib.contextmanager
def _overridden(model: _Model, overrides: dict[str, float]) -> Iterator[None]:
    """Apply parameter overrides, restoring the previous values afterwards."""
    if not overrides:
        yield
        return
    values = model.get_addressable_values()
    previous = {k: values[k] for k in overrides}
    model.update_addressable(overrides)
    try:
        yield
    finally:
        model.update_addressable(previous)


def _segments(
    model: _Model,
    protocol: pd.DataFrame | None,
    end: float,
    constant_names: list[str],
    f_constant_names: list[str],
) -> list[_Segment]:
    """Parameter values per protocol step, or one segment up to `end`.

    Like `Simulator.simulate_protocol_time_course`, every step applies its
    row of the protocol, and a time point on a step's end belongs to it.
    Steps of zero length change nothing and are left out, and the last one
    ends at `end`, the last time point.
    """
    names = model.get_addressable_names()

    def segment(start: float, stop: float, overrides: dict[str, float]) -> _Segment:
        with _overridden(model, overrides):
            values = all_parameter_values(model)
            return _Segment(
                start=start,
                end=stop,
                overrides=overrides,
                p=np.array([values[k] for k in names], dtype=float),
                c=constant_values(model, constant_names),
                f_constants=constant_values(model, f_constant_names),
            )

    if protocol is None:
        return [segment(0.0, end, {})]
    out = []
    start = 0.0
    ends = cast(pd.TimedeltaIndex, protocol.index).total_seconds()
    for stop, (_, row) in zip(ends, protocol.iterrows(), strict=True):
        if start >= end:
            break
        if stop > start:
            clipped = min(float(stop), end)
            out.append(segment(start, clipped, cast(dict, row.to_dict())))
            start = clipped
    return out


@dataclass
class _Problem:
    model: _Model
    integrator: IntegratorType
    backward_integrator: IntegratorType
    segments: list[_Segment]
    vjp_x: CompiledVjp
    vjp_p: CompiledVjp
    vjp_out: CompiledVjp | None
    jac: Callable[[float, Array], Array]
    rtol: float
    atol: float
    compiled_integrator: bool
    f: Callable[[float, Array, Array, Array], Array]
    seg: _Segment = field(init=False)

    def __post_init__(self) -> None:
        self.seg = self.segments[0]

    def segment_of(self, t: float) -> int:
        """The segment a time point belongs to (the earlier one at a boundary)."""
        ends = [s.end for s in self.segments]
        return min(int(np.searchsorted(ends, t, side="left")), len(ends) - 1)

    def rhs(self, t: float, x: Array) -> Array:
        return self.f(t, x, self.seg.p, self.seg.f_constants)

    def x_bar(self, t: float, x: Array, lam: Array) -> Array:
        return self.vjp_x.fn(t, x, self.seg.p, self.seg.c, lam, np.empty(0))[0]

    def p_bar(self, t: float, x: Array, lam: Array) -> Array:
        return self.vjp_p.fn(t, x, self.seg.p, self.seg.c, lam, np.empty(0))[1]


def _forward(prob: _Problem, x0: Array, times: Array) -> list[_Hermite] | Exception:
    """Forward solve on grids fine enough for Hermite interpolation, per segment.

    One integrator continues through all segments, as in a protocol
    simulation. The model's parameters only matter (and are overridden per
    segment) for integrators compiling the model, the compiled rhs takes them
    as arguments.
    """
    segments = prob.segments
    end = segments[-1].end
    starts = np.array([s.start for s in segments])
    lengths = np.array([s.end - s.start for s in segments])
    base = _merge(
        np.linspace(0.0, end, _N_UNIFORM + 1),
        np.concatenate(
            [
                times,
                starts + lengths,
                *(a + h * _GEOMETRIC for a, h in zip(starts, lengths, strict=True)),
            ]
        ),
    )
    grids = [
        _merge(np.array([s.start, s.end]), base[(base > s.start) & (base < s.end)])
        for s in segments
    ]
    accepted: list[set[tuple[float, float]]] = [set() for _ in segments]
    trajs: list[_Hermite] = []
    # The compiled rhs is faster than the model, unless the integrator
    # compiles the model itself
    rhs = prob.model if prob.compiled_integrator else cast("Rhs", prob.rhs)
    for _ in range(_MAX_REFINEMENTS):
        integrator = prob.integrator(rhs, tuple(x0), None)
        trajs = []
        for g, segment in enumerate(segments):
            prob.seg = segment
            overrides = segment.overrides if prob.compiled_integrator else {}
            with _overridden(prob.model, overrides):
                res = integrator.integrate_time_course(time_points=grids[g])
            if isinstance(tc := res.value, Exception):
                return tc
            t = np.asarray(tc.time, dtype=float)
            x = np.asarray(tc.values, dtype=float)
            if len(t) != len(grids[g]) or not np.isfinite(x).all():
                return ValueError("Forward solve didn't reach all time points")
            trajs.append(
                _Hermite(
                    t,
                    x,
                    np.array([prob.rhs(ti, xi) for ti, xi in zip(t, x, strict=True)]),
                )
            )
        refined = False
        for g, segment in enumerate(segments):
            prob.seg = segment
            new = trajs[g].refinement(prob.rhs, prob.rtol, prob.atol, accepted[g])
            if len(new) > 0:
                grids[g] = _merge(grids[g], new)
                refined = True
        if not refined:
            return trajs
    _LOGGER.debug("Forward interpolation grids not converged")
    return trajs


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


###############################################################################
# Pullbacks
###############################################################################

type _Backward = Callable[[pd.DataFrame | pd.Series], Result[AdjointSensitivities]]
type _Method = Literal[
    "discrete_adjoint", "continuous_adjoint", "adjoint", "forward", "finite_difference"
]


@dataclass(eq=False)
class Pullback:
    """A simulation and the vector-Jacobian products of its values.

    Calling it with cotangents returns the gradient of
    `sum(cotangents * simulated)` w.r.t. `wrt`, without simulating again: the
    loss of a fit can be evaluated on `simulated` first, and its derivative
    then gives the cotangents (as `jax.vjp`). With `discrete_adjoint` the
    gradient is exactly that of `simulated` as computed (see the package
    docstring). A pullback can be called any number of times.

    Attributes
    ----------
    simulated
        Simulated values: time points x columns for time courses, a Series
        for steady states
    wrt
        Parameters and variables (for their initial values) the gradients are
        w.r.t.
    method
        How gradients are computed: `discrete_adjoint`, `continuous_adjoint`,
        `adjoint` (steady states: the implicit function theorem), `forward`
        or `finite_difference` (contracting forward sensitivities)

    """

    simulated: pd.DataFrame | pd.Series
    wrt: list[str]
    method: _Method
    _backward: _Backward = field(repr=False)
    # Protocol time courses ignore cotangents after the protocol's end
    _end: float | None = field(default=None, repr=False)

    def __call__(
        self, cotangents: pd.DataFrame | pd.Series
    ) -> Result[AdjointSensitivities]:
        """Gradient of `sum(cotangents * simulated)`.

        Parameters
        ----------
        cotangents
            Weight of simulated values (NaN or missing for none): time points
            x columns of `simulated` for time courses, a Series of its index
            for steady states

        """
        return self._backward(self._align(cotangents))

    def _align(self, cotangents: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        sim = self.simulated
        if isinstance(sim, pd.Series):
            if not isinstance(cotangents, pd.Series):
                msg = "Cotangents of a steady state are a Series"
                raise TypeError(msg)
            if unknown := [c for c in cotangents.index if c not in sim.index]:
                msg = f"Unknown cotangent entries {unknown}, choose from {list(sim.index)}"
                raise ValueError(msg)
            return cotangents.reindex(sim.index).astype(float).fillna(0.0)
        if not isinstance(cotangents, pd.DataFrame):
            msg = "Cotangents of a time course are a DataFrame (time points x columns)"
            raise TypeError(msg)
        if unknown := [c for c in cotangents.columns if c not in sim.columns]:
            msg = (
                f"Unknown cotangent columns {unknown}, choose from {list(sim.columns)}"
            )
            raise ValueError(msg)
        times = cotangents.index.to_numpy(dtype=float)
        if self._end is not None:
            cotangents = cotangents.loc[times <= self._end]
            times = times[times <= self._end]
        if missing := sorted(set(times) - set(sim.index.to_numpy(dtype=float))):
            msg = f"Cotangent time points {missing} weren't simulated"
            raise ValueError(msg)
        aligned = cast(pd.DataFrame, cotangents).copy()
        aligned.index = pd.Index(times)
        return aligned.reindex(index=sim.index, columns=sim.columns).fillna(0.0)


def _scaled(cotangents: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Cotangents scaled to max |w| = 1, and the scale.

    λ is linear in the cotangents, but integrated with the absolute tolerance
    of the states, so small cotangents (close to a fit's optimum) would lose
    accuracy.
    """
    scale = float(np.abs(cotangents.to_numpy(dtype=float)).max(initial=0.0))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    return cotangents / scale, scale


def _x0(model: _Model, y0: dict[str, float] | None) -> Array:
    start = model.get_initial_conditions() if y0 is None else y0
    return np.array([start[k] for k in model.get_variable_names()], dtype=float)


def _discrete_pullback(
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    protocol: pd.DataFrame | None,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[Pullback]:
    """Discrete adjoint of `Native`'s solver, over a recorded forward solve."""
    if protocol is not None:
        _check_protocol(model, protocol)
    var_names = model.get_variable_names()
    compiled_wrt, chain, s0 = _chain_factors(
        model, wrt, y0_from_model=y0 is None, displacement=displacement
    )
    segments = None
    if protocol is not None:
        segments = [
            (s.end, s.overrides)
            for s in _segments(model, protocol, float(times[-1]), [], [])
        ]
    res = _discrete.forward(
        model,
        times,
        columns,
        compiled_wrt,
        x0=_x0(model, y0),
        integrator=integrator,
        segments=segments,
    )
    if isinstance(res, Exception):
        return Result(res)
    predictions, tape = res
    simulated = pd.DataFrame(predictions, index=times, columns=columns)

    def backward(cotangents: pd.DataFrame | pd.Series) -> Result[AdjointSensitivities]:
        out = tape.backward(cast(pd.DataFrame, cotangents).to_numpy(dtype=float))
        if isinstance(out, Exception):
            return Result(out)
        theta_bar, lam0 = out
        return Result(
            AdjointSensitivities(
                simulated=simulated,
                gradient=pd.Series(chain.T @ theta_bar + s0.T @ lam0, index=wrt),
                method="discrete_adjoint",
                adjoint=pd.DataFrame(lam0[None, :], index=[0.0], columns=var_names),
            )
        )

    return Result(
        Pullback(
            simulated=simulated, wrt=wrt, method="discrete_adjoint", _backward=backward
        )
    )


def _continuous_pullback(
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    protocol: pd.DataFrame | None,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[Pullback]:
    """Continuous adjoint, in compiled code for `Native` (Radau, LSODA)."""
    if protocol is not None:
        _check_protocol(model, protocol)
    compiled_wrt, chain, s0 = _chain_factors(
        model, wrt, y0_from_model=y0 is None, displacement=displacement
    )
    x0 = _x0(model, y0)
    compiled_integrator = bool(getattr(_unwrap(integrator)[0], "compiles_model", False))
    if compiled_integrator and times[-1] > 0 and _native.supports(integrator):
        try:
            return _compiled_continuous(
                model,
                times,
                columns,
                wrt,
                compiled_wrt=compiled_wrt,
                chain=chain,
                s0=s0,
                x0=x0,
                protocol=protocol,
                integrator=integrator,
            )
        except _native.NativeUnsupported as e:
            _LOGGER.debug("Using the Python adjoint, as %s", e)
    return _python_continuous(
        model,
        times,
        columns,
        wrt,
        compiled_wrt=compiled_wrt,
        chain=chain,
        s0=s0,
        x0=x0,
        protocol=protocol,
        integrator=integrator,
    )


def _compiled_continuous(
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    compiled_wrt: list[str],
    chain: Array,
    s0: Array,
    x0: Array,
    protocol: pd.DataFrame | None,
    integrator: IntegratorType,
) -> Result[Pullback]:
    """Continuous adjoint compiled by `Native`.

    mxn_adjoint_vjp solves forwards and backwards in one call, so the pullback
    solves once for the simulated values and again per call.
    """
    var_names = model.get_variable_names()

    def run(cotangents: pd.DataFrame) -> tuple[Array, Array, Array] | Exception:
        return _native.vjp(
            model,
            cotangents,
            compiled_wrt,
            x0=x0,
            integrator=integrator,
            protocol=protocol,
        )

    first = run(pd.DataFrame(0.0, index=times, columns=columns))
    if isinstance(first, Exception):
        return Result(first)
    simulated = pd.DataFrame(first[0], index=times, columns=columns)

    def backward(cotangents: pd.DataFrame | pd.Series) -> Result[AdjointSensitivities]:
        w, scale = _scaled(cast(pd.DataFrame, cotangents))
        if isinstance(out := run(w), Exception):
            return Result(out)
        _, theta_bar, lam0 = out
        return Result(
            AdjointSensitivities(
                simulated=simulated,
                gradient=pd.Series(
                    (chain.T @ theta_bar + s0.T @ lam0) * scale, index=wrt
                ),
                method="continuous_adjoint",
                adjoint=pd.DataFrame(
                    lam0[None, :] * scale, index=[0.0], columns=var_names
                ),
            )
        )

    return Result(
        Pullback(
            simulated=simulated,
            wrt=wrt,
            method="continuous_adjoint",
            _backward=backward,
        )
    )


def _python_continuous(
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    compiled_wrt: list[str],
    chain: Array,
    s0: Array,
    x0: Array,
    protocol: pd.DataFrame | None,
    integrator: IntegratorType,
) -> Result[Pullback]:
    """The forward trajectory is interpolated once, λ solved per call."""
    var_names = model.get_variable_names()
    outputs = [c for c in columns if c not in var_names]
    vjp_out = compile_vjp(model, wrt=compiled_wrt, outputs=outputs) if outputs else None
    vjp_p = compile_vjp(model, wrt=compiled_wrt)
    rtol, atol = _tolerances(integrator)
    compiled_integrator = bool(getattr(_unwrap(integrator)[0], "compiles_model", False))
    backward_integrator = (
        functools.partial(DefaultIntegrator, rtol=rtol, atol=atol)
        if compiled_integrator
        else integrator
    )
    f, f_constant_names = compile_rhs(model)
    segments = _segments(
        model,
        protocol if times[-1] > 0 else None,
        float(times[-1]),
        vjp_p.constant_names,
        f_constant_names,
    )
    prob = _Problem(
        model=model,
        integrator=integrator,
        backward_integrator=backward_integrator,
        segments=segments,
        vjp_x=compile_vjp(model, wrt=[]),
        vjp_p=vjp_p,
        vjp_out=vjp_out,
        jac=jacobian_fn(model),
        rtol=rtol,
        atol=atol,
        compiled_integrator=compiled_integrator,
        f=f,
    )

    if times[-1] > 0:
        if isinstance(res := _forward(prob, x0, times), Exception):
            return Result(res)
        trajs = res
    else:
        # Only the initial values: nothing to integrate
        trajs = [
            _Hermite(np.array([0.0, 1.0]), np.vstack([x0, x0]), np.zeros((2, len(x0))))
        ]

    # Outputs depend on the parameters of each time point's segment
    blocks = []
    which = np.array([prob.segment_of(float(t)) for t in times])
    for g, segment in enumerate(segments):
        if not (mask := which == g).any():
            continue
        states = pd.DataFrame(
            np.array([trajs[g](float(t)) for t in times[mask]]),
            index=times[mask],
            columns=var_names,
        )
        with _overridden(model, segment.overrides):
            blocks.append(_values(model, states).loc[:, columns])
    simulated = cast(pd.DataFrame, pd.concat(blocks))

    def backward(cotangents: pd.DataFrame | pd.Series) -> Result[AdjointSensitivities]:
        w_frame, scale = _scaled(cast(pd.DataFrame, cotangents))
        w = w_frame.reindex(columns=var_names, fill_value=0.0).to_numpy(dtype=float)
        w_out = w_frame.loc[:, outputs].to_numpy(dtype=float)
        n = len(var_names)
        lam = np.zeros(n)
        theta_bar = np.zeros(len(compiled_wrt))
        pieces: list[pd.DataFrame] = []

        def piece(g: int, t_hi: float, t_lo: float) -> Exception | None:
            """λ from t_hi down to t_lo within segment g, adding its quadrature."""
            nonlocal lam, theta_bar
            if not t_hi > t_lo:
                return None
            prob.seg = segments[g]
            if isinstance(res := _backward(prob, trajs[g], t_lo, t_hi, lam), Exception):
                return res
            traj, quadrature = res
            theta_bar = theta_bar + quadrature
            lam = traj.y[0].copy()
            pieces.append(pd.DataFrame(traj.y, index=traj.t, columns=var_names))
            return None

        # From the last time point down to 0, stopping at every time point (its
        # jump) and segment boundary (λ is continuous there)
        t_hi = float(times[-1])
        g = prob.segment_of(t_hi)
        for i in range(len(times) - 1, -1, -1):
            t_i = float(times[i])
            while g > prob.segment_of(t_i):
                if (err := piece(g, t_hi, segments[g].start)) is not None:
                    return Result(err)
                t_hi = segments[g].start
                g -= 1
            if (err := piece(g, t_hi, t_i)) is not None:
                return Result(err)
            t_hi = t_i
            lam = lam + w[i]
            if vjp_out is not None:
                x_bar, p_bar = vjp_out.fn(
                    t_i,
                    trajs[g](t_i),
                    segments[g].p,
                    segments[g].c,
                    np.zeros(n),
                    w_out[i],
                )
                lam = lam + x_bar
                theta_bar = theta_bar + p_bar
        while g >= 0:
            if (err := piece(g, t_hi, segments[g].start)) is not None:
                return Result(err)
            t_hi = segments[g].start
            g -= 1

        gradient = chain.T @ theta_bar + s0.T @ lam
        # At data times the value after their jump, i.e. the limit from below
        adjoint = (
            pd.concat(pieces[::-1]).groupby(level=0).first()
            if pieces
            else pd.DataFrame(columns=var_names, dtype=float)
        )
        adjoint.loc[0.0] = lam
        adjoint = adjoint.sort_index()
        return Result(
            AdjointSensitivities(
                simulated=simulated,
                gradient=pd.Series(gradient * scale, index=wrt),
                method="continuous_adjoint",
                adjoint=adjoint * scale,
            )
        )

    return Result(
        Pullback(
            simulated=simulated,
            wrt=wrt,
            method="continuous_adjoint",
            _backward=backward,
        )
    )


def _sensitivity_pullback(
    sens: forward_sensitivities.ForwardSensitivities,
    columns: list[str],
    wrt: list[str],
    *,
    steady_state: bool = False,
    end: float | None = None,
) -> Pullback:
    """Contracts forward sensitivities (or finite differences) with the cotangents."""
    simulated = (
        sens.simulated.iloc[-1].loc[columns]
        if steady_state
        else sens.simulated.loc[:, columns]
    )
    return Pullback(
        simulated=simulated,
        wrt=wrt,
        method=sens.method,
        _backward=lambda w: Result(_contract(sens, w, wrt)),
        _end=end,
    )


###############################################################################
# Shared helpers
###############################################################################


def _check_columns(model: _Model, columns: list[str]) -> None:
    variables = set(model.get_variable_names())
    if all(c in variables for c in columns):
        # The output names cost an evaluation of the model
        return
    available = model.get_variable_names() + _output_names(model)
    if unknown := [c for c in columns if c not in available]:
        msg = f"Unknown columns {unknown}, choose from {available}"
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
        simulated = sens.simulated.loc[cotangents.index, columns]
        w = cotangents.to_numpy(dtype=float)
        gradient = [
            float(
                np.sum(
                    sens.derivatives[k]
                    .loc[cotangents.index, columns]
                    .to_numpy(dtype=float)
                    * w
                )
            )
            for k in wrt
        ]
    return AdjointSensitivities(
        simulated=simulated,
        gradient=pd.Series(gradient, index=wrt, dtype=float),
        method=sens.method,
    )


def _check_times(times: Array) -> None:
    if len(times) == 0 or (times < 0).any() or (np.diff(times) <= 0).any():
        msg = "Time points must be ascending, distinct and non-negative"
        raise ValueError(msg)


###############################################################################
# Choosing the method
###############################################################################


def uses_adjoint(
    model: _Model,
    n_wrt: int,
    integrator: IntegratorType | None = None,
    *,
    protocol: bool = False,
) -> bool:
    """Whether `method="auto"` uses adjoints for a time course.

    Always with `Native`, which runs discrete adjoints in compiled code (and
    can't integrate forward sensitivities, so these would fall back to finite
    differences). Otherwise from `ADJOINT_THRESHOLD` variables x parameters /
    initial values on, and for protocols (whose alternative is finite
    differences, 2 simulations per parameter) from
    `PROTOCOL_ADJOINT_THRESHOLD` parameters / initial values on. Never for
    models with events.
    """
    if getattr(model, "_events", None):
        return False
    if integrator is not None and _discrete.supports(integrator):
        return True
    if protocol:
        return n_wrt >= PROTOCOL_ADJOINT_THRESHOLD
    return len(model.get_variable_names()) * n_wrt >= ADJOINT_THRESHOLD


def _candidates(
    method: DerivativeMethod,
    model: _Model,
    n_wrt: int,
    integrator: IntegratorType,
    *,
    protocol: bool,
) -> list[_Method]:
    """Methods to try in order; only the last one's errors propagate."""
    fallback: _Method = "finite_difference" if protocol else "forward"
    events = bool(getattr(model, "_events", None))
    discrete = _discrete.supports(integrator)
    if method in ("adjoint", "discrete_adjoint", "continuous_adjoint") and events:
        msg = "Adjoint sensitivities aren't supported for models with events"
        raise ValueError(msg)
    if method == "discrete_adjoint":
        if not discrete:
            msg = "Discrete adjoints need the Native integrator"
            raise ValueError(msg)
        return ["discrete_adjoint"]
    if method == "continuous_adjoint":
        return ["continuous_adjoint"]
    adjoints: list[_Method] = [
        *(["discrete_adjoint"] if discrete else []),
        "continuous_adjoint",
    ]
    if method == "adjoint":
        return adjoints
    if method == "forward":
        if protocol:
            msg = "Forward sensitivities don't support protocols, use adjoint or finite_difference"
            raise ValueError(msg)
        return ["forward"]
    if method == "finite_difference":
        return ["finite_difference"]
    if uses_adjoint(model, n_wrt, integrator, protocol=protocol):
        return [*adjoints, fallback]
    return [fallback]


###############################################################################
# Public API
###############################################################################


def time_course_pullback(
    model: _Model,
    time_points: ArrayLike,
    wrt: list[str],
    *,
    columns: list[str] | None = None,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
) -> Result[Pullback]:
    """Simulate a time course and return its vector-Jacobian products.

    The returned `Pullback` holds the simulated values; called with
    cotangents (e.g. `d loss / d y(t)`) it returns the gradient of
    `sum(cotangents * y(t))` w.r.t. `wrt`, without simulating again. With
    adjoints that costs one backward pass, however many parameters there are.

    Examples
    --------
        >>> pullback = adjoint_sensitivities.time_course_pullback(model, [1.0, 2.0], ["k1"]).unwrap_or_err()
        >>> pullback(pd.DataFrame({"x": [1.0, -1.0]}, index=[1.0, 2.0])).unwrap_or_err().gradient

    Parameters
    ----------
    model
        Model to simulate
    time_points
        Ascending, distinct, non-negative time points
    wrt
        Parameters and variables (for their initial values) to differentiate w.r.t.
    columns
        Simulated values cotangents may weight (variables, derived quantities
        and reactions), all variables if None
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator. With `Native`, adjoints
        are discrete and run in compiled code
    method
        `discrete_adjoint` differentiates `Native`'s solver steps, exact for the
        computed solution; `continuous_adjoint` solves the adjoint equations
        backwards in time (compiled for `Native` Radau / LSODA, Python
        otherwise); `adjoint` takes the discrete one where possible, else
        (or if it fails, e.g. beyond `Native.max_tape_bytes`) the continuous one;
        `forward` and `finite_difference` contract forward sensitivities (see
        `forward_sensitivities`); `auto` uses adjoints for large problems and
        `Native` (see `uses_adjoint`) and forward sensitivities otherwise, or if
        adjoints aren't possible (models with events)
    displacement
        Relative step of the finite differences of initial assignments

    Returns
    -------
    Result[Pullback]
        The simulated values and their vector-Jacobian product

    """
    return _time_course_pullback(
        model,
        np.asarray(time_points, dtype=float),
        wrt,
        columns=columns,
        protocol=None,
        y0=y0,
        integrator=integrator,
        method=method,
        displacement=displacement,
    )


def protocol_time_course_pullback(
    model: _Model,
    protocol: pd.DataFrame,
    time_points: ArrayLike,
    wrt: list[str],
    *,
    columns: list[str] | None = None,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
) -> Result[Pullback]:
    """Simulate a protocol time course and return its vector-Jacobian products.

    As `time_course_pullback`, for a simulation with
    `Simulator.simulate_protocol_time_course`: every protocol step sets its
    parameters, and a time point on a step's end belongs to that step. Time
    points after the protocol's end are left out, as in the simulation, and
    cotangents there are ignored.

    Examples
    --------
        >>> protocol = make_protocol([(1, {"k_in": 1.0}), (2, {"k_in": 0.0})])
        >>> pullback = adjoint_sensitivities.protocol_time_course_pullback(model, protocol, [0.5, 2.0], ["k1"]).unwrap_or_err()

    Parameters
    ----------
    model
        Model to simulate
    protocol
        Parameter values per step (index: step end times)
    time_points
        Time points, as for `time_course_pullback`
    wrt
        Parameters and variables (for their initial values) to differentiate
        w.r.t., not the parameters the protocol sets
    columns
        As for `time_course_pullback`
    y0
        Initial values, as for `time_course_pullback`
    integrator
        Integrator, as for `time_course_pullback`
    method
        As for `time_course_pullback`, except that forward sensitivities don't
        support protocols: `auto` uses adjoints where they are cheaper (see
        `uses_adjoint`) and finite differences otherwise
    displacement
        Relative step of the finite differences

    Returns
    -------
    Result[Pullback]
        The simulated values and their vector-Jacobian product

    """
    protocol = _normalise_protocol_index(protocol)
    if overridden := sorted(set(wrt) & set(protocol.columns)):
        msg = f"Can't differentiate w.r.t. parameters the protocol sets: {overridden}"
        raise ValueError(msg)
    end = float(cast(pd.TimedeltaIndex, protocol.index).total_seconds()[-1])
    times = np.asarray(time_points, dtype=float)
    times = times[times <= end]
    if len(times) == 0:
        msg = f"No time points within the protocol (ending at {end})"
        raise ValueError(msg)
    res = _time_course_pullback(
        model,
        times,
        wrt,
        columns=columns,
        protocol=protocol,
        y0=y0,
        integrator=integrator,
        method=method,
        displacement=displacement,
    )
    if not isinstance(pullback := res.value, Exception):
        pullback._end = end  # noqa: SLF001
    return res


def _time_course_pullback(
    model: _Model,
    times: Array,
    wrt: list[str],
    *,
    columns: list[str] | None,
    protocol: pd.DataFrame | None,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    method: DerivativeMethod,
    displacement: float,
) -> Result[Pullback]:
    integrator = DefaultIntegrator if integrator is None else integrator
    columns = list(model.get_variable_names()) if columns is None else list(columns)
    _check_names(model, wrt)
    _check_columns(model, columns)
    _check_times(times)

    candidates = _candidates(
        method, model, len(wrt), integrator, protocol=protocol is not None
    )
    for i, candidate in enumerate(candidates):
        last = i == len(candidates) - 1
        try:
            res = _build(
                candidate,
                model,
                times,
                columns,
                wrt,
                protocol=protocol,
                y0=y0,
                integrator=integrator,
                displacement=displacement,
            )
        except Exception as e:
            if last:
                raise
            _LOGGER.debug("Using %s instead of %s: %s", candidates[i + 1], candidate, e)
            continue
        # A failed solve, e.g. of the discrete adjoint's tighter Newton
        # iterations, may still succeed with the next method
        if last or not isinstance(res.value, Exception):
            return res
        _LOGGER.debug(
            "Using %s instead of %s: %s", candidates[i + 1], candidate, res.value
        )
    msg = "unreachable"
    raise AssertionError(msg)


def _build(
    method: _Method,
    model: _Model,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    protocol: pd.DataFrame | None,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[Pullback]:
    if method == "discrete_adjoint":
        return _discrete_pullback(
            model,
            times,
            columns,
            wrt,
            protocol=protocol,
            y0=y0,
            integrator=integrator,
            displacement=displacement,
        )
    if method == "continuous_adjoint":
        return _continuous_pullback(
            model,
            times,
            columns,
            wrt,
            protocol=protocol,
            y0=y0,
            integrator=integrator,
            displacement=displacement,
        )
    if protocol is not None:
        res = _protocol_finite_difference(
            model,
            protocol,
            times,
            columns,
            wrt,
            y0=y0,
            integrator=integrator,
            displacement=displacement,
        )
    else:
        outputs = [c for c in columns if c not in model.get_variable_names()]
        res = forward_sensitivities.time_course(
            model,
            times,
            wrt,
            y0=y0,
            integrator=integrator,
            outputs=outputs,
            method="auto" if method == "forward" else "finite_difference",
            displacement=displacement,
        )
    if isinstance(sens := res.value, Exception):
        return Result(sens)
    return Result(_sensitivity_pullback(sens, columns, wrt))


def _check_protocol(model: _Model, protocol: pd.DataFrame) -> None:
    """Raise if initial assignments depend on parameters the protocol sets.

    Their chain-rule factors would then differ between the steps.
    """
    columns = set(protocol.columns)
    sources = _computed_from(model)
    if bad := sorted(
        k for k in _assigned_parameters(model) if columns & {k, *sources[k]}
    ):
        msg = f"Parameters set by initial assignments depend on the protocol: {bad}"
        raise ValueError(msg)


def _protocol_finite_difference(
    model: _Model,
    protocol: pd.DataFrame,
    times: Array,
    columns: list[str],
    wrt: list[str],
    *,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
) -> Result[forward_sensitivities.ForwardSensitivities]:
    restore = {k: model.get_addressable_values()[k] for k in protocol.columns}
    steps = _segments(model, protocol, float(times[-1]), [], [])

    def simulate(start: dict[str, float]) -> Result[pd.DataFrame]:
        try:
            res = (
                Simulator(model, y0=start, integrator=integrator)
                .simulate_protocol_time_course(protocol, times)
                .get_result()
            )
        finally:
            # The simulation leaves the last step's parameters in the model
            model.update_addressable(restore)
        if isinstance(sim := res.value, Exception):
            return Result(sim)
        variables = _raw_variables(model, sim)
        variables = variables.loc[
            _select_times(variables.index.to_numpy(dtype=float), times)
        ]
        # Outputs with each step's parameters, as in the simulation
        t = variables.index.to_numpy(dtype=float)
        blocks = []
        for segment in steps:
            mask = (t > segment.start) & (t <= segment.end)
            if segment.start == 0:
                mask |= t == 0
            if mask.any():
                with _overridden(model, segment.overrides):
                    blocks.append(_values(model, variables.loc[mask]))
        return Result(pd.concat(blocks).loc[:, columns])

    return _fd(model, wrt, y0, simulate, displacement)


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

    `time_course_pullback` at the cotangents' time points and columns, called once.

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
        Initial values, as for `time_course_pullback`
    integrator
        Integrator, as for `time_course_pullback`
    method
        As for `time_course_pullback`
    displacement
        Relative step of the finite differences of initial assignments

    Returns
    -------
    Result[AdjointSensitivities]
        Gradient and the simulated values at the cotangents' time points

    """
    res = time_course_pullback(
        model,
        cotangents.index.to_numpy(dtype=float),
        wrt,
        columns=list(cotangents.columns),
        y0=y0,
        integrator=integrator,
        method=method,
        displacement=displacement,
    )
    if isinstance(pullback := res.value, Exception):
        return Result(pullback)
    return pullback(cotangents)


def protocol_time_course(
    model: _Model,
    protocol: pd.DataFrame,
    cotangents: pd.DataFrame,
    wrt: list[str],
    *,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
) -> Result[AdjointSensitivities]:
    """Gradient of `sum(cotangents * y(t))` over a protocol.

    `protocol_time_course_pullback` at the cotangents' time points and columns,
    called once. Cotangents after the protocol's end are ignored.

    Examples
    --------
        >>> protocol = make_protocol([(1, {"k_in": 1.0}), (2, {"k_in": 0.0})])
        >>> w = pd.DataFrame({"x": [1.0, -1.0]}, index=[0.5, 2.0])
        >>> adjoint_sensitivities.protocol_time_course(model, protocol, w, ["k1"])

    Parameters
    ----------
    model
        Model to simulate
    protocol
        Parameter values per step (index: step end times)
    cotangents
        Weight of every simulated value, as for `time_course`
    wrt
        Parameters and variables (for their initial values) to differentiate
        w.r.t., not the parameters the protocol sets
    y0
        Initial values, as for `time_course_pullback`
    integrator
        Integrator, as for `time_course_pullback`
    method
        As for `protocol_time_course_pullback`
    displacement
        Relative step of the finite differences

    Returns
    -------
    Result[AdjointSensitivities]
        Gradient and the simulated values at the cotangents' time points

    """
    res = protocol_time_course_pullback(
        model,
        protocol,
        cotangents.index.to_numpy(dtype=float),
        wrt,
        columns=list(cotangents.columns),
        y0=y0,
        integrator=integrator,
        method=method,
        displacement=displacement,
    )
    if isinstance(pullback := res.value, Exception):
        return Result(pullback)
    return pullback(cotangents)


###############################################################################
# Steady states
###############################################################################


def steady_state_pullback(
    model: _Model,
    wrt: list[str],
    *,
    columns: list[str] | None = None,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    method: DerivativeMethod = "auto",
    displacement: float = 1e-3,
    tolerance: float = 1e-6,
    rel_norm: bool = False,
) -> Result[Pullback]:
    """Simulate to the steady state `y*` and return its vector-Jacobian products.

    With adjoints, one linear solve `Jᵀλ = cotangents` per call replaces one
    per parameter (implicit function theorem). Conserved quantities, which make
    `J` singular, are fixed by the initial values instead.

    Examples
    --------
        >>> pullback = adjoint_sensitivities.steady_state_pullback(model, ["k1", "k2"]).unwrap_or_err()
        >>> pullback(pd.Series({"x": 1.0})).unwrap_or_err().gradient

    Parameters
    ----------
    model
        Model to simulate
    wrt
        Parameters and variables (for their initial values) to differentiate
        w.r.t. Initial values only matter for conserved quantities
    columns
        Steady-state values cotangents may weight (variables, derived
        quantities and reactions), all variables if None
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator
    method
        Any adjoint method solves the adjoint system, `forward` and
        `finite_difference` contract forward sensitivities, `auto` uses the
        adjoint unless the model can't be differentiated symbolically
    displacement
        Relative step of the finite differences of initial assignments
    tolerance
        Steady-state tolerance
    rel_norm
        Whether to use the relative norm for the steady-state check

    Returns
    -------
    Result[Pullback]
        The steady-state values and their vector-Jacobian product

    """
    integrator = DefaultIntegrator if integrator is None else integrator
    columns = list(model.get_variable_names()) if columns is None else list(columns)
    _check_names(model, wrt)
    _check_columns(model, columns)
    if method in ("auto", "adjoint", "discrete_adjoint", "continuous_adjoint"):
        try:
            return _adjoint_steady_state(
                model,
                columns,
                wrt,
                y0=y0,
                integrator=integrator,
                displacement=displacement,
                tolerance=tolerance,
                rel_norm=rel_norm,
            )
        except Exception as e:
            if method != "auto":
                raise
            _LOGGER.debug("Using forward sensitivities instead of adjoints: %s", e)

    outputs = [c for c in columns if c not in model.get_variable_names()]
    res = forward_sensitivities.steady_state(
        model,
        wrt,
        y0=y0,
        integrator=integrator,
        outputs=outputs,
        method="finite_difference" if method == "finite_difference" else "auto",
        displacement=displacement,
        tolerance=tolerance,
        rel_norm=rel_norm,
    )
    if isinstance(sens := res.value, Exception):
        return Result(sens)
    return Result(_sensitivity_pullback(sens, columns, wrt, steady_state=True))


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

    `steady_state_pullback` for the cotangents' entries, called once.

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
        Initial values, as for `steady_state_pullback`
    integrator
        Integrator, defaults to the default integrator
    method
        As for `steady_state_pullback`
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
    res = steady_state_pullback(
        model,
        wrt,
        columns=[str(c) for c in cotangents.index],
        y0=y0,
        integrator=integrator,
        method=method,
        displacement=displacement,
        tolerance=tolerance,
        rel_norm=rel_norm,
    )
    if isinstance(pullback := res.value, Exception):
        return Result(pullback)
    return pullback(cotangents)


def _adjoint_steady_state(
    model: _Model,
    columns: list[str],
    wrt: list[str],
    *,
    y0: dict[str, float] | None,
    integrator: IntegratorType,
    displacement: float,
    tolerance: float,
    rel_norm: bool,
) -> Result[Pullback]:
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
    outputs = [c for c in columns if c not in var_names]
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
    p = np.array([values[k] for k in model.get_addressable_names()], dtype=float)
    c = constant_values(model, vjp.constant_names)
    n = len(var_names)
    j = jac(t, x)
    left_null = _left_null_space(j)
    a = np.vstack([j, left_null])
    simulated = _values(model, variables).iloc[-1].loc[columns]

    def backward(cotangents: pd.DataFrame | pd.Series) -> Result[AdjointSensitivities]:
        w_series = cast(pd.Series, cotangents)
        w = w_series.reindex(var_names, fill_value=0.0).to_numpy(dtype=float)
        w_out = w_series.loc[outputs].to_numpy(dtype=float)
        x_out, p_out = vjp.fn(t, x, p, c, np.zeros(n), w_out)
        w = w + x_out
        mu = np.linalg.lstsq(a.T, w, rcond=None)[0]
        mu_j, mu_n = mu[:n], mu[n:]
        _, p_bar = vjp.fn(t, x, p, c, mu_j, np.zeros(len(outputs)))
        theta_bar = p_out - p_bar
        gradient = chain.T @ theta_bar + s0.T @ (left_null.T @ mu_n)
        return Result(
            AdjointSensitivities(
                simulated=simulated,
                gradient=pd.Series(gradient, index=wrt, dtype=float),
                method="adjoint",
                adjoint=pd.DataFrame(mu_j[None, :], index=[t], columns=var_names),
            )
        )

    return Result(
        Pullback(simulated=simulated, wrt=wrt, method="adjoint", _backward=backward)
    )
