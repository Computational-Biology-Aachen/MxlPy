# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Parameterised time-dependent input functions.

Shapes of external signals (a ligand, light, an un-modelled upstream
species) whose parameters can be fitted together with the kinetic ones
("input estimation"). Use them as derived values of the model time:

    >>> from mxlpy import fns_input as inp
    >>> model.add_derived(
    ...     "drug", inp.smooth_step, args=["time", "d_low", "t_on", "d_high", "steepness"]
    ... )
    >>> model.add_spline_input("light", knots=[0, 10, 20, 40], initial=100.0)

Unlike protocols, which are piecewise constant and change parameters between
integrations, these are model functions: they can be fitted, differentiated
for sensitivities, analysed symbolically and exported by all code generators.

Functions
---------
- `smooth_step`, `step`: one switch between two levels
- `periodic_step`: smooth periodic switching, e.g. light / dark cycles
- `bolus`: Gaussian pulse with a given area

Factories (fixed times, fitted values)
--------------------------------------
- `cubic_spline`: cubic spline through values at fixed knot times, with
  positive, monotone and initial-slope variants
- `multi_step`: several (smooth) switches at fixed times

Smooth shapes are preferred over hard steps, which make the right-hand side
discontinuous and force integrators into tiny steps. Hard steps whose levels
aren't fitted are better expressed as a protocol.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sympy

from mxlpy.meta.sympy_tools import compile_fn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "bolus",
    "cubic_spline",
    "multi_step",
    "periodic_step",
    "smooth_step",
    "spline_parameter_names",
    "step",
]


###############################################################################
# Plain functions
###############################################################################


def smooth_step(
    t: float,
    level1: float,
    switch_time: float,
    level2: float,
    steepness: float,
) -> float:
    """Smooth (logistic) switch from `level1` to `level2` around `switch_time`.

    `level1 + (level2 - level1) / (1 + exp(-steepness (t - switch_time)))`,
    written with `tanh` so it can't overflow. The switch takes about
    `4 / steepness` time units from 12% to 88% of the way.

    Examples
    --------
    >>> smooth_step(10.0, 0.0, 10.0, 2.0, 1.0)
    1.0

    """
    return level1 + (level2 - level1) * 0.5 * (
        1.0 + math.tanh(0.5 * steepness * (t - switch_time))
    )


def step(t: float, level1: float, switch_time: float, level2: float) -> float:
    """Hard switch from `level1` to `level2` at `switch_time`.

    Discontinuous: integrators have to resolve the jump with tiny steps and
    the switch time can't be fitted with gradients. Prefer `smooth_step`.

    Examples
    --------
    >>> step(9.0, 0.0, 10.0, 2.0), step(10.0, 0.0, 10.0, 2.0)
    (0.0, 2.0)

    """
    return level1 if t < switch_time else level2


def periodic_step(  # noqa: PLR0917, model functions take positional args
    t: float,
    low: float,
    high: float,
    period: float,
    on_fraction: float,
    sharpness: float,
) -> float:
    """Smooth periodic switching between `low` and `high`.

    Each period starts with the `high` phase, which lasts `on_fraction` of
    the period, e.g. 16 h light in a 24 h day: `period=24, on_fraction=2/3`.
    `sharpness` (dimensionless) sets how square the wave is: around 1 it is
    close to a sine, at 50 the switches take about 1% of a period.
    The switches are centred on the phase boundaries, so at `t = 0` the
    value is halfway between `low` and `high`.

    Examples
    --------
    >>> round(periodic_step(4.0, 0.0, 1.0, 24.0, 0.5, 50.0), 6)
    1.0
    >>> round(periodic_step(16.0, 0.0, 1.0, 24.0, 0.5, 50.0), 6)
    0.0

    """
    phase = math.cos(2.0 * math.pi * (t / period - 0.5 * on_fraction))
    return low + (high - low) * 0.5 * (
        1.0 + math.tanh(sharpness * (phase - math.cos(math.pi * on_fraction)))
    )


def bolus(t: float, area: float, peak_time: float, width: float) -> float:
    """Gaussian pulse of total `area` centred at `peak_time` with standard deviation `width`.

    E.g. an injected dose: as a reaction rate it adds `area` in total.

    Examples
    --------
    >>> round(bolus(5.0, 1.0, 5.0, 1.0), 6)
    0.398942

    """
    return (
        area
        / (width * math.sqrt(2.0 * math.pi))
        * math.exp(-0.5 * ((t - peak_time) / width) ** 2)
    )


###############################################################################
# Factories
###############################################################################

# Relative size below which spline weights are dropped
_DROP = 1e-15


def _check_times(times: Sequence[float], what: str, minimum: int) -> np.ndarray:
    x = np.asarray(times, dtype=float)
    if x.ndim != 1 or len(x) < minimum:
        msg = f"{what} need at least {minimum} times, got {list(times)}"
        raise ValueError(msg)
    if not np.isfinite(x).all() or (np.diff(x) <= 0).any():
        msg = f"{what} have to be finite and strictly increasing, got {list(times)}"
        raise ValueError(msg)
    return x


def _second_derivatives(
    x: np.ndarray,
    initial_slope: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Second derivatives at the knots as `K @ y + c`.

    Natural spline (zero curvature) at the end, and at the start too unless
    the first derivative there is clamped to `initial_slope`.
    """
    n = len(x)
    h = np.diff(x)
    a = np.zeros((n, n))
    b = np.zeros((n, n))
    c = np.zeros(n)
    if initial_slope is None:
        a[0, 0] = 1.0
    else:
        # S'(x0) = (y1 - y0)/h0 - h0 (2 M0 + M1) / 6
        a[0, 0], a[0, 1] = h[0] / 3, h[0] / 6
        b[0, 0], b[0, 1] = -1 / h[0], 1 / h[0]
        c[0] = -initial_slope
    for i in range(1, n - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        b[i, i - 1] = 6 / h[i - 1]
        b[i, i] = -6 / h[i - 1] - 6 / h[i]
        b[i, i + 1] = 6 / h[i]
    a[-1, -1] = 1.0
    inv = np.linalg.inv(a)
    return inv @ b, inv @ c


def _linear_form(
    weights: np.ndarray,
    constant: float,
    y: list[sympy.Expr],
) -> sympy.Expr:
    """`weights @ y + constant`, dropping weights below 1e-15 of the largest.

    Built without evaluation, as simplifying sums of many terms is slow.
    """
    tol = _DROP * max(float(np.abs(weights).max()), abs(constant))
    terms = [
        sympy.Float(w) * yj for w, yj in zip(weights, y, strict=True) if abs(w) > tol
    ]
    if abs(constant) > tol:
        terms.append(sympy.Float(constant))
    if not terms:
        return sympy.Float(0)
    return terms[0] if len(terms) == 1 else sympy.Add(*terms, evaluate=False)


def _natural_pieces(
    x: np.ndarray,
    y: list[sympy.Expr],
    t: sympy.Symbol,
    initial_slope: float | None,
) -> list[sympy.Expr]:
    """Cubic per interval in Horner form, coefficients linear in `y`.

    The second derivatives `M = K y + c` depend on all values, but the
    weights decay geometrically away from each knot, so weights below
    `1e-15` of the largest of each coefficient are dropped and the
    expression stays small.
    """
    n = len(x)
    k, c = _second_derivatives(x, initial_slope)
    eye = np.eye(n)
    pieces = []
    for i in range(n - 1):
        h = float(x[i + 1] - x[i])
        b = (eye[i + 1] - eye[i]) / h - h * (2 * k[i] + k[i + 1]) / 6
        b0 = -h * (2 * c[i] + c[i + 1]) / 6
        c2, c20 = k[i] / 2, c[i] / 2
        c3, c30 = (k[i + 1] - k[i]) / (6 * h), (c[i + 1] - c[i]) / (6 * h)
        s = t - sympy.Float(x[i])
        pieces.append(
            y[i]
            + s
            * (
                _linear_form(b, b0, y)
                + s * (_linear_form(c2, c20, y) + s * _linear_form(c3, c30, y))
            )
        )
    return pieces


def _monotone_pieces(
    x: np.ndarray,
    y: list[sympy.Expr],
    t: sympy.Symbol,
    initial_slope: float | None,
) -> list[sympy.Expr]:
    """Monotone piecewise cubic Hermite interpolation (Fritsch-Butland slopes).

    Interior slopes are the weighted harmonic mean of the neighbouring
    secants, or 0 at local extrema, the end slopes the secants of the end
    intervals. That keeps the spline monotone wherever the values are,
    unless a given `initial_slope` has the wrong sign or exceeds three times
    the first secant.
    """
    n = len(x)
    h = [sympy.Float(v) for v in np.diff(x)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    slopes: list[sympy.Expr] = [
        d[0] if initial_slope is None else sympy.Float(initial_slope)
    ]
    for i in range(1, n - 1):
        w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
        extremum = d[i - 1] * d[i] <= 0
        # Weighted harmonic mean (w1 + w2) / (w1 / d0 + w2 / d1). The guarded
        # denominator keeps backends that evaluate both branches (jnp.select)
        # from dividing by zero, which would make their gradients NaN
        # Adding 1 where it is zero, rather than a Piecewise denominator,
        # which sympy would fold into the outer one or invert branch-wise
        denominator = w1 * d[i] + w2 * d[i - 1]
        denominator += sympy.Piecewise(
            (sympy.Float(1), sympy.Eq(denominator, 0)), (sympy.Float(0), True)
        )
        slopes.append(
            sympy.Piecewise(
                (sympy.Float(0), extremum),
                ((w1 + w2) * d[i - 1] * d[i] / denominator, True),
            )
        )
    slopes.append(d[-1])
    pieces = []
    for i in range(n - 1):
        s = (t - sympy.Float(x[i])) / h[i]
        pieces.append(
            (2 * s**3 - 3 * s**2 + 1) * y[i]
            + (s**3 - 2 * s**2 + s) * h[i] * slopes[i]
            + (-2 * s**3 + 3 * s**2) * y[i + 1]
            + (s**3 - s**2) * h[i] * slopes[i + 1]
        )
    return pieces


def spline_parameter_names(name: str, n_knots: int) -> list[str]:
    """Names of the knot values `add_spline_input` creates: `{name}_0`, `{name}_1`, ..."""
    return [f"{name}_{i}" for i in range(n_knots)]


def cubic_spline(
    knots: Sequence[float],
    *,
    positive: bool = False,
    monotone: bool = False,
    initial_slope: float | None = None,
) -> Callable[..., float]:
    """Cubic spline through values at fixed knot times.

    Returns a function `f(t, y0, ..., y{n-1})` interpolating the values
    `y_i` at the times `knots[i]`, constant beyond the first and last knot.
    The values are its arguments, so they can be fitted as parameters. The
    function carries its piecewise-polynomial sympy expression, so it works
    with sensitivities, symbolic analysis, serialisation and all code
    generators.

    Parameters
    ----------
    knots
        Knot times, at least two, strictly increasing
    positive
        Interpolate the logarithms of the values and exponentiate, so the
        input stays positive. The values have to be positive
    monotone
        Monotone piecewise cubic (Fritsch-Butland slopes, once
        differentiable) instead of a natural spline (twice
        differentiable), which can overshoot between knots
    initial_slope
        Fix the first derivative at the first knot, e.g. 0 for an input
        that starts at steady state. With `positive`, it is the slope of the
        logarithm. Default: natural (zero curvature), or the secant for
        `monotone`

    Examples
    --------
    >>> from mxlpy import fns_input as inp
    >>> knots = [0, 5, 10, 30, 60]
    >>> model.add_parameters({f"u{i}": 1.0 for i in range(len(knots))})
    >>> model.add_derived(
    ...     "EGF",
    ...     inp.cubic_spline(knots, positive=True),
    ...     args=["time", *(f"u{i}" for i in range(len(knots)))],
    ... )

    See `add_spline_input` of the model builders for a shortcut.

    """
    x = _check_times(knots, "Spline knots", 2)
    if initial_slope is not None and not math.isfinite(initial_slope):
        msg = f"initial_slope has to be finite, got {initial_slope}"
        raise ValueError(msg)
    t = sympy.Symbol("t")
    args = [f"y{i}" for i in range(len(x))]
    values = [sympy.Symbol(a) for a in args]
    y: list[sympy.Expr] = [sympy.log(v) for v in values] if positive else values  # type: ignore[misc]
    pieces = (_monotone_pieces if monotone else _natural_pieces)(x, y, t, initial_slope)
    branches = [(y[0], t < sympy.Float(x[0]))]
    branches.extend(
        (piece, t < sympy.Float(x[i + 1])) for i, piece in enumerate(pieces)
    )
    branches.append((y[-1], True))
    expr: sympy.Expr = sympy.Piecewise(*branches)  # type: ignore[assignment]
    if positive:
        expr = sympy.exp(expr)  # type: ignore[assignment]
    kind = "monotone_spline" if monotone else "cubic_spline"
    return compile_fn(expr, ["t", *args], name=f"{kind}_{len(x)}")


def multi_step(
    switch_times: Sequence[float],
    *,
    steepness: float | None,
) -> Callable[..., float]:
    """Switches between levels at fixed times.

    Returns a function `f(t, level0, ..., level{n})` that is `level0`
    before `switch_times[0]`, `level{i}` between the i-th and (i+1)-th switch
    and `level{n}` after the last one; the levels can be fitted as
    parameters. With a `steepness`, each switch is a `smooth_step`; with
    `steepness=None` the switches are hard (discontinuous, see `step`).
    There is no default, as a smooth switch needs a time scale.

    Examples
    --------
    >>> from mxlpy import fns_input as inp
    >>> model.add_derived(
    ...     "light",
    ...     inp.multi_step([10.0, 20.0], steepness=5.0),
    ...     args=["time", "dark", "light_high", "light_low"],
    ... )

    """
    x = _check_times(switch_times, "Switch times", 1)
    t = sympy.Symbol("t")
    args = [f"level{i}" for i in range(len(x) + 1)]
    levels = [sympy.Symbol(a) for a in args]
    if steepness is None:
        branches = [(levels[i], t < sympy.Float(ti)) for i, ti in enumerate(x)]
        branches.append((levels[-1], True))
        expr: sympy.Expr = sympy.Piecewise(*branches)  # type: ignore[assignment]
    else:
        if not (math.isfinite(steepness) and steepness > 0):
            msg = f"steepness has to be positive, got {steepness}"
            raise ValueError(msg)
        k = sympy.Float(0.5 * steepness)
        expr = levels[0] + sum(
            (
                (levels[i + 1] - levels[i])
                * sympy.Float(0.5)
                * (1 + sympy.tanh(k * (t - sympy.Float(ti))))
                for i, ti in enumerate(x)
            ),
            sympy.Float(0),
        )
    return compile_fn(expr, ["t", *args], name=f"multi_step_{len(x)}")


def _add_spline_input[M](
    model: M,
    name: str,
    knots: Sequence[float],
    *,
    initial: float | Sequence[float],
    positive: bool,
    monotone: bool,
    initial_slope: float | None,
) -> M:
    """Implementation of the builders' `add_spline_input`."""
    names = spline_parameter_names(name, len(knots))
    values = (
        [float(cast("float", initial))] * len(knots)
        if np.ndim(initial) == 0
        else [float(v) for v in initial]  # type: ignore[union-attr]
    )
    if len(values) != len(knots):
        msg = f"initial needs one value per knot ({len(knots)}), got {len(values)}"
        raise ValueError(msg)
    if positive and any(v <= 0 for v in values):
        msg = f"A positive spline needs positive initial values, got {values}"
        raise ValueError(msg)
    fn = cubic_spline(
        knots, positive=positive, monotone=monotone, initial_slope=initial_slope
    )
    model.add_parameters(dict(zip(names, values, strict=True)))  # type: ignore[attr-defined]
    model.add_derived(name, fn, args=["time", *names])  # type: ignore[attr-defined]
    return model
