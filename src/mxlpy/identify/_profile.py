# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Profile likelihoods, confidence intervals and identifiability."""

from __future__ import annotations

import itertools
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from mxlpy.fit._gradient import _log_names
from mxlpy.identify._problem import (
    BETTER_OPTIMUM,
    _Problem,
    check_threshold,
    likelihood_threshold,
    refit,
    setup_problem,
    start_values,
)
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mxlpy import fit
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.fit.penalties import Penalty
    from mxlpy.identify._problem import _Model
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol

__all__ = [
    "Identifiability",
    "Profile",
    "ProfileLikelihood",
    "SideStatus",
    "profile_likelihood",
]

type Identifiability = Literal[
    "identifiable",
    "practically_non_identifiable",
    "structurally_non_identifiable",
    "inconclusive",
]
"""Identifiability of a parameter from its profile, see `Profile.identifiability`."""

type SideStatus = Literal["threshold", "bound", "max_steps", "failed"]
"""Why a profile stopped on one side: it crossed the threshold, reached the
end of the profiled range, ran out of steps or the fits failed."""

# Profiles whose largest increase stays below this fraction of the threshold
# count as flat
_FLAT = 1e-2
# Default range of a profile without bounds: three orders of magnitude
_RANGE = 1e3
# Step control in the profiled coordinate (log10 units or relative to the value)
_FIRST_STEP = 1e-2
_MAX_STEP = 0.25
_MIN_STEP = _FIRST_STEP / 32
# Each step aims to increase the objective by this fraction of the threshold
# and is retried with half the size if it increases it by more than twice that
_TARGET = 0.1
# Fits locating the threshold crossing, and their tolerance relative to it
_REFINE = 3
_REFINE_TOL = 0.02
# Steps in a row without progress that end a side, see `march`
_STALLS = 3


###############################################################################
# Results
###############################################################################


def crossing(
    delta: pd.Series,
    value: float,
    threshold: float,
    side: Literal[-1, 1],
    *,
    log: bool = False,
) -> float:
    """Where `delta` (indexed by ascending values) crosses `threshold` beside `value`.

    Interpolates linearly (in log10 space with `log`) between the last point
    below and the first at or above the threshold, `side * inf` if none is.
    """
    x = delta.index.to_numpy(dtype=float)
    d = delta.to_numpy(dtype=float)
    order = np.flatnonzero(x >= value if side > 0 else x <= value)
    if side < 0:
        order = order[::-1]
    for inner, outer in itertools.pairwise(order):
        if d[outer] >= threshold:
            u = np.log10(x[[inner, outer]]) if log else x[[inner, outer]]
            frac = (threshold - d[inner]) / (d[outer] - d[inner])
            bound = u[0] + float(np.clip(frac, 0, 1)) * (u[1] - u[0])
            return float(10**bound if log else bound)
    return side * math.inf


@dataclass(kw_only=True)
class Profile:
    """Likelihood profile of one parameter.

    Each point fixes the parameter and refits all others.

    Attributes
    ----------
    name
        The profiled parameter
    value
        Its value at the optimum
    log
        Whether the profile stepped in log10 space (the parameter has a
        `log` transform), where the confidence bounds are interpolated as well
    threshold
        Increase of the objective that bounds the confidence interval
    delta
        Increase of the objective over the optimum, indexed by the profiled
        values (ascending)
    parameters
        Values of all fitted parameters along the profile (same index), the
        paths of the other parameters show which compensate for this one
    lower_status, upper_status
        Why the profile stopped below / above the optimum

    """

    name: str
    value: float
    log: bool
    threshold: float
    delta: pd.Series
    parameters: pd.DataFrame
    lower_status: SideStatus
    upper_status: SideStatus

    def _crossing(self, side: Literal[-1, 1]) -> float:
        status = self.upper_status if side > 0 else self.lower_status
        if status != "threshold":
            return side * math.inf
        return crossing(self.delta, self.value, self.threshold, side, log=self.log)

    @property
    def ci_lower(self) -> float:
        """Lower confidence bound, `-inf` if the threshold isn't crossed."""
        return self._crossing(-1)

    @property
    def ci_upper(self) -> float:
        """Upper confidence bound, `inf` if the threshold isn't crossed."""
        return self._crossing(1)

    @property
    def identifiability(self) -> Identifiability:
        """Identifiability of the parameter from the shape of its profile.

        - `identifiable`: the threshold is crossed on both sides
        - `structurally_non_identifiable`: the profile is flat (below 1% of
          the threshold) over the whole profiled range. Numerically this
          can't be told apart from a parameter the data barely constrains
        - `practically_non_identifiable`: the profile reaches the end of the
          profiled range on at least one side without crossing the threshold
        - `inconclusive`: a side ran out of steps or its fits failed
        """
        sides = (self.lower_status, self.upper_status)
        if sides == ("threshold", "threshold"):
            return "identifiable"
        if any(s in ("max_steps", "failed") for s in sides):
            return "inconclusive"
        if sides == ("bound", "bound") and self.delta.max() < _FLAT * self.threshold:
            return "structurally_non_identifiable"
        return "practically_non_identifiable"


@dataclass(kw_only=True)
class ProfileLikelihood:
    """Profile likelihoods of several parameters, see `profile_likelihood`.

    Attributes
    ----------
    profiles
        Profile per parameter
    optimum
        Optimum the profiles start from (refitted from the given one)
    objective
        Objective (loss + penalties) at the optimum
    threshold
        Increase of the objective that bounds the confidence intervals
    alpha
        Confidence level the threshold was computed from, None if the
        threshold was given
    log_parameters
        Fitted parameters with a `log` transform, whose profiles step in
        log10 space
    better_optimum
        Best parameters found on a profile, if they improve on `optimum`.
        Then the fit the profiles started from was a local optimum: refit
        from these and profile again

    """

    profiles: dict[str, Profile]
    optimum: dict[str, float]
    objective: float
    threshold: float
    alpha: float | None
    log_parameters: list[str]
    better_optimum: dict[str, float] | None = None

    def __getitem__(self, name: str) -> Profile:
        """Profile of a parameter."""
        return self.profiles[name]

    @property
    def ci(self) -> pd.DataFrame:
        """Confidence intervals per parameter.

        Columns value, lower, upper (like `FisherInformation.wald_ci`) and
        identifiability; bounds are infinite where the profile doesn't cross
        the threshold.
        """
        return pd.DataFrame(
            {
                "value": [p.value for p in self.profiles.values()],
                "lower": [p.ci_lower for p in self.profiles.values()],
                "upper": [p.ci_upper for p in self.profiles.values()],
                "identifiability": [p.identifiability for p in self.profiles.values()],
            },
            index=pd.Index(list(self.profiles), name="parameter"),
        )


###############################################################################
# Stepping
###############################################################################


@dataclass(frozen=True)
class _Coordinate:
    """Profiled coordinate: log10 or linear values of one parameter."""

    log: bool
    value: float
    lower: float
    upper: float

    def to_u(self, x: float) -> float:
        return math.log10(x) if self.log else x

    def to_x(self, u: float) -> float:
        return 10**u if self.log else u

    def steps(self, u: float) -> tuple[float, float, float]:
        """First, smallest and largest step at `u`."""
        scale = 1.0 if self.log else (max(abs(u), abs(self.value)) or 1.0)
        return _FIRST_STEP * scale, _MIN_STEP * scale, _MAX_STEP * scale


def _range(
    value: float,
    bounds: tuple[float | None, float | None] | None,
    *,
    log: bool,
) -> tuple[float, float]:
    """Profiled range: the bounds, else three orders of magnitude around `value`.

    In log10 space that's a factor of 1000 either way, in linear space
    `±1000 · max(|value|, 1)`, which includes zero and the other sign.
    """
    if log:
        default = (value / _RANGE, value * _RANGE)
    else:
        width = _RANGE * max(abs(value), 1.0)
        default = (value - width, value + width)
    lower, upper = bounds if bounds is not None else (None, None)
    lower = default[0] if lower is None or (log and lower <= 0) else lower
    upper = default[1] if upper is None else upper
    return float(lower), float(upper)


type _Point = tuple[float, float, dict[str, float]]


@dataclass(frozen=True)
class Step:
    """A point of a profile: coordinate, increase of the objective, parameters.

    `info` holds whatever else the profile records, e.g. the prediction.
    """

    u: float
    delta: float
    pars: dict[str, float]
    info: Any = None


def march(
    at: Callable[[float, dict[str, float]], Step | None],
    *,
    start: Step,
    direction: int,
    limit: float,
    steps: Callable[[float], tuple[float, float, float]],
    threshold: float,
    max_steps: int,
    stalled: Callable[[Step, Step], bool] | None = None,
) -> tuple[list[Step], SideStatus]:
    """Step from `start` in `direction` until the objective increases by `threshold`.

    `at(u, p0)` refits at coordinate `u` from the parameters `p0`, None if
    the fit failed, and `steps(u)` gives the first, smallest and largest step
    at `u`. Steps adapt so each increases the objective by about
    `_TARGET · threshold` and are retried with half the size if they
    increase it by more than twice that. The side ends at the threshold
    (located by a few more fits), at `limit`, after `max_steps` fits, or
    after `_STALLS` steps in a row for which `stalled(previous, new)` holds.
    """
    target = _TARGET * threshold
    points: list[Step] = []
    prev = start
    h, h_min, h_max = steps(prev.u)
    fits = 0
    n_stalled = 0
    while fits < max_steps:
        if (limit - prev.u) * direction <= 0:
            return points, "bound"
        u = prev.u + direction * h
        if (limit - u) * direction < 0:
            u = limit
        fits += 1
        point = at(u, prev.pars)
        if point is None:
            if h > h_min:
                h /= 2
                continue
            return points, "failed"
        if point.delta == math.inf:
            # Edge of a prior's support: approach it, it bounds the profile
            if h > h_min:
                h /= 2
                continue
            return points, "bound"
        increase = point.delta - prev.delta
        if increase > 2 * target and h > h_min:
            h /= 2
            continue
        points.append(point)
        if point.delta >= threshold:
            points.extend(_refine(at, prev, point, threshold))
            return points, "threshold"
        n_stalled = n_stalled + 1 if stalled is not None and stalled(prev, point) else 0
        if n_stalled >= _STALLS:
            return points, "bound"
        prev = point
        _, h_min, h_max = steps(prev.u)
        factor = math.sqrt(target / increase) if increase > 0 else 2.0
        h = min(max(h * min(max(factor, 0.5), 2.0), h_min), h_max)
    return points, "max_steps"


def _refine(
    at: Callable[[float, dict[str, float]], Step | None],
    inside: Step,
    outside: Step,
    threshold: float,
) -> list[Step]:
    """Points locating the threshold crossing by regula falsi."""
    points = []
    for _ in range(_REFINE):
        if abs(outside.delta - threshold) < _REFINE_TOL * threshold:
            break
        frac = (threshold - inside.delta) / (outside.delta - inside.delta)
        u = inside.u + frac * (outside.u - inside.u)
        if (point := at(u, inside.pars)) is None or point.delta == math.inf:
            break
        points.append(point)
        if abs(point.delta - threshold) < _REFINE_TOL * threshold:
            break
        if point.delta < threshold:
            inside = point
        else:
            outside = point
    return points


def walk(
    task: tuple[str, int],
    *,
    problem: _Problem,
    optimum: dict[str, float],
    objective: float,
    threshold: float,
    coordinates: dict[str, _Coordinate],
    max_steps: int,
    fixed: dict[str, float] | None = None,
    start_delta: float = 0.0,
) -> tuple[list[_Point], SideStatus]:
    """Profile one parameter in one direction from `optimum`.

    `fixed` values stay fixed in all refits, then `optimum` is the best fit
    with them, whose objective is `start_delta` above `objective`.
    """
    name, direction = task
    coord = coordinates[name]
    model = deepcopy(problem.model)
    fixed = {} if fixed is None else fixed

    def at(u: float, start: dict[str, float]) -> Step | None:
        x = coord.to_x(u)
        p0 = {k: v for k, v in start.items() if k != name and k not in fixed}
        if (res := problem.fit(model, {**fixed, name: x}, p0)) is None:
            return None
        return Step(u, res[0] - objective, res[1])

    points, status = march(
        at,
        start=Step(coord.to_u(optimum[name]), start_delta, optimum),
        direction=direction,
        limit=coord.to_u(coord.upper if direction > 0 else coord.lower),
        steps=coord.steps,
        threshold=threshold,
        max_steps=max_steps,
    )
    return [(coord.to_x(p.u), p.delta, p.pars) for p in points], status


def coordinates_of(
    names: Sequence[str],
    start: dict[str, float],
    problem: _Problem,
    log: set[str],
) -> dict[str, _Coordinate]:
    """Profiled coordinate per parameter, see `_range`."""
    out = {}
    for k in names:
        lower, upper = _range(start[k], problem.bounds.get(k), log=k in log)
        out[k] = _Coordinate(log=k in log, value=start[k], lower=lower, upper=upper)
    return out


###############################################################################
# Entry point
###############################################################################


def profile_likelihood(
    model: _Model,
    data: pd.Series | pd.DataFrame,
    parameters: fit.Fit | Mapping[str, float],
    *,
    loss_fn: LossFn,
    names: Sequence[str] | None = None,
    alpha: float = 0.95,
    threshold: float | None = None,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
    runner: Any | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    transform: Transform | None = None,
    postprocess_fn: PostprocessFn | None = None,
    standard_scale: bool = True,
    max_steps: int = 100,
    parallel: bool = True,
    max_workers: int | None = None,
) -> ProfileLikelihood:
    """Profile likelihoods, confidence intervals and identifiability.

    For each profiled parameter, steps away from the optimum in both
    directions, fixing the parameter and refitting all other fitted
    parameters (warm-started from the previous point). Steps adapt so each
    increases the objective by about a tenth of the threshold, dense near
    the optimum and sparse where the profile is flat. A side stops when the
    objective has increased by `threshold` (the confidence bound, located by
    a few extra fits), at the end of the profiled range or after
    `max_steps` fits. Both sides of all parameters run in parallel.

    With a likelihood loss (`fit.GaussianNll`), the threshold is the
    likelihood-ratio bound `chi2.ppf(alpha, 1) / 2` of pointwise confidence
    intervals. Pass `threshold=chi2.ppf(alpha, n) / 2` for intervals that
    hold simultaneously for `n` parameters, or a threshold of your choice
    for other losses.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        A Series fits a steady state, a DataFrame (indexed by time) a time
        course, or a protocol time course if `protocol` is given
    parameters
        Fitted parameters (and initial values) the profiles start from, e.g.
        a fit result, as for `fisher_information`. They are refitted first,
        so they only need to be close to the optimum
    loss_fn
        Loss of the fits, the same as for the optimum
    names
        Parameters to profile, all of `parameters` by default
    alpha
        Confidence level of the intervals, for likelihood losses
    threshold
        Increase of the objective bounding the confidence intervals,
        required for losses that aren't likelihoods
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits. Variables in `parameters` are fitted
        initial values, as in the fit functions
    protocol
        Protocol of a protocol time course
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter, which also limit the profiles.
        Without bounds, profiles span a factor of 1000 around the optimum in
        both directions (log10 steps), or `±1000 · max(|value|, 1)`
    prior
        Priors of the parameters, as in the fit functions. The profiled
        parameter's prior stays in the objective at its fixed values
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit these parameters in log10 space and
        profile them in log10 steps, which suits values spanning orders of
        magnitude
    postprocess_fn
        Applied to the simulation before the loss, as in the fit functions
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    max_steps
        Maximum number of fits per side of a profile
    parallel
        Whether to profile in parallel
    max_workers
        Maximum number of worker processes

    Examples
    --------
        >>> res = fit.time_course(model, p0=p0, data=data, loss_fn=nll,
        ...                       minimizer=fit.LeastSquaresMinimizer()).unwrap_or_err()
        >>> profiles = identify.profile_likelihood(model, data, res, loss_fn=nll,
        ...                                        transform="log")
        >>> profiles.ci
        >>> plot.profile_likelihood(profiles)

    """
    start = start_values(parameters)
    names = list(start) if names is None else list(names)
    if unknown := sorted(set(names) - set(start)):
        msg = f"Profiled parameters have to be fitted ones, not {unknown}"
        raise ValueError(msg)
    if max_steps < 1:
        msg = f"max_steps has to be positive, got {max_steps}"
        raise ValueError(msg)
    if threshold is None:
        threshold = likelihood_threshold(loss_fn, alpha, "Confidence thresholds")
        used_alpha: float | None = alpha
    else:
        threshold = check_threshold(threshold)
        used_alpha = None
    problem = setup_problem(
        model,
        data,
        start,
        loss_fn=loss_fn,
        minimizer=minimizer,
        y0=y0,
        protocol=protocol,
        runner=runner,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        transform=transform,
        postprocess_fn=postprocess_fn,
        standard_scale=standard_scale,
    )
    log = _log_names(start, transform)
    objective, start = refit(problem, start)

    coordinates = coordinates_of(names, start, problem, log)
    tasks = [((k, d), (k, d)) for k in names for d in (-1, 1)]
    walks = dict(
        parallelise(
            partial(
                walk,
                problem=problem,
                optimum=start,
                objective=objective,
                threshold=threshold,
                coordinates=coordinates,
                max_steps=max_steps,
            ),
            inputs=tasks,
            parallel=parallel,
            max_workers=max_workers,
            tqdm_desc="profiles",
        )
    )

    profiles = {}
    best_delta, better = -BETTER_OPTIMUM * threshold, None
    for k in names:
        lower, lower_status = walks[(k, -1)]
        upper, upper_status = walks[(k, 1)]
        points = sorted([*lower, (start[k], 0.0, start), *upper], key=lambda p: p[0])
        index = pd.Index([p[0] for p in points], name=k)
        profiles[k] = Profile(
            name=k,
            value=start[k],
            log=k in log,
            threshold=threshold,
            delta=pd.Series([p[1] for p in points], index=index, name="delta"),
            parameters=pd.DataFrame(
                [p[2] for p in points], index=index, columns=list(start)
            ),
            lower_status=lower_status,
            upper_status=upper_status,
        )
        for _, delta, pars in points:
            if delta < best_delta:
                best_delta, better = delta, pars
    if better is not None:
        warnings.warn(
            "A profile found a better fit than the optimum, which is a local "
            "optimum. Refit from ProfileLikelihood.better_optimum and profile again",
            stacklevel=2,
        )
    return ProfileLikelihood(
        profiles=profiles,
        optimum=start,
        objective=objective,
        threshold=threshold,
        alpha=used_alpha,
        log_parameters=sorted(log),
        better_optimum=better,
    )
