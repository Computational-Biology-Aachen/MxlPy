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
from scipy import stats

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit._gradient import _log_names
from mxlpy.fit.abstract import _Settings, _split_p0
from mxlpy.fit.losses import _is_likelihood
from mxlpy.fit.penalties import collect_penalties, fix_penalties
from mxlpy.identify._fisher import residual_fn
from mxlpy.parallel import parallelise
from mxlpy.runners import default_runner, precompile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.fit.penalties import Penalty
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

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

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


###############################################################################
# Results
###############################################################################


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
        x = self.delta.index.to_numpy(dtype=float)
        d = self.delta.to_numpy(dtype=float)
        order = np.flatnonzero(x >= self.value if side > 0 else x <= self.value)
        if side < 0:
            order = order[::-1]
        for inner, outer in itertools.pairwise(order):
            if d[outer] >= self.threshold:
                u = np.log10(x[[inner, outer]]) if self.log else x[[inner, outer]]
                frac = (self.threshold - d[inner]) / (d[outer] - d[inner])
                bound = u[0] + float(np.clip(frac, 0, 1)) * (u[1] - u[0])
                return float(10**bound if self.log else bound)
        return side * math.inf

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
# Constrained fits
###############################################################################


@dataclass(kw_only=True)
class _Problem:
    """A fit with some of its parameters held fixed."""

    model: _Model
    data: pd.Series | pd.DataFrame
    loss_fn: LossFn
    minimizer: MinimizerProtocol
    y0: dict[str, float] | None
    protocol: pd.DataFrame | None
    runner: Any
    bounds: Bounds
    penalties: list[Penalty]
    transform: Transform | None
    postprocess_fn: PostprocessFn | None
    standard_scale: bool

    def _subset(self, names: Sequence[str]) -> tuple[Bounds, Transform | None]:
        bounds = {k: v for k, v in self.bounds.items() if k in names}
        if isinstance(self.transform, dict):
            return bounds, {k: v for k, v in self.transform.items() if k in names}
        return bounds, self.transform

    def _evaluate(self, model: _Model, y0: dict[str, float] | None) -> float:
        """Loss without free parameters, there is nothing to fit."""
        settings = _Settings(
            model=model,
            data=self.data,
            y0=y0,
            runner=self.runner,
            loss_fn=self.loss_fn,
            postprocess_fn=self.postprocess_fn,
            p_names=[],
            v_names=[],
            standard_scale=self.standard_scale,
            protocol=self.protocol,
        )
        return float(residual_fn(self.data, self.protocol)({}, settings=settings))

    def fit(
        self,
        model: _Model,
        fixed: dict[str, float],
        p0: dict[str, float],
    ) -> tuple[float, dict[str, float]] | None:
        """Objective and parameters of the best fit with `fixed` values.

        `model` is modified, pass a copy owned by the caller.
        """
        fixed_p, fixed_v = _split_p0(model, fixed)
        for k in fixed_p:
            model.update_parameter(k, fixed[k])
        y0 = self.y0
        if fixed_v:
            y0 = {**(self.y0 or {}), **{k: fixed[k] for k in fixed_v}}
        penalties, constant = fix_penalties(self.penalties, fixed)
        if constant == math.inf:
            # Outside the support of a prior of a fixed parameter
            return math.inf, {**p0, **fixed}
        if not p0:
            loss = self._evaluate(model, y0)
            return (loss + constant, dict(fixed)) if math.isfinite(loss) else None

        bounds, transform = self._subset(list(p0))
        kwargs: dict[str, Any] = {
            "p0": p0,
            "data": self.data,
            "minimizer": self.minimizer,
            "y0": y0,
            "runner": self.runner,
            "loss_fn": self.loss_fn,
            "postprocess_fn": self.postprocess_fn,
            "standard_scale": self.standard_scale,
            "bounds": bounds,
            "penalty": penalties,
            "transform": transform,
            "as_deepcopy": False,
        }
        if self.protocol is not None:
            res = fit.protocol_time_course(
                model,  # type: ignore[arg-type]
                protocol=self.protocol,
                **kwargs,
            )
        elif isinstance(self.data, pd.Series):
            res = fit.steady_state(model, **kwargs)
        else:
            res = fit.time_course(model, **kwargs)  # type: ignore[arg-type]
        match res.value:
            case fit.Fit() as best if math.isfinite(best.objective):
                return best.objective + constant, {**best.best_pars, **fixed}
            case _:
                return None


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


def _walk(
    task: tuple[str, int],
    *,
    problem: _Problem,
    optimum: dict[str, float],
    objective: float,
    threshold: float,
    coordinates: dict[str, _Coordinate],
    max_steps: int,
) -> tuple[list[_Point], SideStatus]:
    """Profile one parameter in one direction from the optimum."""
    name, direction = task
    coord = coordinates[name]
    model = deepcopy(problem.model)
    target = _TARGET * threshold
    limit = coord.to_u(coord.upper if direction > 0 else coord.lower)

    def at(u: float, start: dict[str, float]) -> _Point | None:
        x = coord.to_x(u)
        p0 = {k: v for k, v in start.items() if k != name}
        if (res := problem.fit(model, {name: x}, p0)) is None:
            return None
        return x, res[0] - objective, res[1]

    points: list[_Point] = []
    prev_u, prev_delta, prev_pars = coord.to_u(coord.value), 0.0, optimum
    h, h_min, h_max = coord.steps(prev_u)
    fits = 0
    while fits < max_steps:
        if (limit - prev_u) * direction <= 0:
            return points, "bound"
        u = prev_u + direction * h
        if (limit - u) * direction < 0:
            u = limit
        fits += 1
        point = at(u, prev_pars)
        if point is None:
            if h > h_min:
                h /= 2
                continue
            return points, "failed"
        if point[1] == math.inf:
            # Edge of a prior's support: approach it, it bounds the profile
            if h > h_min:
                h /= 2
                continue
            return points, "bound"
        increase = point[1] - prev_delta
        if increase > 2 * target and h > h_min:
            h /= 2
            continue
        points.append(point)
        if point[1] >= threshold:
            points.extend(
                _refine(at, coord, (prev_u, prev_delta, prev_pars), point, threshold)
            )
            return points, "threshold"
        prev_u, prev_delta, prev_pars = coord.to_u(point[0]), point[1], point[2]
        _, h_min, h_max = coord.steps(prev_u)
        factor = math.sqrt(target / increase) if increase > 0 else 2.0
        h = min(max(h * min(max(factor, 0.5), 2.0), h_min), h_max)
    return points, "max_steps"


def _refine(
    at: Any,
    coord: _Coordinate,
    inside: tuple[float, float, dict[str, float]],
    outside: _Point,
    threshold: float,
) -> list[_Point]:
    """Points locating the threshold crossing by regula falsi."""
    (u_in, d_in, pars_in), (u_out, d_out) = inside, (coord.to_u(outside[0]), outside[1])
    points = []
    for _ in range(_REFINE):
        if abs(d_out - threshold) < _REFINE_TOL * threshold:
            break
        u = u_in + (threshold - d_in) / (d_out - d_in) * (u_out - u_in)
        if (point := at(u, pars_in)) is None or point[1] == math.inf:
            break
        points.append(point)
        if abs(point[1] - threshold) < _REFINE_TOL * threshold:
            break
        if point[1] < threshold:
            u_in, d_in, pars_in = u, point[1], point[2]
        else:
            u_out, d_out = u, point[1]
    return points


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
    if isinstance(parameters, fit.Fit):
        parameters = parameters.best_pars
    start = {k: float(v) for k, v in parameters.items()}
    names = list(start) if names is None else list(names)
    if unknown := sorted(set(names) - set(start)):
        msg = f"Profiled parameters have to be fitted ones, not {unknown}"
        raise ValueError(msg)
    p_names, v_names = _split_p0(model, start)
    if unknown := sorted(set(start) - set(p_names) - set(v_names)):
        msg = f"Unknown parameters or variables: {unknown}"
        raise ValueError(msg)
    if max_steps < 1:
        msg = f"max_steps has to be positive, got {max_steps}"
        raise ValueError(msg)
    if threshold is None:
        if not _is_likelihood(loss_fn):
            msg = (
                "Confidence thresholds need a likelihood loss like fit.GaussianNll, "
                "pass threshold= for other losses"
            )
            raise ValueError(msg)
        if not 0 < alpha < 1:
            msg = f"alpha has to be between 0 and 1, got {alpha}"
            raise ValueError(msg)
        threshold = float(stats.chi2.ppf(alpha, df=1) / 2)
        used_alpha: float | None = alpha
    else:
        if not (math.isfinite(threshold) and threshold > 0):
            msg = f"threshold has to be positive, got {threshold}"
            raise ValueError(msg)
        used_alpha = None
    if isinstance(model, SteadyStateModelBuilder) and y0 is not None:
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)

    model = deepcopy(model)
    runner = default_runner(model) if runner is None else runner
    precompile(runner, model)
    log = _log_names(start, transform)
    problem = _Problem(
        model=model,
        data=data,
        loss_fn=loss_fn,
        minimizer=fit.LeastSquaresMinimizer() if minimizer is None else minimizer,
        y0=y0,
        protocol=protocol,
        runner=runner,
        bounds={} if bounds is None else dict(bounds),
        penalties=collect_penalties(prior, penalty, list(start)),
        transform=transform,
        postprocess_fn=postprocess_fn,
        standard_scale=standard_scale,
    )

    if (best := problem.fit(deepcopy(model), {}, start)) is None:
        msg = "Refitting the optimum failed"
        raise ValueError(msg)
    objective, start = best

    coordinates = {}
    for k in names:
        lower, upper = _range(start[k], problem.bounds.get(k), log=k in log)
        coordinates[k] = _Coordinate(
            log=k in log, value=start[k], lower=lower, upper=upper
        )
    tasks = [((k, d), (k, d)) for k in names for d in (-1, 1)]
    walks = dict(
        parallelise(
            partial(
                _walk,
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
    best_delta, better = -_FLAT * threshold, None
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
        better_optimum=better,
    )
