# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numerical identifiability test by pulling the parameters away from the optimum."""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mxlpy.fit._gradient import _log_names
from mxlpy.fit._problem import (
    BETTER_OPTIMUM,
    _Problem,
    check_threshold,
    condition_losses,
    is_likelihood,
    likelihood_threshold,
    refit,
    setup_problem,
    start_values,
)
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy import fit
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit._problem import _Model
    from mxlpy.fit.abstract import Data, PostprocessFn
    from mxlpy.fit.callbacks import Callback
    from mxlpy.fit.penalties import Penalty
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
    from mxlpy.types import Array

__all__ = [
    "IdentifiabilityTest",
    "PullingRound",
    "identifiability_test",
]

# A pulled fit reached the radius if its distance is at least this fraction of it
_REACHED = 0.9
# Weight of the radius penalty: missing the radius by 10% costs the threshold
_WEIGHT = 100.0
# Smaller relative thresholds mean a numerically exact fit, where the radius
# penalty (weighted by the threshold) would vanish
_MIN_THRESHOLD = 1e-10


###############################################################################
# Results
###############################################################################


@dataclass(kw_only=True)
class PullingRound:
    """Pulled fits of one round of `identifiability_test`.

    Attributes
    ----------
    fixed
        Parameters held at their optimal values in this round
    fits
        One row per pulled fit: `delta`, the increase of the objective over
        the optimum, `distance` from the optimum, whether it `reached` the
        radius (at least 90% of it) and whether it is `flat` (reached it
        without increasing the objective by more than the threshold).
        Failed fits have NaN `delta` and `distance`, and are neither
    displacement
        Signed change of each free parameter per fit, in the test's
        coordinates (log10 or relative, see `identifiability_test`)
    parameters
        Fitted values of all parameters per fit
    non_identifiable
        Free parameters that moved by at least `min_displacement · radius`
        in a flat fit, by decreasing displacement
    threshold
        Increase of the objective up to which a fit counts as flat

    """

    fixed: list[str]
    fits: pd.DataFrame
    displacement: pd.DataFrame
    parameters: pd.DataFrame
    non_identifiable: list[str]
    threshold: float

    @property
    def is_identifiable(self) -> bool:
        """Whether no pulled fit found a flat direction."""
        return not bool(self.fits["flat"].to_numpy(dtype=bool).any())

    @property
    def n_blocked(self) -> int:
        """Number of fits that stayed within the threshold but short of the radius.

        Something other than the data stopped them, usually bounds, so they
        don't show whether the direction they moved in is flat. If a round
        without flat fits has blocked ones, its result is inconclusive.
        """
        fits = self.fits
        blocked = ~fits["reached"].to_numpy(dtype=bool) & (
            fits["delta"].to_numpy(dtype=float) < self.threshold
        )
        return int(blocked.sum())

    @property
    def n_failed(self) -> int:
        """Number of pulled fits that failed."""
        return int(np.isnan(self.fits["delta"].to_numpy(dtype=float)).sum())

    @property
    def max_displacement(self) -> pd.Series:
        """Largest absolute displacement per free parameter, descending.

        Over the flat fits if there are any, else over all successful fits.
        """
        disp = self.displacement.abs()
        if not self.is_identifiable:
            disp = disp.loc[self.fits["flat"].to_numpy()]
        out = disp.max(axis=0, skipna=True).fillna(0.0)
        return out.sort_values(ascending=False, kind="stable").rename("displacement")


@dataclass(kw_only=True)
class IdentifiabilityTest:
    """Result of `identifiability_test`.

    Attributes
    ----------
    rounds
        Pulled fits per round. Only one without `recursive`, else each round
        fixes the parameter that moved most in the round before, until a
        round finds no flat direction
    optimum
        Optimum the fits are pulled away from (refitted from the given one)
    objective
        Objective (loss + penalties) at the optimum
    threshold
        Increase of the objective up to which a pulled fit counts as flat
    radius
        Distance the parameters are pulled to
    min_displacement
        Fraction of the radius a parameter has to move in a flat fit to count
        as non-identifiable
    log_parameters
        Parameters whose distances are measured in log10 units, the others
        relative to their optimal value
    fixed
        Parameters fixed by the recursion, in order. Fixing them at their
        optimal values leaves no flat direction the test finds, so they are
        candidates to fix. Empty without `recursive` or if the model passes
    better_optimum
        Best parameters of a pulled fit, if they improve on `optimum`. Then
        the optimum was a local one: refit from these and test again

    """

    rounds: list[PullingRound]
    optimum: dict[str, float]
    objective: float
    threshold: float
    radius: float
    min_displacement: float
    log_parameters: list[str]
    fixed: list[str]
    better_optimum: dict[str, float] | None = None

    @property
    def is_identifiable(self) -> bool:
        """Whether the first round found no flat direction."""
        return self.rounds[0].is_identifiable

    @property
    def non_identifiable(self) -> list[str]:
        """Parameters that moved along a flat direction in any round.

        In order of their first appearance, then by decreasing displacement.
        """
        out: list[str] = []
        for r in self.rounds:
            out.extend(k for k in r.non_identifiable if k not in out)
        return out

    @property
    def displacement(self) -> pd.Series:
        """Largest absolute displacement per parameter in the first round, descending."""
        return self.rounds[0].max_displacement


###############################################################################
# Radius penalty
###############################################################################


@dataclass(frozen=True)
class _RadiusPenalty:
    """`weight · (‖u - u*‖ - radius)²` in the test's coordinates u.

    u is log10 θ for log parameters, else θ / |θ*| (θ for θ* = 0).
    """

    names: list[str]
    center: Array
    log: Array
    scale: Array
    radius: float
    weight: float

    def coordinates(self, values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log10(np.where(self.log, values, 1.0))
        return np.where(self.log, logged, values / self.scale)

    def displacement(self, values: Array) -> Array:
        return self.coordinates(values) - self.center

    def distance(self, values: Array) -> float:
        return float(np.linalg.norm(self.displacement(values)))

    def __call__(self, values: Array) -> float:
        return self.weight * (self.distance(values) - self.radius) ** 2

    def residuals(self, values: Array) -> Array:
        return np.array(
            [math.sqrt(2 * self.weight) * (self.distance(values) - self.radius)]
        )

    def jacobian(self, values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        disp = self.displacement(values)
        dist = float(np.linalg.norm(disp))
        if dist == 0:
            return np.zeros((1, len(values)))
        with np.errstate(divide="ignore", invalid="ignore"):
            du = np.where(self.log, 1 / (values * math.log(10)), 1 / self.scale)
        return (math.sqrt(2 * self.weight) * disp / dist * du)[None, :]

    def to_values(self, u: Array) -> Array:
        return np.where(self.log, 10.0**u, u * self.scale)


def _radius_penalty(
    optimum: dict[str, float],
    names: list[str],
    log: set[str],
    radius: float,
    weight: float,
) -> _RadiusPenalty:
    is_log = np.array([k in log for k in names])
    values = np.array([optimum[k] for k in names], dtype=float)
    scale = np.where(is_log | (values == 0), 1.0, np.abs(values))
    with np.errstate(divide="ignore", invalid="ignore"):
        center = np.where(is_log, np.log10(np.abs(values)), values / scale)
    return _RadiusPenalty(
        names=names,
        center=center,
        log=is_log,
        scale=scale,
        radius=radius,
        weight=weight,
    )


def _starts(
    penalty: _RadiusPenalty,
    bounds: Bounds,
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Points drawn uniformly in the ball around the optimum, within the bounds."""
    dim = len(penalty.names)
    lower = np.array([(bounds.get(k) or (None, None))[0] for k in penalty.names])
    upper = np.array([(bounds.get(k) or (None, None))[1] for k in penalty.names])
    lower = np.where(lower == None, -np.inf, lower).astype(float)  # noqa: E711
    upper = np.where(upper == None, np.inf, upper).astype(float)  # noqa: E711
    out = []
    for _ in range(n):
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        u = penalty.center + penalty.radius * rng.uniform() ** (1 / dim) * direction
        values = np.clip(penalty.to_values(u), lower, upper)
        out.append(dict(zip(penalty.names, values.tolist(), strict=True)))
    return out


###############################################################################
# Rounds
###############################################################################


def _pull(
    p0: dict[str, float],
    *,
    problem: _Problem,
    fixed: dict[str, float],
    penalty: _RadiusPenalty,
) -> tuple[float, dict[str, float]] | None:
    return problem.fit(deepcopy(problem.model), fixed, p0, extra=[penalty])


def _round(
    problem: _Problem,
    *,
    optimum: dict[str, float],
    objective: float,
    fixed: list[str],
    log: set[str],
    radius: float,
    threshold: float,
    min_displacement: float,
    n_fits: int,
    rng: np.random.Generator,
    parallel: bool,
    max_workers: int | None,
) -> PullingRound:
    free = [k for k in optimum if k not in fixed]
    penalty = _radius_penalty(
        optimum, free, log, radius, _WEIGHT * threshold / radius**2
    )
    starts = _starts(penalty, problem.bounds, n_fits, rng)
    results = dict(
        parallelise(
            partial(
                _pull,
                problem=problem,
                fixed={k: optimum[k] for k in fixed},
                penalty=penalty,
            ),
            inputs=list(enumerate(starts)),
            parallel=parallel,
            max_workers=max_workers,
            tqdm_desc="identifiability test",
        )
    )
    index = pd.RangeIndex(n_fits, name="fit")
    rows, disp, pars = [], [], []
    for i in index:
        res = results.get(i)
        if res is None or not math.isfinite(res[0]):
            rows.append((math.nan, math.nan, False, False))
            disp.append([math.nan] * len(free))
            pars.append(dict.fromkeys(optimum, math.nan))
            continue
        values = np.array([res[1][k] for k in free], dtype=float)
        delta, distance = res[0] - objective, penalty.distance(values)
        reached = distance >= _REACHED * radius
        rows.append((delta, distance, reached, reached and delta < threshold))
        disp.append(penalty.displacement(values).tolist())
        pars.append({k: res[1][k] for k in optimum})
    fits = pd.DataFrame(
        rows, index=index, columns=["delta", "distance", "reached", "flat"]
    )
    rnd = PullingRound(
        fixed=list(fixed),
        fits=fits,
        displacement=pd.DataFrame(disp, index=index, columns=free),
        parameters=pd.DataFrame(pars, index=index, columns=list(optimum)),
        non_identifiable=[],
        threshold=threshold,
    )
    if not rnd.is_identifiable:
        moved = rnd.max_displacement
        rnd.non_identifiable = [
            str(k) for k, v in moved.items() if float(v) >= min_displacement * radius
        ]
    return rnd


###############################################################################
# Entry point
###############################################################################


def identifiability_test(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    loss_fn: LossFn | None = None,
    radius: float = 1.0,
    n_fits: int = 10,
    recursive: bool = False,
    alpha: float = 0.95,
    threshold: float | None = None,
    threshold_rtol: float = 0.05,
    min_displacement: float = 0.1,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
    runner: Any | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    transform: Transform | None = "log",
    postprocess_fn: PostprocessFn | None = None,
    standard_scale: bool = True,
    seed: int = 0,
    parallel: bool = True,
    max_workers: int | None = None,
    callback: Callback | None = None,
) -> IdentifiabilityTest:
    """Fast numerical identifiability test around the optimum.

    Refits the parameters `n_fits` times from random points in a ball around
    the optimum, with a penalty `w · (‖u - u*‖ - radius)²` that pulls them
    to distance `radius` from it. If a fit reaches the radius (at least 90%
    of it) without increasing the objective by more than `threshold`, the
    objective is flat in that direction and the parameters that moved along
    it (by at least `min_displacement · radius`) are non-identifiable. The
    weight `w` is set so that missing the radius by 10% costs the threshold.

    This costs `n_fits` fits instead of a profile per parameter, and is a
    check, not a proof: flat directions the random starts miss, and ones
    blocked by bounds, go unnoticed. Follow up with `profile_likelihood` on
    the flagged parameters.

    Distances are measured in log10 units for parameters with a `log`
    transform (`radius=1` pulls by a decade), and relative to the optimal
    value for the others (`θ / |θ*|`).

    With `recursive`, the parameter that moved most is fixed at its optimal
    value and the test repeated, until a round finds no flat direction.
    `IdentifiabilityTest.fixed` is then a set of parameters whose fixing
    makes the model pass, `non_identifiable` all that moved in any round.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        A Series fits a steady state, a DataFrame (indexed by time) a time
        course, or a protocol time course if `protocol` is given. Several
        data sets are fitted jointly: a mapping of names to data or
        `fit.Condition`s (with their own initial values, protocol,
        parameters and noise)
    parameters
        Fitted parameters (and initial values), e.g. a fit result. They are
        refitted first, so they only need to be close to the optimum
    loss_fn
        Loss of the fits, the same as for the optimum
    radius
        Distance to pull the parameters to
    n_fits
        Number of pulled fits per round
    recursive
        Whether to fix the most displaced parameter and repeat until the
        model passes
    alpha
        Confidence level of the threshold `chi2.ppf(alpha, 1) / 2`, for
        likelihood losses (`fit.GaussianNll`)
    threshold
        Increase of the objective up to which a pulled fit counts as flat.
        By default the likelihood-ratio bound for likelihood losses, else
        `threshold_rtol` times the objective at the optimum (which raises if that's
        below 1e-10, e.g. for noise-free data)
    threshold_rtol
        Relative threshold of losses that aren't likelihoods
    min_displacement
        Fraction of the radius a parameter has to move in a flat fit to count
        as non-identifiable
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default. It
        needs a sum-of-squares loss (`fit.GaussianNll`, `mean_squared`), use
        `fit.LocalScipyMinimizer()` for others
    y0
        Initial conditions of the fits. Variables in `parameters` are fitted
        initial values, as in the fit functions
    protocol
        Protocol of a protocol time course
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter
    prior
        Priors of the parameters, as in the fit functions
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit these parameters and measure their
        distances in log10 space; parameters that can be non-positive need
        `"linear"`
    postprocess_fn
        Applied to the simulation before the loss, as in the fit functions
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    seed
        Seed of the random starts
    parallel
        Whether to run the fits of a round in parallel
    max_workers
        Maximum number of worker processes
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every refit, see `mxlpy.fit.callbacks`. Has to be picklable with
        `parallel=True`

    Examples
    --------
        >>> res = fit.time_course(model, p0=p0, data=data, loss_fn=nll,
        ...                       minimizer=fit.LeastSquaresMinimizer()).unwrap_or_err()
        >>> test = identify.identifiability_test(model, data, res, loss_fn=nll)
        >>> test.is_identifiable, test.non_identifiable
        >>> test.displacement

    """
    if not (math.isfinite(radius) and radius > 0):
        msg = f"radius has to be positive, got {radius}"
        raise ValueError(msg)
    if n_fits < 1:
        msg = f"n_fits has to be positive, got {n_fits}"
        raise ValueError(msg)
    if not 0 < min_displacement <= 1:
        msg = f"min_displacement has to be in (0, 1], got {min_displacement}"
        raise ValueError(msg)
    if threshold is not None:
        threshold = check_threshold(threshold)
    elif is_likelihood(condition_losses(data, loss_fn)):
        threshold = likelihood_threshold(
            condition_losses(data, loss_fn), alpha, "Thresholds from alpha"
        )
    elif not (math.isfinite(threshold_rtol) and threshold_rtol > 0):
        msg = f"threshold_rtol has to be positive, got {threshold_rtol}"
        raise ValueError(msg)
    start = start_values(parameters)
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
        callback=callback,
    )
    log = _log_names(start, transform)
    objective, start = refit(problem, start)
    if threshold is None:
        threshold = threshold_rtol * abs(objective)
        if not threshold > _MIN_THRESHOLD:
            msg = (
                f"The objective at the optimum ({objective:.3g}) is too small for a "
                "relative threshold (noise-free data?), pass threshold="
            )
            raise ValueError(msg)

    rng = np.random.default_rng(seed)
    rounds: list[PullingRound] = []
    fixed: list[str] = []
    while len(fixed) < len(start):
        rnd = _round(
            problem,
            optimum=start,
            objective=objective,
            fixed=fixed,
            log=log,
            radius=radius,
            threshold=threshold,
            min_displacement=min_displacement,
            n_fits=n_fits,
            rng=rng,
            parallel=parallel,
            max_workers=max_workers,
        )
        if rnd.n_failed == n_fits:
            if not rounds:
                msg = "All pulled fits failed"
                raise ValueError(msg)
            warnings.warn(
                f"All pulled fits failed with {fixed} fixed, stopping the recursion",
                stacklevel=2,
            )
            break
        rounds.append(rnd)
        if rnd.is_identifiable or not recursive:
            break
        fixed = [*fixed, str(rnd.max_displacement.index[0])]

    if (last := rounds[-1]).is_identifiable and last.n_blocked:
        warnings.warn(
            f"{last.n_blocked} pulled fits stayed short of the radius without "
            "exceeding the threshold, usually because of bounds, so the test is "
            "inconclusive for the directions they moved in",
            stacklevel=2,
        )
    better, best_delta = None, -BETTER_OPTIMUM * threshold
    for rnd in rounds:
        deltas = rnd.fits["delta"].to_numpy(dtype=float)
        for i, delta in zip(rnd.fits.index, deltas, strict=True):
            if delta < best_delta:
                best_delta = float(delta)
                better = {k: float(v) for k, v in rnd.parameters.loc[i].items()}
    if better is not None:
        warnings.warn(
            "A pulled fit found a better fit than the optimum, which is a local "
            "optimum. Refit from IdentifiabilityTest.better_optimum and test again",
            stacklevel=2,
        )
    return IdentifiabilityTest(
        rounds=rounds,
        optimum=start,
        objective=objective,
        threshold=threshold,
        radius=radius,
        min_displacement=min_displacement,
        log_parameters=sorted(log),
        fixed=fixed if recursive else [],
        better_optimum=better,
    )
