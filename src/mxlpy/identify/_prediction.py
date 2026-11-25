# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Prediction and validation profiles and prediction bands.

The profile of a prediction `g(θ)` is `min χ²(θ) s.t. g(θ) = z`. It is
computed from refits with a pseudo data point `z ± sigma` of the prediction
(Kreutz et al. 2012): the best fit `θ*` with it is also the best fit among
all parameters predicting `g(θ*)`, so `(g(θ*), objective(θ*))` is an exact
point of the prediction profile for any `sigma`. Stepping `z` traces the
profile, `sigma` only sets how closely the prediction follows `z`. With the
measurement error of a new data point as `sigma`, the objective including the
pseudo data point is its validation profile.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from mxlpy.fit._gradient import _log_names
from mxlpy.fit._problem import (
    Constraint,
    Prediction,
    _Problem,
    check_threshold,
    condition_losses,
    likelihood_threshold,
    refit,
    setup_problem,
    start_values,
)
from mxlpy.identify._profile import SideStatus, Step, crossing, march
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
    "PredictionBand",
    "PredictionProfile",
    "ValidationProfile",
    "prediction_band",
    "prediction_profile",
    "validation_profile",
]

# sigma of the pseudo data point of prediction profiles relative to the
# linearised standard deviation s of the prediction, so the prediction moves
# about 1 / (1 + 0.5²) = 80% of the way to z
_SIGMA = 0.5
# Steps in z relative to s (or sqrt(s² + sigma²) for validation profiles)
_FIRST_STEP = 0.5
_MIN_STEP = _FIRST_STEP / 64
_MAX_STEP = 1e3
# A step moving the prediction by less than this fraction of the step in z
# made no progress: the prediction is bounded (e.g. by parameter bounds)
_PROGRESS = 1e-2
# Predictor-corrector fits of the integrated band per time, and their
# tolerance relative to the threshold
_CORRECT = 4
_CORRECT_TOL = 0.05


###############################################################################
# Results
###############################################################################


def _side(
    delta: pd.Series,
    value: float,
    threshold: float,
    status: SideStatus,
    side: Literal[-1, 1],
) -> float:
    """Confidence bound of a prediction on one side.

    At the end of the reachable predictions (`bound`) that's the most
    extreme prediction found, as none beyond it is reachable within the
    bounds of the fit.
    """
    if status == "threshold":
        return crossing(delta, value, threshold, side)
    if status == "bound":
        x = delta.index.to_numpy(dtype=float)
        return float(x.max() if side > 0 else x.min())
    return side * math.inf


@dataclass(kw_only=True)
class PredictionProfile:
    """Likelihood profile of a prediction, see `prediction_profile`.

    Attributes
    ----------
    prediction
        The profiled prediction
    value
        Its value at the optimum
    threshold
        Increase of the objective that bounds the confidence interval
    alpha
        Confidence level the threshold was computed from, None if the
        threshold was given
    sigma
        sigma of the pseudo data point the profile was traced with
    delta
        Increase of the objective over the optimum, indexed by the predicted
        values (ascending)
    z
        Value of the pseudo data point per profile point (same index)
    parameters
        Values of all fitted parameters along the profile (same index)
    lower_status, upper_status
        Why the profile stopped below / above the optimum. `bound` means
        the prediction can't move further within the bounds of the fit

    """

    prediction: Prediction
    value: float
    threshold: float
    alpha: float | None
    sigma: float
    delta: pd.Series
    z: pd.Series
    parameters: pd.DataFrame
    lower_status: SideStatus
    upper_status: SideStatus

    @property
    def ci_lower(self) -> float:
        """Lower confidence bound, `-inf` if it wasn't found."""
        return _side(self.delta, self.value, self.threshold, self.lower_status, -1)

    @property
    def ci_upper(self) -> float:
        """Upper confidence bound, `inf` if it wasn't found."""
        return _side(self.delta, self.value, self.threshold, self.upper_status, 1)

    @property
    def ci(self) -> tuple[float, float]:
        """Confidence interval `(lower, upper)` of the prediction."""
        return self.ci_lower, self.ci_upper


@dataclass(kw_only=True)
class ValidationProfile:
    """Validation profile of a new data point, see `validation_profile`.

    Attributes
    ----------
    prediction
        The predicted quantity the data point measures
    value
        Its prediction at the optimum
    sigma
        Measurement error of the new data point
    threshold
        Increase of the objective (including the data point) that bounds the
        values consistent with the model
    alpha
        Confidence level the threshold was computed from, None if the
        threshold was given
    delta
        Increase of the objective including the data point over the
        optimum, indexed by the value of the data point (ascending)
    predicted
        Prediction of the best fit including the data point (same index)
    data_delta
        Increase of the objective without the data point (same index); at
        the `predicted` values these are exact points of the prediction
        profile
    parameters
        Values of all fitted parameters along the profile (same index)
    lower_status, upper_status
        Why the profile stopped below / above the optimum

    """

    prediction: Prediction
    value: float
    sigma: float
    threshold: float
    alpha: float | None
    delta: pd.Series
    predicted: pd.Series
    data_delta: pd.Series
    parameters: pd.DataFrame
    lower_status: SideStatus
    upper_status: SideStatus

    @property
    def validation_ci(self) -> tuple[float, float]:
        """Values of a new data point consistent with the model, `(lower, upper)`.

        Infinite where the profile stopped before crossing the threshold.
        """
        pairs: tuple[tuple[Literal[-1, 1], SideStatus], ...] = (
            (-1, self.lower_status),
            (1, self.upper_status),
        )
        lower, upper = (
            crossing(self.delta, self.value, self.threshold, side)
            if status == "threshold"
            else side * math.inf
            for side, status in pairs
        )
        return lower, upper

    def is_consistent(self, z: float) -> bool:
        """Whether a measured value `z` is consistent with the calibrated model.

        The likelihood-ratio test of the data point: it is, if adding it
        increases the objective by less than the threshold, i.e. if `z` is
        within `validation_ci`. Raises a `ValueError` for values beyond a
        side whose profile ran out of steps or failed, as it can't tell.
        """
        lower, upper = self.validation_ci
        walked = self.delta.index.to_numpy(dtype=float)
        if (z < walked.min() and self.lower_status in ("max_steps", "failed")) or (
            z > walked.max() and self.upper_status in ("max_steps", "failed")
        ):
            msg = (
                f"The validation profile stopped ({self.lower_status} / "
                f"{self.upper_status}) before reaching z = {z}, increase max_steps"
            )
            raise ValueError(msg)
        return lower <= z <= upper


@dataclass(kw_only=True)
class PredictionBand:
    """Pointwise confidence band of a prediction over time, see `prediction_band`.

    Attributes
    ----------
    prediction
        The predicted quantity and condition (its `time` is None)
    threshold
        Increase of the objective that bounds the band at each time
    alpha
        Confidence level the threshold was computed from, None if the
        threshold was given
    method
        `integrated` or `pointwise`
    value
        Prediction of the optimum, indexed by time
    lower, upper
        Bounds of the band (same index), infinite where they weren't found
    lower_status, upper_status
        Why the profile stopped below / above the optimum at each time; the
        integrated method only profiles the times where tracking the
        bounds failed, the others count as `threshold`
    profiles
        Prediction profile per time, for the pointwise method

    """

    prediction: Prediction
    threshold: float
    alpha: float | None
    method: Literal["integrated", "pointwise"]
    value: pd.Series
    lower: pd.Series
    upper: pd.Series
    lower_status: pd.Series
    upper_status: pd.Series
    profiles: dict[float, PredictionProfile] | None = None


###############################################################################
# Stepping
###############################################################################


def prediction_scale(
    problem: _Problem,
    prediction: Prediction,
    *,
    model: _Model,
    optimum: dict[str, float],
    values: Array,
    log: set[str],
    times: Sequence[float] | None = None,
) -> Array:
    """Linearised standard deviation of the prediction (at `times`).

    `sqrt(∇gᵀ H⁺ ∇g)` with the Gauss-Newton Hessian `H` of the loss, in
    log10 space for log-transformed parameters. Only scales the steps, so a
    tenth of the prediction is used if the derivatives aren't available.
    """
    fallback = 0.1 * np.maximum(np.abs(values), 1e-6)
    try:
        jac, factor = problem.loss_jacobian(optimum)
        grad = problem.prediction_jacobian(prediction, model, optimum, times)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return fallback
    if factor is None or not (np.isfinite(jac).all() and np.isfinite(grad).all()):
        return fallback
    scale = np.array(
        [v * math.log(10) if k in log else 1.0 for k, v in optimum.items()]
    )
    jac, grad = jac * scale, grad * scale
    hessian = factor * jac.T @ jac
    s2 = np.einsum("ti,ij,tj->t", grad, np.linalg.pinv(hessian, hermitian=True), grad)
    s = np.sqrt(np.maximum(s2, 0.0))
    # A prediction independent of the parameters doesn't move, the steps
    # still need a scale to find that out
    return np.where(np.isfinite(s), np.maximum(s, 1e-9 * fallback), fallback)


@dataclass(frozen=True, kw_only=True)
class _Walker:
    """Refits with a pseudo data point of one prediction."""

    problem: _Problem
    prediction: Prediction
    objective: float
    sigma: float
    validation: bool
    fixed: dict[str, float]

    def at_fn(self) -> Any:
        """`at(z, p0)` of `march`, with models owned by the returned function."""
        model = deepcopy(self.problem.model)
        condition = self.problem.condition(self.prediction)

        def at(z: float, start: dict[str, float]) -> Step | None:
            p0 = {k: v for k, v in start.items() if k not in self.fixed}
            constraint = Constraint(
                prediction=self.prediction, model=condition, z=z, sigma=self.sigma
            )
            res = self.problem.fit_constrained(model, dict(self.fixed), p0, constraint)
            if res is None:
                return None
            objective, pars, g = res
            delta = objective - self.objective
            if self.validation:
                delta += ((g - z) / self.sigma) ** 2 / 2
            return Step(z, delta, pars, info=g)

        return at

    def walk(
        self,
        direction: int,
        *,
        start: Step,
        scale: float,
        threshold: float,
        max_steps: int,
    ) -> tuple[list[Step], SideStatus]:
        """Step z away from `start` (whose `info` is its prediction)."""

        def stalled(prev: Step, new: Step) -> bool:
            return (new.info - prev.info) * direction < _PROGRESS * abs(new.u - prev.u)

        return march(
            self.at_fn(),
            start=start,
            direction=direction,
            limit=direction * math.inf,
            steps=lambda _: (_FIRST_STEP * scale, _MIN_STEP * scale, _MAX_STEP * scale),
            threshold=threshold,
            max_steps=max_steps,
            # The pseudo data point bounds validation profiles
            stalled=None if self.validation else stalled,
        )


def _walk_task(
    task: tuple[Any, int],
    *,
    walkers: Mapping[Any, _Walker],
    starts: Mapping[Any, Step],
    scales: Mapping[Any, float],
    threshold: float,
    max_steps: int,
) -> tuple[list[Step], SideStatus]:
    key, direction = task
    return walkers[key].walk(
        direction,
        start=starts[key],
        scale=scales[key],
        threshold=threshold,
        max_steps=max_steps,
    )


def prediction_points(
    start: Step,
    lower: list[Step],
    upper: list[Step],
    columns: list[str],
    *,
    by: Literal["prediction", "z"],
) -> tuple[pd.Index, list[Step], pd.DataFrame]:
    """Points of both sides sorted by the prediction or z, with their parameters."""
    points = sorted(
        [*lower, start, *upper], key=lambda p: p.info if by == "prediction" else p.u
    )
    index = pd.Index(
        [p.info if by == "prediction" else p.u for p in points],
        name="prediction" if by == "prediction" else "z",
    )
    return (
        index,
        points,
        pd.DataFrame([p.pars for p in points], index=index, columns=columns),
    )


def _prediction_profile(
    walker: _Walker,
    *,
    start: Step,
    sides: tuple[tuple[list[Step], SideStatus], tuple[list[Step], SideStatus]],
    threshold: float,
    alpha: float | None,
    columns: list[str],
) -> PredictionProfile:
    (lower, lower_status), (upper, upper_status) = sides
    index, points, parameters = prediction_points(
        start, lower, upper, columns, by="prediction"
    )
    return PredictionProfile(
        prediction=walker.prediction,
        value=float(start.info),
        threshold=threshold,
        alpha=alpha,
        sigma=walker.sigma,
        delta=pd.Series([p.delta for p in points], index=index, name="delta"),
        z=pd.Series([p.u for p in points], index=index, name="z"),
        parameters=parameters,
        lower_status=lower_status,
        upper_status=upper_status,
    )


###############################################################################
# Setup shared by the entry points
###############################################################################


@dataclass(kw_only=True)
class _Setup:
    problem: _Problem
    optimum: dict[str, float]
    objective: float
    threshold: float
    alpha: float | None
    log: set[str]


def _setup(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    loss_fn: LossFn | None = None,
    alpha: float,
    threshold: float | None,
    max_steps: int,
    what: str,
    df: int = 1,
    **kwargs: Any,
) -> _Setup:
    start = start_values(parameters)
    if max_steps < 1:
        msg = f"max_steps has to be positive, got {max_steps}"
        raise ValueError(msg)
    if threshold is None:
        threshold = likelihood_threshold(
            condition_losses(data, loss_fn), alpha, what, df=df
        )
        used_alpha: float | None = alpha
    else:
        threshold = check_threshold(threshold)
        used_alpha = None
    problem = setup_problem(model, data, start, loss_fn=loss_fn, **kwargs)
    log = _log_names(start, kwargs["transform"])
    objective, start = refit(problem, start)
    return _Setup(
        problem=problem,
        optimum=start,
        objective=objective,
        threshold=threshold,
        alpha=used_alpha,
        log=log,
    )


def predicted_values(
    problem: _Problem,
    prediction: Prediction,
    pars: dict[str, float],
    times: Sequence[float] | None = None,
) -> Array:
    """Values of a prediction, with a clear error if it can't be computed."""
    try:
        values = problem.predict(prediction, problem.condition(prediction), pars, times)
    except KeyError as e:
        msg = f"Unknown prediction target {prediction.target!r}"
        raise ValueError(msg) from e
    if values is None:
        msg = (
            f"Simulating the prediction of {prediction.target!r} at the optimum failed"
        )
        raise ValueError(msg)
    return values


type _Side = tuple[list[Step], SideStatus]


def _profile_sides(
    setup: _Setup,
    prediction: Prediction,
    *,
    sigma: float | None,
    max_steps: int,
    parallel: bool,
    max_workers: int | None,
) -> tuple[_Walker, Step, tuple[_Side, _Side]]:
    """Both sides of a prediction profile, or of a validation profile with `sigma`."""
    if not isinstance(prediction, Prediction):
        msg = f"prediction has to be an identify.Prediction, not {type(prediction).__name__}"
        raise TypeError(msg)
    problem, optimum = setup.problem, setup.optimum
    value = predicted_values(problem, prediction, optimum)
    s = float(
        prediction_scale(
            problem,
            prediction,
            model=problem.condition(prediction),
            optimum=optimum,
            values=value,
            log=setup.log,
        )[0]
    )
    walker = _Walker(
        problem=problem,
        prediction=prediction,
        objective=setup.objective,
        sigma=_SIGMA * s if sigma is None else sigma,
        validation=sigma is not None,
        fixed={},
    )
    start = Step(float(value[0]), 0.0, optimum, info=float(value[0]))
    sides = dict(
        parallelise(
            partial(
                _walk_task,
                walkers={0: walker},
                starts={0: start},
                scales={0: s if sigma is None else math.hypot(s, sigma)},
                threshold=setup.threshold,
                max_steps=max_steps,
            ),
            inputs=[(d, (0, d)) for d in (-1, 1)],
            parallel=parallel,
            max_workers=max_workers,
            tqdm_desc="validation profile"
            if sigma is not None
            else "prediction profile",
        )
    )
    return walker, start, (sides[-1], sides[1])


###############################################################################
# Entry points
###############################################################################


def prediction_profile(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    prediction: Prediction,
    loss_fn: LossFn | None = None,
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
    callback: Callback | None = None,
) -> PredictionProfile:
    """Profile likelihood and confidence interval of a model prediction.

    The profile of a prediction `g` is the best objective of all parameters
    predicting a value `z`, `min objective(θ) s.t. g(θ) = z`; thresholding
    it gives a likelihood-based confidence interval of the prediction. A
    prediction can be well determined even if the parameters aren't.

    The profile is traced by refits with a pseudo data point `z ± sigma` of the
    prediction, stepping `z` away from the optimum's prediction in both
    directions (in parallel) with the step control of `profile_likelihood`.
    The best fit `θ*` with the pseudo data point is the best fit among all
    parameters predicting `g(θ*)`, so each refit gives an exact point
    `(g(θ*), objective(θ*))` of the profile. `sigma` is half the linearised
    standard deviation of the prediction, which only sets how closely the
    prediction follows `z`. A side stops at the threshold, after `max_steps`
    fits or when the prediction stops moving (e.g. parameters at their
    bounds), then the most extreme prediction found bounds the interval.

    Takes the same arguments as `profile_likelihood`, see there.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        Data of the fit, as for `profile_likelihood`
    parameters
        Fitted parameters (and initial values), e.g. a fit result. They are
        refitted first, so they only need to be close to the optimum
    prediction
        The predicted output, time and condition, see `Prediction`
    loss_fn
        Loss of the fits, the same as for the optimum
    alpha
        Confidence level of the interval, for likelihood losses
    threshold
        Increase of the objective bounding the interval, required for
        losses that aren't likelihoods
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits
    protocol
        Protocol of a protocol time course fit
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter
    prior
        Priors of the parameters, as in the fit functions
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit these parameters in log10 space
    postprocess_fn
        Applied to the simulation before the loss and the prediction
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    max_steps
        Maximum number of fits per side
    parallel
        Whether to profile both sides in parallel
    max_workers
        Maximum number of worker processes
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every refit, see `mxlpy.fit.callbacks`. Has to be picklable with
        `parallel=True`

    Examples
    --------
        >>> pred = identify.prediction_profile(
        ...     model, data, res, loss_fn=fit.GaussianNll(0.05),
        ...     prediction=identify.Prediction("x", time=30.0))
        >>> pred.ci
        (0.41, 0.57)
        >>> plot.prediction_profile(pred)

    """
    setup = _setup(
        model,
        data,
        parameters,
        loss_fn=loss_fn,
        alpha=alpha,
        threshold=threshold,
        max_steps=max_steps,
        what="Prediction profiles",
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
    walker, start, sides = _profile_sides(
        setup,
        prediction,
        sigma=None,
        max_steps=max_steps,
        parallel=parallel,
        max_workers=max_workers,
    )
    return _prediction_profile(
        walker,
        start=start,
        sides=sides,
        threshold=setup.threshold,
        alpha=setup.alpha,
        columns=list(setup.optimum),
    )


def validation_profile(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    prediction: Prediction,
    sigma: float,
    loss_fn: LossFn | None = None,
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
    callback: Callback | None = None,
) -> ValidationProfile:
    """Validation profile: which new data points are consistent with the model.

    For a new (planned or measured) data point `z ± sigma` of a prediction,
    the validation profile is the increase of the best objective including
    the data point over the optimum without it. A measured value is
    consistent with the calibrated model (likelihood-ratio test) if the
    increase is below the threshold, see `ValidationProfile.is_consistent`.
    The consistent values are wider than the prediction's confidence
    interval, as they include the measurement error.

    Steps `z` away from the optimum's prediction in both directions with the
    step control of `profile_likelihood`, refitting with the data point.
    Takes the same arguments as `prediction_profile`, see there.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        Data of the fit, as for `profile_likelihood`
    parameters
        Fitted parameters (and initial values), e.g. a fit result
    prediction
        What the new data point measures, see `Prediction`
    sigma
        Measurement error (standard deviation) of the new data point
    loss_fn
        Loss of the fits, a likelihood (`fit.GaussianNll`) unless
        `threshold` is given
    alpha
        Confidence level of the test, for likelihood losses
    threshold
        Increase of the objective bounding the consistent values
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits
    protocol
        Protocol of a protocol time course fit
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter
    prior
        Priors of the parameters, as in the fit functions
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit these parameters in log10 space
    postprocess_fn
        Applied to the simulation before the loss and the prediction
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    max_steps
        Maximum number of fits per side
    parallel
        Whether to profile both sides in parallel
    max_workers
        Maximum number of worker processes
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every refit, see `mxlpy.fit.callbacks`. Has to be picklable with
        `parallel=True`

    Examples
    --------
        >>> val = identify.validation_profile(
        ...     model, data, res, loss_fn=fit.GaussianNll(0.05), sigma=0.05,
        ...     prediction=identify.Prediction("x", time=30.0))
        >>> val.is_consistent(0.42)
        True

    """
    if not (math.isfinite(sigma) and sigma > 0):
        msg = f"sigma has to be positive, got {sigma}"
        raise ValueError(msg)
    setup = _setup(
        model,
        data,
        parameters,
        loss_fn=loss_fn,
        alpha=alpha,
        threshold=threshold,
        max_steps=max_steps,
        what="Validation profiles",
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
    _, start, ((lower, lower_status), (upper, upper_status)) = _profile_sides(
        setup,
        prediction,
        sigma=float(sigma),
        max_steps=max_steps,
        parallel=parallel,
        max_workers=max_workers,
    )
    index, points, pars = prediction_points(
        start, lower, upper, list(setup.optimum), by="z"
    )
    data_delta = [p.delta - ((p.info - p.u) / sigma) ** 2 / 2 for p in points]
    return ValidationProfile(
        prediction=prediction,
        value=float(start.info),
        sigma=float(sigma),
        threshold=setup.threshold,
        alpha=setup.alpha,
        delta=pd.Series([p.delta for p in points], index=index, name="delta"),
        predicted=pd.Series([p.info for p in points], index=index, name="prediction"),
        data_delta=pd.Series(data_delta, index=index, name="data_delta"),
        parameters=pars,
        lower_status=lower_status,
        upper_status=upper_status,
    )


###############################################################################
# Bands
###############################################################################


@dataclass(frozen=True, kw_only=True)
class _Band:
    """Everything a side of a prediction band needs."""

    problem: _Problem
    prediction: Prediction
    times: list[float]
    optimum: dict[str, float]
    objective: float
    values: Array
    scales: Array
    threshold: float
    max_steps: int

    def walker(self, k: int) -> _Walker:
        return _Walker(
            problem=self.problem,
            prediction=self.prediction.at(self.times[k]),
            objective=self.objective,
            sigma=_SIGMA * float(self.scales[k]),
            validation=False,
            fixed={},
        )

    def start(self, k: int) -> Step:
        return Step(
            float(self.values[k]), 0.0, self.optimum, info=float(self.values[k])
        )

    def profile(self, k: int, direction: int) -> tuple[list[Step], SideStatus]:
        return self.walker(k).walk(
            direction,
            start=self.start(k),
            scale=float(self.scales[k]),
            threshold=self.threshold,
            max_steps=self.max_steps,
        )

    def bound(self, k: int, points: list[Step], status: SideStatus, side: int) -> float:
        start = self.start(k)
        index, ordered, _ = prediction_points(start, points, [], [], by="prediction")
        delta = pd.Series([p.delta for p in ordered], index=index)
        return _side(delta, start.info, self.threshold, status, 1 if side > 0 else -1)

    def correct(
        self, k: int, direction: int, edge: Step, edge_scale: float
    ) -> tuple[float, Step] | None:
        """Bound at time `k` from the bound `edge` of the previous time.

        Predicts z from the edge's parameters and its offset `z - g`
        (proportional to the scale), then corrects z by secant steps on
        `sqrt(Δ)`, which is linear in z for a linear model, through the
        optimum (`z = g0`, `Δ = 0`) and the fits so far.
        """
        walker = self.walker(k)
        at = walker.at_fn()
        g0, s = float(self.values[k]), float(self.scales[k])
        predicted = self.problem.predict(
            walker.prediction,
            self.problem.condition(walker.prediction),
            edge.pars,
        )
        if predicted is None:
            return None
        z = float(predicted[0]) + (edge.u - edge.info) * s / edge_scale
        target = math.sqrt(self.threshold)
        prev_z, prev_w = g0, 0.0
        start = edge.pars
        for _ in range(_CORRECT):
            if (point := at(z, start)) is None:
                return None
            if abs(point.delta - self.threshold) < _CORRECT_TOL * self.threshold:
                # Linear correction of the prediction to the threshold
                g = g0 + (point.info - g0) * math.sqrt(
                    self.threshold / max(point.delta, 1e-300)
                )
                return g, point
            w = math.sqrt(max(point.delta, 0.0))
            if w == prev_w:
                return None
            z, prev_z, prev_w = (
                prev_z + (target - prev_w) * (z - prev_z) / (w - prev_w),
                z,
                w,
            )
            start = point.pars
            if not math.isfinite(z) or (z - g0) * direction <= 0:
                return None
        return None


def _band_side(direction: int, *, band: _Band) -> tuple[list[float], list[SideStatus]]:
    """Bounds of one side of an integrated band, marching along time."""
    bounds: list[float] = []
    statuses: list[SideStatus] = []
    edge: Step | None = None
    edge_scale = 1.0
    for k in range(len(band.times)):
        if (
            edge is not None
            and (res := band.correct(k, direction, edge, edge_scale)) is not None
        ):
            bound, edge = res
            bounds.append(bound)
            statuses.append("threshold")
        else:
            points, status = band.profile(k, direction)
            bounds.append(band.bound(k, points, status, direction))
            statuses.append(status)
            edge = points[-1] if status == "threshold" else None
        edge_scale = float(band.scales[k])
    return bounds, statuses


def prediction_band(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    prediction: str | Prediction,
    times: Sequence[float] | Array,
    loss_fn: LossFn | None = None,
    method: Literal["integrated", "pointwise"] = "integrated",
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
    callback: Callback | None = None,
) -> PredictionBand:
    """Pointwise likelihood-based confidence band of a prediction over time.

    The confidence interval of the prediction at each time (see
    `prediction_profile`), which has the right coverage even for sloppy or
    non-identifiable models, where ensemble bands often miss the edges.

    - `integrated` (default) tracks the bounds along time: the refit at the
      bound of one time predicts the one of the next time, which a few
      corrective refits move onto the threshold (about 2-4 fits per time
      and side, the two sides in parallel). Times where that fails get a
      full prediction profile. Much cheaper for dense time grids
    - `pointwise` computes a full prediction profile at every time, all in
      parallel, and keeps them in `PredictionBand.profiles`

    Takes the same arguments as `prediction_profile`, see there.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        Data of the fit, as for `profile_likelihood`
    parameters
        Fitted parameters (and initial values), e.g. a fit result
    prediction
        Name of the predicted output in the fitted condition, or a
        `Prediction` without a time for another condition
    times
        Times of the band
    loss_fn
        Loss of the fits, the same as for the optimum
    method
        `integrated` or `pointwise`, see above
    alpha
        Confidence level of the band at each time, for likelihood losses
    threshold
        Increase of the objective bounding the band
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits
    protocol
        Protocol of a protocol time course fit
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter
    prior
        Priors of the parameters, as in the fit functions
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit these parameters in log10 space
    postprocess_fn
        Applied to the simulation before the loss and the prediction
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    max_steps
        Maximum number of fits per side of each profile
    parallel
        Whether to compute in parallel
    max_workers
        Maximum number of worker processes
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every refit, see `mxlpy.fit.callbacks`. Has to be picklable with
        `parallel=True`

    Examples
    --------
        >>> band = identify.prediction_band(model, data, res, prediction="x",
        ...                                 times=np.linspace(0, 60, 31),
        ...                                 loss_fn=fit.GaussianNll(0.05))
        >>> plot.prediction_band(band)

    """
    if isinstance(prediction, str):
        prediction = Prediction(target=prediction)
    if prediction.time is not None:
        msg = "The prediction of a band takes its times from times=, not time="
        raise ValueError(msg)
    times = sorted({float(t) for t in np.asarray(times, dtype=float)})
    if not times:
        msg = "A prediction band needs times"
        raise ValueError(msg)
    if method not in ("integrated", "pointwise"):
        msg = f"method has to be 'integrated' or 'pointwise', got {method!r}"
        raise ValueError(msg)
    setup = _setup(
        model,
        data,
        parameters,
        loss_fn=loss_fn,
        alpha=alpha,
        threshold=threshold,
        max_steps=max_steps,
        what="Prediction bands",
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
    problem, optimum = setup.problem, setup.optimum
    template = prediction.at(times[0])
    values = predicted_values(problem, template, optimum, times)
    scales = prediction_scale(
        problem,
        template,
        model=problem.condition(template),
        optimum=optimum,
        values=values,
        log=setup.log,
        times=times,
    )
    band = _Band(
        problem=problem,
        prediction=prediction,
        times=times,
        optimum=optimum,
        objective=setup.objective,
        values=values,
        scales=scales,
        threshold=setup.threshold,
        max_steps=max_steps,
    )
    index = pd.Index(times, name="time")

    profiles = None
    if method == "pointwise":
        walkers = {k: band.walker(k) for k in range(len(times))}
        sides = dict(
            parallelise(
                partial(
                    _walk_task,
                    walkers=walkers,
                    starts={k: band.start(k) for k in walkers},
                    scales={k: float(scales[k]) for k in walkers},
                    threshold=setup.threshold,
                    max_steps=max_steps,
                ),
                inputs=[((k, d), (k, d)) for k in walkers for d in (-1, 1)],
                parallel=parallel,
                max_workers=max_workers,
                tqdm_desc="prediction band",
            )
        )
        profiles = {
            times[k]: _prediction_profile(
                walkers[k],
                start=band.start(k),
                sides=(sides[(k, -1)], sides[(k, 1)]),
                threshold=setup.threshold,
                alpha=setup.alpha,
                columns=list(optimum),
            )
            for k in walkers
        }
        lower = [p.ci_lower for p in profiles.values()]
        upper = [p.ci_upper for p in profiles.values()]
        lower_status = [p.lower_status for p in profiles.values()]
        upper_status = [p.upper_status for p in profiles.values()]
    else:
        tracked = dict(
            parallelise(
                partial(_band_side, band=band),
                inputs=[(d, d) for d in (-1, 1)],
                parallel=parallel,
                max_workers=max_workers,
                tqdm_desc="prediction band",
            )
        )
        (lower, lower_status), (upper, upper_status) = tracked[-1], tracked[1]

    return PredictionBand(
        prediction=prediction,
        threshold=setup.threshold,
        alpha=setup.alpha,
        method=method,
        value=pd.Series(values, index=index, name=prediction.target),
        lower=pd.Series(lower, index=index, name="lower"),
        upper=pd.Series(upper, index=index, name="upper"),
        lower_status=pd.Series(lower_status, index=index, name="lower_status"),
        upper_status=pd.Series(upper_status, index=index, name="upper_status"),
        profiles=profiles,
    )
