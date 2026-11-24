# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Refits with fixed parameters, shared by the identifiability analyses."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit._gradient import GradientResidual, transformed
from mxlpy.fit._joint import _with_derivatives
from mxlpy.fit._single import _evaluate_loss, _fit_data, _settings_for
from mxlpy.fit.abstract import _split_p0
from mxlpy.fit.losses import GaussianNll, _is_likelihood, _sigma_array
from mxlpy.fit.penalties import collect_penalties, fix_penalties, penalised
from mxlpy.minimizers.abstract import OptimisationState
from mxlpy.runners import default_runner, precompile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import FitResidual, PostprocessFn, _Settings
    from mxlpy.fit.penalties import Penalty
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
    from mxlpy.types import Array

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

# A refit improving the objective by more than this fraction of the threshold
# means the optimum it started from was a local one
BETTER_OPTIMUM = 1e-2
# Unobserved second time point of predictions at t = 0 only, as the
# integrators need a final time after the start
_AFTER_START = 1e-9


@dataclass(frozen=True, eq=False)
class Prediction:
    """A model prediction: one output at one time in one condition.

    Parameters
    ----------
    target
        Name of the predicted output: a variable, flux or derived value, or a
        column added by the `postprocess_fn` of the analysis
    time
        Time of the prediction, None for the steady state
    parameters
        Parameter values of the condition, overriding the fitted model's
        (fitted values of these parameters don't apply to the prediction)
    y0
        Initial values of the condition, overriding the fitted model's and
        fitted initial values
    protocol
        Protocol of the condition, as for `fit.protocol_time_course`.
        Predictions at a time default to the protocol of a protocol fit

    Examples
    --------
        >>> identify.Prediction("ERK_p", time=30.0, parameters={"EGF": 10.0})

    """

    target: str
    time: float | None = None
    _: KW_ONLY
    parameters: Mapping[str, float] | None = None
    y0: Mapping[str, float] | None = None
    protocol: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Check the time."""
        if self.time is not None and not (math.isfinite(self.time) and self.time >= 0):
            msg = f"The time of a prediction has to be non-negative, got {self.time}"
            raise ValueError(msg)

    def at(self, time: float | None) -> Prediction:
        """The same prediction at another time."""
        return Prediction(
            target=self.target,
            time=time,
            parameters=self.parameters,
            y0=self.y0,
            protocol=self.protocol,
        )

    @property
    def overridden(self) -> set[str]:
        """Names whose values the condition sets, so they aren't fitted in it."""
        return set(self.parameters or {}) | set(self.y0 or {})


def _prediction_data(
    prediction: Prediction, values: Sequence[float], times: Sequence[float]
) -> pd.Series | pd.DataFrame:
    """Pseudo data of a prediction: `values` of the target at `times`."""
    if prediction.time is None:
        return pd.Series({prediction.target: float(values[0])}, dtype=float)
    data = pd.DataFrame(
        {prediction.target: np.asarray(values, dtype=float)},
        index=pd.Index(np.asarray(times, dtype=float), name="time"),
    )
    if float(np.max(np.asarray(times, dtype=float))) == 0.0:
        data.loc[_AFTER_START] = np.nan
    return data


@dataclass(frozen=True, kw_only=True)
class Constraint:
    """Pseudo data point `z ± sigma` of a prediction, see `_Problem.fit_constrained`.

    `model` is the model of the prediction's condition (see
    `_Problem.condition`), a copy owned by the caller.
    """

    prediction: Prediction
    model: _Model
    z: float
    sigma: float


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
        return _evaluate_loss(
            model,
            self.data,
            self.protocol,
            y0=y0,
            runner=self.runner,
            loss_fn=self.loss_fn,
            postprocess_fn=self.postprocess_fn,
            standard_scale=self.standard_scale,
        )

    def fit(
        self,
        model: _Model,
        fixed: dict[str, float],
        p0: dict[str, float],
        extra: Sequence[Penalty] = (),
        *,
        data: pd.Series | pd.DataFrame | None = None,
        loss_fn: LossFn | None = None,
    ) -> tuple[float, dict[str, float]] | None:
        """Objective and parameters of the best fit with `fixed` values.

        `extra` penalties are minimised as well, but left out of the returned
        objective (loss + the problem's penalties). `data` and `loss_fn`
        replace the problem's. `model` is modified, pass a copy owned by the
        caller.
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
            "minimizer": self.minimizer,
            "y0": y0,
            "runner": self.runner,
            "loss_fn": self.loss_fn if loss_fn is None else loss_fn,
            "postprocess_fn": self.postprocess_fn,
            "standard_scale": self.standard_scale,
            "bounds": bounds,
            "penalty": [*penalties, *extra],
            "transform": transform,
            "as_deepcopy": False,
        }
        data = self.data if data is None else data
        match _fit_data(model, data, self.protocol, **kwargs).value:
            case fit.Fit() as best if math.isfinite(best.objective):
                if not extra:
                    return best.objective + constant, {**best.best_pars, **fixed}
                pars = best.best_pars
                penalty = sum(
                    float(p(np.array([pars[k] for k in p.names], dtype=float)))
                    for p in penalties
                )
                return best.loss + penalty + constant, {**pars, **fixed}
            case _:
                return None

    ###########################################################################
    # Predictions
    ###########################################################################

    def condition(self, prediction: Prediction) -> _Model:
        """A copy of the model with the parameters of the prediction's condition."""
        if isinstance(self.model, SteadyStateModelBuilder) and (
            prediction.time is not None or prediction.y0
        ):
            msg = "A SteadyStateModelBuilder only predicts steady states without y0"
            raise TypeError(msg)
        if prediction.time is None and prediction.protocol is not None:
            msg = "A prediction with a protocol needs a time"
            raise ValueError(msg)
        model = deepcopy(self.model)
        if prediction.parameters:
            model.update_parameters(dict(prediction.parameters))
        return model

    def _protocol(self, prediction: Prediction) -> pd.DataFrame | None:
        """Protocol of a prediction: its own, else the fit's (not at steady state)."""
        if prediction.time is None:
            return None
        return self.protocol if prediction.protocol is None else prediction.protocol

    def _prediction_part(
        self,
        prediction: Prediction,
        *,
        model: _Model,
        data: pd.Series | pd.DataFrame,
        names: Sequence[str],
        fixed: Mapping[str, float],
        loss_fn: LossFn,
    ) -> tuple[FitResidual, _Settings[Any]]:
        """Residual and settings comparing the prediction to `data`."""
        overridden = prediction.overridden
        fixed_p, fixed_v = _split_p0(model, dict(fixed))
        for k in fixed_p:
            if k not in overridden:
                model.update_parameter(k, fixed[k])
        y0 = None
        if not isinstance(model, SteadyStateModelBuilder):
            y0 = {
                **(self.y0 or {}),
                **{k: fixed[k] for k in fixed_v},
                **(prediction.y0 or {}),
            } or None
        return _settings_for(
            model,
            data,
            self._protocol(prediction),
            y0=y0,
            runner=self.runner,
            loss_fn=loss_fn,
            postprocess_fn=self.postprocess_fn,
            standard_scale=False,
            names=[k for k in names if k not in overridden],
        )

    def _prediction_residual(
        self,
        prediction: Prediction,
        model: _Model,
        pars: Mapping[str, float],
        times: Sequence[float],
    ) -> tuple[GradientResidual, dict[str, float]]:
        """Prediction minus zero at `times` and the updates of its fitted names."""
        fn, settings = self._prediction_part(
            prediction,
            model=model,
            data=_prediction_data(prediction, [0.0] * max(len(times), 1), times),
            names=list(pars),
            fixed={},
            loss_fn=GaussianNll(1.0),
        )
        names = settings.p_names + settings.v_names
        return GradientResidual(fn, settings), {k: pars[k] for k in names}

    def predict(
        self,
        prediction: Prediction,
        model: _Model,
        pars: Mapping[str, float],
        times: Sequence[float] | None = None,
    ) -> Array | None:
        """Values of the prediction with `pars` (at `times`), None if it fails."""
        times = [] if prediction.time is None else times or [prediction.time]
        residual, updates = self._prediction_residual(prediction, model, pars, times)
        # Not `residuals`, which zeroes non-finite predictions
        values = residual._vector(residual._predict(updates))  # noqa: SLF001
        if not np.isfinite(values).all():
            return None
        return values

    def prediction_jacobian(
        self,
        prediction: Prediction,
        model: _Model,
        pars: Mapping[str, float],
        times: Sequence[float] | None = None,
    ) -> Array:
        """Derivative of the prediction (at `times`) w.r.t. all of `pars`."""
        times = [] if prediction.time is None else times or [prediction.time]
        residual, updates = self._prediction_residual(prediction, model, pars, times)
        jac = np.zeros((max(len(times), 1), len(pars)))
        columns = [list(pars).index(k) for k in updates]
        if updates:
            jac[:, columns] = residual.jacobian(updates)
        return jac

    def loss_jacobian(self, pars: Mapping[str, float]) -> tuple[Array, float | None]:
        """Residual Jacobian of the data w.r.t. `pars` and its squares factor."""
        fn, settings = _settings_for(
            deepcopy(self.model),
            self.data,
            self.protocol,
            y0=self.y0,
            runner=self.runner,
            loss_fn=self.loss_fn,
            postprocess_fn=self.postprocess_fn,
            standard_scale=self.standard_scale,
            names=list(pars),
        )
        residual = GradientResidual(fn, settings)
        return residual.jacobian(dict(pars)), residual.squares_factor()

    def _merged(
        self, constraint: Constraint
    ) -> tuple[pd.Series | pd.DataFrame, LossFn] | None:
        """The data with the pseudo data point added and its `GaussianNll`.

        Only for a prediction in the fitted condition and an unmeasured entry
        of the data, else None and the prediction is simulated separately.
        """
        prediction, target = constraint.prediction, constraint.prediction.target
        if (
            prediction.parameters
            or prediction.y0
            or prediction.protocol is not None
            or self.protocol is not None
            or not isinstance(self.loss_fn, GaussianNll)
        ):
            return None
        data = self.data
        sigma_values = _sigma_array(self.loss_fn.sigma, data)
        if isinstance(data, pd.Series):
            if prediction.time is not None or (
                target in data.index and np.isfinite(data[target])
            ):
                return None
            sigma = pd.Series(sigma_values, index=data.index, dtype=float)
            merged = data.copy()
            merged[target], sigma[target] = constraint.z, constraint.sigma
            return merged, GaussianNll(sigma)
        time = prediction.time
        if time is None:
            return None
        if int(np.sum(np.asarray(data.index) == time)) > 1:
            # Replicates: the pseudo data point would count once per row
            return None
        if target in data.columns and time in data.index:
            measured = np.asarray(data.loc[[time], target], dtype=float)
            if np.isfinite(measured).any():
                return None
        sigma = pd.DataFrame(sigma_values, index=data.index, columns=data.columns)
        merged = data.copy()
        if target not in merged.columns:
            merged[target] = np.nan
            sigma[target] = np.nan
        if time not in merged.index:
            merged.loc[time] = np.nan
            sigma.loc[time] = np.nan
            merged, sigma = merged.sort_index(), sigma.sort_index()
        merged.loc[time, target] = constraint.z
        sigma.loc[time, target] = constraint.sigma
        return merged, GaussianNll(sigma)

    def fit_constrained(
        self,
        model: _Model,
        fixed: dict[str, float],
        p0: dict[str, float],
        constraint: Constraint,
    ) -> tuple[float, dict[str, float], float] | None:
        """Best fit with `fixed` values and a pseudo data point of a prediction.

        Minimises the objective plus `((g - z) / sigma)² / 2` of the
        prediction `g`. Returns the objective without the pseudo data point,
        the parameters and `g`. `model` (and `constraint.model`) are
        modified, pass copies owned by the caller.
        """
        fixed_p, fixed_v = _split_p0(model, fixed)
        for k in fixed_p:
            model.update_parameter(k, fixed[k])
        y0 = self.y0
        if fixed_v:
            y0 = {**(self.y0 or {}), **{k: fixed[k] for k in fixed_v}}
        penalties, constant = fix_penalties(self.penalties, fixed)
        if constant == math.inf:
            return None
        prediction = constraint.prediction
        if not p0:
            loss = self._evaluate(model, y0)
            g = self.predict(prediction, constraint.model, dict(fixed))
            if not math.isfinite(loss) or g is None:
                return None
            return loss + constant, dict(fixed), float(g[0])
        if (merged := self._merged(constraint)) is not None:
            # One simulation per evaluation instead of two
            res = self.fit(model, fixed, p0, data=merged[0], loss_fn=merged[1])
            if res is None or not math.isfinite(res[0]):
                return None
            g = self.predict(prediction, constraint.model, res[1])
            if g is None:
                return None
            pseudo = ((float(g[0]) - constraint.z) / constraint.sigma) ** 2 / 2
            return res[0] - pseudo, res[1], float(g[0])

        data_fn, data_settings = _settings_for(
            model,
            self.data,
            self.protocol,
            y0=y0,
            runner=self.runner,
            loss_fn=self.loss_fn,
            postprocess_fn=self.postprocess_fn,
            standard_scale=self.standard_scale,
            names=list(p0),
        )
        times = [] if prediction.time is None else [prediction.time]
        parts = [
            (data_fn, data_settings),
            self._prediction_part(
                prediction,
                model=constraint.model,
                data=_prediction_data(prediction, [constraint.z], times),
                names=list(p0),
                fixed=fixed,
                loss_fn=GaussianNll(constraint.sigma),
            ),
        ]

        def loss_fn(updates: dict[str, float]) -> float:
            return sum(fn(updates, settings) for fn, settings in parts)

        residual: Any = _with_derivatives(loss_fn, parts, p0)
        if penalties:
            residual = penalised(residual, penalties)
        bounds, transform = self._subset(list(p0))
        residual, start, minimizer_bounds, back = transformed(
            residual, p0, bounds, transform
        )
        match self.minimizer(residual, start, minimizer_bounds).value:
            case OptimisationState(parameters, value) if math.isfinite(value):
                pars = back(parameters)
            case _:
                return None
        loss = data_fn(pars, data_settings)
        g = self.predict(prediction, constraint.model, {**pars, **fixed})
        if not math.isfinite(loss) or g is None:
            return None
        penalty = sum(
            float(p(np.array([pars[k] for k in p.names], dtype=float)))
            for p in penalties
        )
        return loss + penalty + constant, {**pars, **fixed}, float(g[0])


def likelihood_threshold(
    loss_fn: LossFn, alpha: float, what: str, df: int = 1
) -> float:
    """Likelihood-ratio threshold `chi2.ppf(alpha, df) / 2` of a likelihood loss."""
    if not _is_likelihood(loss_fn):
        msg = (
            f"{what} need a likelihood loss like fit.GaussianNll, "
            "pass threshold= for other losses"
        )
        raise ValueError(msg)
    if not 0 < alpha < 1:
        msg = f"alpha has to be between 0 and 1, got {alpha}"
        raise ValueError(msg)
    return float(stats.chi2.ppf(alpha, df=df) / 2)


def check_threshold(threshold: float) -> float:
    """A given threshold, which has to be positive."""
    if not (math.isfinite(threshold) and threshold > 0):
        msg = f"threshold has to be positive, got {threshold}"
        raise ValueError(msg)
    return float(threshold)


def start_values(parameters: fit.Fit | Mapping[str, float]) -> dict[str, float]:
    """Fitted parameter values, from a fit result or a mapping."""
    if isinstance(parameters, fit.Fit):
        parameters = parameters.best_pars
    return {k: float(v) for k, v in parameters.items()}


def setup_problem(
    model: _Model,
    data: pd.Series | pd.DataFrame,
    start: dict[str, float],
    *,
    loss_fn: LossFn,
    minimizer: MinimizerProtocol | None,
    y0: dict[str, float] | None,
    protocol: pd.DataFrame | None,
    runner: Any | None,
    bounds: Bounds | None,
    prior: Mapping[str, Prior] | None,
    penalty: Penalty | Sequence[Penalty] | None,
    transform: Transform | None,
    postprocess_fn: PostprocessFn | None,
    standard_scale: bool,
) -> _Problem:
    """Check the inputs and build the problem of fitting the `start` values."""
    p_names, v_names = _split_p0(model, start)
    if unknown := sorted(set(start) - set(p_names) - set(v_names)):
        msg = f"Unknown parameters or variables: {unknown}"
        raise ValueError(msg)
    if isinstance(model, SteadyStateModelBuilder) and y0 is not None:
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)

    model = deepcopy(model)
    runner = default_runner(model) if runner is None else runner
    precompile(runner, model)
    return _Problem(
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


def refit(problem: _Problem, start: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Objective and parameters of the optimum, refitted from `start`."""
    if (best := problem.fit(deepcopy(problem.model), {}, start)) is None:
        msg = "Refitting the optimum failed"
        raise ValueError(msg)
    return best
