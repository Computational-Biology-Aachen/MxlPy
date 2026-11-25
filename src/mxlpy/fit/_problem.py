# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fits of one or several data sets with fixed values, shared by the analyses.

Used by `mxlpy.identify`, `mxlpy.select` and `mxlpy.lab`.
"""

from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit import losses
from mxlpy.fit._gradient import GradientResidual, transformed
from mxlpy.fit._joint import _JointGradient, _minimize_joint
from mxlpy.fit._single import _fit_data, _settings_for
from mxlpy.fit.abstract import (
    _split_p0,
    as_conditions,
    with_weight_bounds,
    without_weights,
)
from mxlpy.fit.callbacks import failed, failure_message, finished, tagged, tracked
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
    from mxlpy.fit.abstract import Data, FitResidual, PostprocessFn, _Settings
    from mxlpy.fit.callbacks import Callback
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
    local
        Condition-specific estimates of the condition, as for
        `fit.Condition`: the model parameter (key) takes the value of the
        estimated name (value), whose uncertainty enters the prediction

    Notes
    -----
    Fitted to a single data set, a prediction defaults to its condition
    (parameters, initial values and protocol of a `fit.Condition`). Fitted to
    several, only the prediction's own settings apply on top of the model.

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
    local: Mapping[str, str] | None = None

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
            local=self.local,
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


def _no_loss(updates: dict[str, float]) -> float:  # noqa: ARG001
    """Placeholder loss of a joint residual only used for its Jacobian."""
    return math.nan


def _is_squares(loss_fn: LossFn) -> bool:
    """Whether a loss is a sum of squares (residual vectors for least squares)."""
    return isinstance(loss_fn, GaussianNll) or loss_fn is losses.mean_squared


@dataclass(frozen=True, kw_only=True, eq=False)
class Part:
    """One data set of a problem: a `fit.Condition` with its resolved loss."""

    name: str
    data: pd.Series | pd.DataFrame
    loss_fn: LossFn
    y0: dict[str, float] | None
    protocol: pd.DataFrame | None
    parameters: dict[str, float]
    local: dict[str, str]


@dataclass(frozen=True, kw_only=True)
class Refit:
    """Result of `_Problem.fit_result`, the fixed values included."""

    objective: float
    loss: float
    penalty: float
    parameters: dict[str, float]


type _Sources = dict[str, str]
"""Model name -> estimated (or fixed) name it takes its value from, per part."""


class _RenamedJoint(_JointGradient):
    """Summed residual of several parts, each reading its own names of the updates.

    A condition-specific name (`fit.Condition.local`) gives a model parameter
    of one part the value of another estimated name.
    """

    def __init__(
        self,
        loss: Any,
        parts: list[GradientResidual],
        p0: dict[str, float],
        sources: list[_Sources],
    ) -> None:
        super().__init__(loss, parts, p0)
        names = list(p0)
        self.sources = sources
        self.names = [list(src) for src in sources]
        self.columns = [[names.index(v) for v in src.values()] for src in sources]

    def _sub(self, i: int, updates: dict[str, float]) -> dict[str, float]:
        return {k: updates[v] for k, v in self.sources[i].items()}


def _joint(
    parts: Sequence[tuple[FitResidual, _Settings[Any]]],
    sources: Sequence[_Sources],
    p0: dict[str, float],
) -> Any:
    """The summed loss of `parts` (with derivatives if they all provide them)."""

    def loss(updates: dict[str, float]) -> float:
        return float(
            sum(
                fn({k: updates[v] for k, v in src.items()}, settings)
                for (fn, settings), src in zip(parts, sources, strict=True)
            )
        )

    if not all(GradientResidual.supports(fn) for fn, _ in parts):
        return loss
    return _RenamedJoint(
        loss,
        [GradientResidual(fn, settings) for fn, settings in parts],
        p0,
        list(sources),
    )


@dataclass(kw_only=True)
class _Problem:
    """A fit of one or several data sets, with some of its parameters held fixed.

    Several data sets (`parts`) are fitted jointly, each in a copy of the
    model with its condition's parameters and condition-specific names. A
    single one is fitted directly.
    """

    model: _Model
    parts: list[Part]
    minimizer: MinimizerProtocol
    runner: Any
    bounds: Bounds
    penalties: list[Penalty]
    transform: Transform | None
    postprocess_fn: PostprocessFn | None
    standard_scale: bool
    callback: Callback | None = None
    _n_refits: int = field(default=0, init=False, repr=False)

    @property
    def y0(self) -> dict[str, float] | None:
        """Initial values of the only data set, None for several."""
        return self.parts[0].y0 if len(self.parts) == 1 else None

    @property
    def protocol(self) -> pd.DataFrame | None:
        """Protocol of the only data set, None for several."""
        return self.parts[0].protocol if len(self.parts) == 1 else None

    @property
    def n_data(self) -> int:
        """Number of finite data points of all data sets."""
        return sum(
            int(np.isfinite(p.data.to_numpy(dtype=float)).sum()) for p in self.parts
        )

    @property
    def _direct(self) -> bool:
        """Whether the only data set is fitted directly, without renames."""
        return len(self.parts) == 1 and not self.parts[0].local

    @cached_property
    def _model_names(self) -> set[str]:
        names = set(self.model.get_addressable_names())
        if not isinstance(self.model, SteadyStateModelBuilder):
            names |= set(self.model.get_variable_names())
        return names

    def _refit_callback(self) -> Callback | None:
        """The callback with a label unique to the next refit."""
        if self.callback is None:
            return None
        self._n_refits += 1
        return tagged(self.callback, f"refit-{os.getpid()}-{self._n_refits}")

    def _subset(self, names: Sequence[str]) -> tuple[Bounds, Transform | None]:
        bounds = {k: v for k, v in self.bounds.items() if k in names}
        if isinstance(self.transform, dict):
            return bounds, {k: v for k, v in self.transform.items() if k in names}
        return bounds, self.transform

    def _sources(
        self,
        part: Part | None,
        names: Sequence[str],
        overridden: set[str] | None = None,
    ) -> _Sources:
        """The fitted model names of a part and the names they read."""
        overridden = set() if overridden is None else overridden
        skip = overridden | (
            set() if part is None else set(part.parameters) | set(part.local)
        )
        out = {k: k for k in names if k in self._model_names and k not in skip}
        if part is not None:
            out |= {
                k: v
                for k, v in part.local.items()
                if v in names and k not in overridden
            }
        return out

    def _settings(
        self,
        model: _Model,
        names: Sequence[str],
        fixed: Mapping[str, float],
    ) -> tuple[list[tuple[FitResidual, _Settings[Any]]], list[_Sources]]:
        """Residuals, settings and sources of all data sets, fitting `names`.

        The first data set uses `model`, the others copies of it. `fixed`
        values (fixed parameters already set in `model`) set fixed initial
        values and condition-specific names.
        """
        variables = self._model_names - set(model.get_addressable_names())
        fixed_v = {k: v for k, v in fixed.items() if k in variables}
        models = [model, *(deepcopy(model) for _ in self.parts[1:])]
        out, sources = [], []
        for part, m in zip(self.parts, models, strict=True):
            if part.parameters:
                m.update_addressable(part.parameters)
            if local_fixed := {
                k: fixed[v] for k, v in part.local.items() if v in fixed
            }:
                m.update_addressable(local_fixed)
            src = self._sources(part, names)
            sources.append(src)
            out.append(
                _settings_for(
                    m,
                    part.data,
                    part.protocol,
                    y0=self._y0(part, fixed_v),
                    runner=self.runner,
                    loss_fn=part.loss_fn,
                    postprocess_fn=self.postprocess_fn,
                    standard_scale=self.standard_scale,
                    names=list(src),
                )
            )
        return out, sources

    @staticmethod
    def _y0(part: Part, fixed_v: Mapping[str, float]) -> dict[str, float] | None:
        if not fixed_v:
            return part.y0
        return {**(part.y0 or {}), **fixed_v}

    def _evaluate(self, model: _Model, fixed: Mapping[str, float]) -> float:
        """Loss without free parameters, there is nothing to fit."""
        parts, _ = self._settings(model, [], fixed)
        return float(sum(fn({}, settings) for fn, settings in parts))

    def _penalty(
        self, penalties: Sequence[Penalty], pars: Mapping[str, float]
    ) -> float:
        return float(
            sum(
                float(p(np.array([pars[k] for k in p.names], dtype=float)))
                for p in penalties
            )
        )

    def fit_result(
        self,
        model: _Model,
        fixed: dict[str, float],
        p0: dict[str, float],
        extra: Sequence[Penalty] = (),
        *,
        data: pd.Series | pd.DataFrame | None = None,
        loss_fn: LossFn | None = None,
    ) -> Refit | None:
        """Best fit with `fixed` values, None if it failed.

        `extra` penalties are minimised as well, but left out of the returned
        objective (loss + the problem's penalties). `data` and `loss_fn`
        replace those of the only data set. `model` is modified, pass a copy
        owned by the caller.
        """
        fixed_p, fixed_v_names = _split_p0(model, fixed)
        model.update_addressable({k: fixed[k] for k in fixed_p})
        penalties, constant = fix_penalties(self.penalties, fixed)
        if constant == math.inf:
            # Outside the support of a prior of a fixed parameter
            return Refit(
                objective=math.inf,
                loss=math.inf,
                penalty=math.inf,
                parameters={**p0, **fixed},
            )
        if not p0:
            loss = self._evaluate(model, fixed)
            if not math.isfinite(loss):
                return None
            return Refit(
                objective=loss + constant,
                loss=loss,
                penalty=constant,
                parameters=dict(fixed),
            )

        bounds, transform = self._subset(list(p0))
        callback = self._refit_callback()
        if not self._direct:
            parts, sources = self._settings(model, list(p0), fixed)
            res = _minimize_joint(
                _joint(parts, sources, p0),
                parts=parts,
                p0=p0,
                minimizer=self.minimizer,
                bounds=bounds,
                prior=None,
                penalty=[*penalties, *extra],
                callback=callback,
                transform=transform,
            ).value
            if isinstance(res, Exception) or not math.isfinite(res.objective):
                return None
            pars, loss = res.best_pars, res.loss
            penalty = res.penalty if not extra else self._penalty(penalties, pars)
        else:
            part = self.parts[0]
            if part.parameters:
                model.update_addressable(part.parameters)
            match _fit_data(
                model,
                part.data if data is None else data,
                part.protocol,
                p0=p0,
                minimizer=self.minimizer,
                y0=self._y0(part, {k: fixed[k] for k in fixed_v_names}),
                runner=self.runner,
                loss_fn=part.loss_fn if loss_fn is None else loss_fn,
                postprocess_fn=self.postprocess_fn,
                standard_scale=self.standard_scale,
                bounds=bounds,
                penalty=[*penalties, *extra],
                transform=transform,
                as_deepcopy=False,
                callback=callback,
            ).value:
                case fit.Fit() as best if math.isfinite(best.objective):
                    pars, loss = best.best_pars, best.loss
                    penalty = (
                        best.penalty if not extra else self._penalty(penalties, pars)
                    )
                case _:
                    return None
        return Refit(
            objective=loss + penalty + constant,
            loss=loss,
            penalty=penalty + constant,
            parameters={**pars, **fixed},
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

        See `fit_result`.
        """
        res = self.fit_result(model, fixed, p0, extra, data=data, loss_fn=loss_fn)
        return None if res is None else (res.objective, res.parameters)

    ###########################################################################
    # Predictions
    ###########################################################################
    # A prediction's own parameters, y0 and protocol define its condition.
    # With a single data set, it otherwise is that data set's condition.

    @property
    def _single(self) -> Part | None:
        return self.parts[0] if len(self.parts) == 1 else None

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
        if (single := self._single) is not None and single.parameters:
            model.update_addressable(single.parameters)
        if prediction.parameters:
            model.update_addressable(dict(prediction.parameters))
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
    ) -> tuple[FitResidual, _Settings[Any], _Sources]:
        """Residual, settings and sources comparing the prediction to `data`."""
        overridden = prediction.overridden
        single = self._single
        fixed_p, fixed_v = _split_p0(model, dict(fixed))
        model.update_addressable({k: fixed[k] for k in fixed_p if k not in overridden})
        if single is not None and (
            local_fixed := {
                k: fixed[v]
                for k, v in single.local.items()
                if v in fixed and k not in overridden
            }
        ):
            model.update_addressable(local_fixed)
        y0 = None
        if not isinstance(model, SteadyStateModelBuilder):
            y0 = {
                **(self.y0 or {}),
                **{k: fixed[k] for k in fixed_v},
                **(prediction.y0 or {}),
            } or None
        sources = self._sources(single, names, overridden)
        if local := dict(prediction.local or {}):
            if local_fixed := {k: fixed[v] for k, v in local.items() if v in fixed}:
                model.update_addressable(local_fixed)
            sources = {k: v for k, v in sources.items() if k not in local}
            sources |= {k: v for k, v in local.items() if v in names}
        fn, settings = _settings_for(
            model,
            data,
            self._protocol(prediction),
            y0=y0,
            runner=self.runner,
            loss_fn=loss_fn,
            postprocess_fn=self.postprocess_fn,
            standard_scale=False,
            names=list(sources),
        )
        return fn, settings, sources

    def _prediction_residual(
        self,
        prediction: Prediction,
        model: _Model,
        pars: Mapping[str, float],
        times: Sequence[float],
    ) -> tuple[GradientResidual, _Sources]:
        """Prediction minus zero at `times` and the sources of its fitted names."""
        fn, settings, sources = self._prediction_part(
            prediction,
            model=model,
            data=_prediction_data(prediction, [0.0] * max(len(times), 1), times),
            names=list(pars),
            fixed={},
            loss_fn=GaussianNll(1.0),
        )
        return GradientResidual(fn, settings), sources

    def predict(
        self,
        prediction: Prediction,
        model: _Model,
        pars: Mapping[str, float],
        times: Sequence[float] | None = None,
    ) -> Array | None:
        """Values of the prediction with `pars` (at `times`), None if it fails."""
        times = [] if prediction.time is None else times or [prediction.time]
        residual, sources = self._prediction_residual(prediction, model, pars, times)
        updates = {k: pars[v] for k, v in sources.items()}
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
        residual, sources = self._prediction_residual(prediction, model, pars, times)
        jac = np.zeros((max(len(times), 1), len(pars)))
        if sources:
            columns = [list(pars).index(v) for v in sources.values()]
            jac[:, columns] = residual.jacobian(
                {k: pars[v] for k, v in sources.items()}
            )
        return jac

    def loss_jacobian(self, pars: Mapping[str, float]) -> tuple[Array, float | None]:
        """Residual Jacobian of the data w.r.t. `pars` and its squares factor."""
        parts, sources = self._settings(deepcopy(self.model), list(pars), {})
        if self._direct:
            residual = GradientResidual(*parts[0])
            return residual.jacobian(dict(pars)), residual.squares_factor()
        joint = _joint(parts, sources, dict(pars))
        factor = (
            None if not isinstance(joint, _JointGradient) else joint.squares_factor()
        )
        if factor is None:
            msg = "Several data sets need sum-of-squares losses (GaussianNll or mean_squared)"
            raise TypeError(msg)
        return joint.jacobian(dict(pars)), factor

    def _merged(
        self, constraint: Constraint
    ) -> tuple[pd.Series | pd.DataFrame, LossFn] | None:
        """The data with the pseudo data point added and its `GaussianNll`.

        Only for a prediction in the fitted condition and an unmeasured entry
        of the data, else None and the prediction is simulated separately.
        """
        prediction, target = constraint.prediction, constraint.prediction.target
        if not self._direct:
            return None
        part = self.parts[0]
        if (
            prediction.parameters
            or prediction.y0
            or prediction.protocol is not None
            or part.protocol is not None
            or part.parameters
            or not isinstance(part.loss_fn, GaussianNll)
        ):
            return None
        data = part.data
        sigma_values = _sigma_array(part.loss_fn.sigma, data)
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
        fixed_p, _ = _split_p0(model, fixed)
        model.update_addressable({k: fixed[k] for k in fixed_p})
        penalties, constant = fix_penalties(self.penalties, fixed)
        if constant == math.inf:
            return None
        prediction = constraint.prediction
        if not p0:
            loss = self._evaluate(model, fixed)
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

        t0 = time.perf_counter()
        callback = self._refit_callback()
        data_parts, data_sources = self._settings(model, list(p0), fixed)
        times = [] if prediction.time is None else [prediction.time]
        fn, settings, src = self._prediction_part(
            prediction,
            model=constraint.model,
            data=_prediction_data(prediction, [constraint.z], times),
            names=list(p0),
            fixed=fixed,
            loss_fn=GaussianNll(constraint.sigma),
        )
        residual: Any = _joint([*data_parts, (fn, settings)], [*data_sources, src], p0)
        if penalties:
            residual = penalised(residual, penalties)
        bounds, transform = self._subset(list(p0))
        models = [s.model for _, s in data_parts]
        bounds = with_weight_bounds(models, p0, bounds)
        transform = without_weights(models, p0, transform)
        residual, start, minimizer_bounds, back = transformed(
            residual, p0, bounds, transform
        )
        residual = tracked(residual, callback, back=back)
        match self.minimizer(residual, start, minimizer_bounds).value:
            case OptimisationState(parameters, value) if math.isfinite(value):
                pars = back(parameters)
            case _ as e:
                failed(
                    callback,
                    residual,
                    start=None,
                    message=failure_message(e)
                    if isinstance(e, Exception)
                    else "No finite objective",
                    duration=time.perf_counter() - t0,
                )
                return None
        loss = sum(
            fn({k: pars[v] for k, v in s.items()}, settings)
            for (fn, settings), s in zip(data_parts, data_sources, strict=True)
        )
        g = self.predict(prediction, constraint.model, {**pars, **fixed})
        if not math.isfinite(loss) or g is None:
            failed(
                callback,
                residual,
                start=None,
                message="Simulation failed at the optimum",
                duration=time.perf_counter() - t0,
            )
            return None
        penalty = self._penalty(penalties, pars)
        finished(
            callback,
            residual,
            start=None,
            value=loss + penalty + constant,
            parameters={**pars, **fixed},
            duration=time.perf_counter() - t0,
        )
        return loss + penalty + constant, {**pars, **fixed}, float(g[0])


def condition_losses(data: Data, loss_fn: LossFn | None) -> list[LossFn]:
    """The loss of every data set: its `GaussianNll` if it has a sigma, else `loss_fn`."""
    return [c.loss(loss_fn) for c in as_conditions(data).values()]


def is_likelihood(losses: Sequence[LossFn]) -> bool:
    """Whether all losses are likelihoods (see `fit.GaussianNll`)."""
    return all(_is_likelihood(i) for i in losses)


def likelihood_threshold(
    losses: Sequence[LossFn], alpha: float, what: str, df: int = 1
) -> float:
    """Likelihood-ratio threshold `chi2.ppf(alpha, df) / 2` of likelihood losses."""
    if not is_likelihood(losses):
        msg = (
            f"{what} need a likelihood loss like fit.GaussianNll (or a sigma for "
            "every condition), pass threshold= for other losses"
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


def make_parts(
    model: _Model,
    data: Data,
    loss_fn: LossFn | None,
    *,
    y0: Mapping[str, float] | None,
    protocol: pd.DataFrame | None,
) -> list[Part]:
    """The data sets of a problem, checked against the model."""
    parts = [
        Part(
            name=name,
            data=c.data,
            loss_fn=c.loss(loss_fn),
            y0=None if c.y0 is None else dict(c.y0),
            protocol=c.protocol,
            parameters=dict(c.parameters or {}),
            local=dict(c.local or {}),
        )
        for name, c in as_conditions(data, y0=y0, protocol=protocol).items()
    ]
    if isinstance(model, SteadyStateModelBuilder) and any(
        p.y0 is not None or p.protocol is not None for p in parts
    ):
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)
    addressable = set(model.get_addressable_names())
    for part in parts:
        if unknown := sorted((set(part.parameters) | set(part.local)) - addressable):
            msg = f"Condition {part.name!r} sets unknown parameters {unknown}"
            raise ValueError(msg)
        if both := sorted(set(part.parameters) & set(part.local)):
            msg = f"Condition {part.name!r} both sets and renames {both}"
            raise ValueError(msg)
    return parts


def check_estimable(parts: Sequence[Part], names: Sequence[str]) -> None:
    """Raise for estimated names no data set depends on, or unknown names."""
    if fixed := sorted(set(names).intersection(*(p.parameters for p in parts))):
        msg = f"Every condition sets {fixed}, so they can't be estimated"
        raise ValueError(msg)
    for part in parts:
        if missing := sorted(set(part.local.values()) - set(names)):
            msg = (
                f"Condition-specific names of {part.name!r} aren't estimated: {missing}"
            )
            raise ValueError(msg)


def pick_minimizer(
    parts: Sequence[Part], minimizer: MinimizerProtocol | None
) -> MinimizerProtocol:
    """Least squares by default, quasi-Newton if residual vectors can't be stacked.

    Joint residual vectors need sum-of-squares losses.
    """
    joint = len(parts) > 1 or any(p.local for p in parts)
    squares = all(_is_squares(p.loss_fn) for p in parts)
    if minimizer is None:
        if joint and not squares:
            return fit.LocalScipyMinimizer()
        return fit.LeastSquaresMinimizer()
    if joint and not squares and isinstance(minimizer, fit.LeastSquaresMinimizer):
        msg = (
            "LeastSquaresMinimizer fits several data sets only with sum-of-squares "
            "losses (GaussianNll, mean_squared or a sigma per condition); use "
            "LocalScipyMinimizer for others"
        )
        raise ValueError(msg)
    return minimizer


def setup_problem(
    model: _Model,
    data: Data,
    start: dict[str, float],
    *,
    loss_fn: LossFn | None,
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
    callback: Callback | None = None,
) -> _Problem:
    """Check the inputs and build the problem of fitting the `start` values."""
    parts = make_parts(model, data, loss_fn, y0=y0, protocol=protocol)
    p_names, v_names = _split_p0(model, start)
    local = {v for p in parts for v in p.local.values()}
    if unknown := sorted(set(start) - set(p_names) - set(v_names) - local):
        msg = f"Unknown parameters or variables: {unknown}"
        raise ValueError(msg)
    check_estimable(parts, list(start))

    model = deepcopy(model)
    runner = default_runner(model) if runner is None else runner
    precompile(runner, model)
    return _Problem(
        model=model,
        parts=parts,
        minimizer=pick_minimizer(parts, minimizer),
        runner=runner,
        bounds={} if bounds is None else dict(bounds),
        penalties=collect_penalties(prior, penalty, list(start)),
        transform=transform,
        postprocess_fn=postprocess_fn,
        standard_scale=standard_scale,
        callback=callback,
    )


def refit(problem: _Problem, start: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Objective and parameters of the optimum, refitted from `start`."""
    if (best := problem.fit(deepcopy(problem.model), {}, start)) is None:
        msg = "Refitting the optimum failed"
        raise ValueError(msg)
    return best
