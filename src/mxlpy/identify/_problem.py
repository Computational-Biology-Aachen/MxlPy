# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Refits with fixed parameters, shared by the identifiability analyses."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit._single import _evaluate_loss, _fit_data
from mxlpy.fit.abstract import _split_p0
from mxlpy.fit.losses import _is_likelihood
from mxlpy.fit.penalties import collect_penalties, fix_penalties
from mxlpy.runners import default_runner, precompile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.fit.penalties import Penalty
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

# A refit improving the objective by more than this fraction of the threshold
# means the optimum it started from was a local one
BETTER_OPTIMUM = 1e-2


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
    ) -> tuple[float, dict[str, float]] | None:
        """Objective and parameters of the best fit with `fixed` values.

        `extra` penalties are minimised as well, but left out of the returned
        objective (loss + the problem's penalties). `model` is modified, pass
        a copy owned by the caller.
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
            "loss_fn": self.loss_fn,
            "postprocess_fn": self.postprocess_fn,
            "standard_scale": self.standard_scale,
            "bounds": bounds,
            "penalty": [*penalties, *extra],
            "transform": transform,
            "as_deepcopy": False,
        }
        match _fit_data(model, self.data, self.protocol, **kwargs).value:
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


def likelihood_threshold(loss_fn: LossFn, alpha: float, what: str) -> float:
    """Likelihood-ratio threshold `chi2.ppf(alpha, 1) / 2` of a likelihood loss."""
    if not _is_likelihood(loss_fn):
        msg = (
            f"{what} need a likelihood loss like fit.GaussianNll, "
            "pass threshold= for other losses"
        )
        raise ValueError(msg)
    if not 0 < alpha < 1:
        msg = f"alpha has to be between 0 and 1, got {alpha}"
        raise ValueError(msg)
    return float(stats.chi2.ppf(alpha, df=1) / 2)


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
