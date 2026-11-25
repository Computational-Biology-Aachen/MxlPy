# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Single fitting
# This is single-model, single-data fitting
###############################################################################

import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import Any, overload

import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.distributions import Prior
from mxlpy.fit import losses
from mxlpy.fit._gradient import GradientResidual, Transform, transformed
from mxlpy.fit.abstract import (
    Fit,
    FitResidual,
    PostprocessFn,
    _Settings,
    _split_p0,
    with_weight_bounds,
    without_weights,
)
from mxlpy.fit.callbacks import (
    Callback,
    failed,
    failure_message,
    finished,
    tracked,
)
from mxlpy.fit.penalties import Penalty, collect_penalties, penalised
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
    Residual,
)
from mxlpy.runners import DynamicRunner, SteadyStateRunner, default_runner
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

__all__ = ["loss_gradient", "protocol_time_course", "steady_state", "time_course"]


type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder


def _check_y0(model: _Model, y0: dict[str, float] | None) -> None:
    if y0 is not None and isinstance(model, SteadyStateModelBuilder):
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)


def _minimize[M: _Model](
    settings: _Settings[M],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    residual_fn: FitResidual,
    bounds: Bounds | None,
    transform: Transform | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    t0 = time.perf_counter()
    residual: Residual = (
        GradientResidual(residual_fn, settings)
        if GradientResidual.supports(residual_fn)
        else partial(residual_fn, settings=settings)
    )
    # Penalties apply to the untransformed values, so the optimum doesn't
    # depend on the transform
    with_penalty = None
    if penalties := collect_penalties(prior, penalty, list(p0)):
        residual = with_penalty = penalised(residual, penalties)
    residual, p0, minimizer_bounds, back = transformed(
        residual,
        p0,
        with_weight_bounds([settings.model], p0, bounds),
        without_weights([settings.model], p0, transform),
    )
    residual = tracked(residual, callback, back=back)
    match minimizer(residual, p0, minimizer_bounds).value:
        case OptimisationState(parameters, value):
            best = back(parameters)
            if with_penalty is None:
                result = Fit(model=settings.model, best_pars=best, loss=value)
            else:
                # Evaluated separately, as `value - penalty` loses the loss's
                # precision if the penalty dominates
                result = Fit(
                    model=settings.model,
                    best_pars=best,
                    loss=with_penalty.inner(best),
                    penalty=with_penalty.penalty(best),
                )
            finished(
                callback,
                residual,
                start=None,
                value=result.objective,
                parameters=best,
                duration=time.perf_counter() - t0,
            )
            return Result(result)
        case _ as e:
            failed(
                callback,
                residual,
                start=None,
                message=failure_message(e),
                duration=time.perf_counter() - t0,
            )
            return Result(e)


@overload
def steady_state(
    model: KineticModelBuilder,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[KineticModelBuilder]]: ...


@overload
def steady_state(
    model: OdeModelBuilder,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[OdeModelBuilder]]: ...


@overload
def steady_state(
    model: SteadyStateModelBuilder,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[SteadyStateModelBuilder]]: ...


def steady_state[M: _Model](
    model: M,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    """Fit model parameters to steady-state experimental data.

    Examples
    --------
        >>> fit.steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Parameters
    ----------
    model
        Model instance to fit
    data
        Experimental steady state data as pandas Series
    p0
        Initial guesses as {name: value}
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.
    loss_fn
        Loss function to use for residual calculation
    postprocess_fn
        Function applied to the combined simulation result before the loss is
        calculated, e.g. to compute quantities derived from the whole time
        series. Its output has to contain the data columns.
    bounds
        Mapping of bounds per parameter
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` after every evaluation and at the end, see
        `mxlpy.fit.callbacks`

    Returns
    -------
        Result object containing fit object

    """
    return _steady_state(
        model,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        runner=runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
        transform=transform,
        callback=callback,
    )


def _steady_state[M: _Model](
    model: M,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    """Non-overloaded implementation of `steady_state`, for internal callers."""
    _check_y0(model, y0)
    if as_deepcopy:
        model = deepcopy(model)
    p_names, v_names = _split_p0(model, p0)

    return _minimize(
        _Settings(
            model=model,
            data=data,
            y0=y0,
            runner=default_runner(model) if runner is None else runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            p_names=p_names,
            v_names=v_names,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        transform=transform,
        callback=callback,
    )


def time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    """Fit model parameters to time course of experimental data.

    Examples
    --------
        >>> fit.time_course(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Parameters
    ----------
    model
        Model instance to fit
    data
        Experimental time course data
    p0
        Initial guesses as {parameter_name: value}
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.
    loss_fn
        Loss function to use for residual calculation
    postprocess_fn
        Function applied to the combined simulation result before the loss is
        calculated, e.g. to compute quantities derived from the whole time
        series. Its output has to contain the data columns.
    bounds
        Mapping of bounds per parameter
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` after every evaluation and at the end, see
        `mxlpy.fit.callbacks`

    Returns
    -------
        Result object containing fit object

    """
    if as_deepcopy:
        model = deepcopy(model)
    p_names, v_names = _split_p0(model, p0)

    return _minimize(
        _Settings(
            model=model,
            data=data,
            y0=y0,
            runner=default_runner(model) if runner is None else runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            p_names=p_names,
            v_names=v_names,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        transform=transform,
        callback=callback,
    )


def protocol_time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    """Fit model parameters to time course of experimental data.

    Time points of protocol time course are taken from the data.

    Examples
    --------
        >>> fit.protocol_time_course(
        ...     model_fn(),
        ...     p0={"k2": 1.87, "k3": 1.093},
        ...     data=res_protocol,
        ...     protocol=protocol,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )
        {'k1': 0.1, 'k2': 0.2}

    Parameters
    ----------
    model
        Model instance to fit
    p0
        Initial parameter guesses as {parameter_name: value}
    data
        Experimental time course data
    protocol
        Experimental protocol
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.
    loss_fn
        Loss function to use for residual calculation
    postprocess_fn
        Function applied to the combined simulation result before the loss is
        calculated, e.g. to compute quantities derived from the whole time
        series. Its output has to contain the data columns.
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` after every evaluation and at the end, see
        `mxlpy.fit.callbacks`

    Returns
    -------
        Result object containing fit object

    """
    if as_deepcopy:
        model = deepcopy(model)
    p_names, v_names = _split_p0(model, p0)
    protocol = _normalise_protocol_index(protocol)

    return _minimize(
        _Settings(
            model=model,
            data=data,
            y0=y0,
            runner=default_runner(model) if runner is None else runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            p_names=p_names,
            v_names=v_names,
            protocol=protocol,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        transform=transform,
        callback=callback,
    )


###############################################################################
# Dispatch on the kind of data, for routines fitting any of them
###############################################################################


def _fit_data(
    model: _Model,
    data: pd.Series | pd.DataFrame,
    protocol: pd.DataFrame | None,
    **kwargs: Any,
) -> Result[Fit[Any]]:
    """Fit a steady state (Series), time course (DataFrame) or protocol time course."""
    if protocol is not None:
        return protocol_time_course(model, data=data, protocol=protocol, **kwargs)  # type: ignore[arg-type]
    if isinstance(data, pd.Series):
        return _steady_state(model, data=data, **kwargs)
    return time_course(model, data=data, **kwargs)  # type: ignore[arg-type]


def _residual_for(
    data: pd.Series | pd.DataFrame, protocol: pd.DataFrame | None
) -> FitResidual:
    """Residual of a steady state (Series), time course (DataFrame) or protocol fit."""
    if protocol is not None:
        return protocol_time_course_residual
    if isinstance(data, pd.Series):
        return steady_state_residual
    return time_course_residual


def _settings_for(
    model: _Model,
    data: pd.Series | pd.DataFrame,
    protocol: pd.DataFrame | None,
    *,
    y0: dict[str, float] | None,
    runner: SteadyStateRunner[Any, Any] | None,
    loss_fn: LossFn,
    postprocess_fn: PostprocessFn | None,
    standard_scale: bool,
    names: list[str],
) -> tuple[FitResidual, _Settings[Any]]:
    """Residual function and settings for any kind of data, fitting `names`."""
    fn = _residual_for(data, protocol)
    p_names, v_names = _split_p0(model, dict.fromkeys(names, 0.0))
    if unknown := [k for k in names if k not in p_names + v_names]:
        msg = f"Unknown parameters / variables {unknown}"
        raise ValueError(msg)
    settings = _Settings(
        model=model,
        data=data,
        y0=y0,
        runner=default_runner(model) if runner is None else runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        p_names=p_names,
        v_names=v_names,
        standard_scale=standard_scale,
        protocol=_normalise_protocol_index(protocol) if protocol is not None else None,
    )
    return fn, settings


def loss_gradient(
    model: _Model,
    data: pd.Series | pd.DataFrame,
    parameters: dict[str, float],
    *,
    protocol: pd.DataFrame | None = None,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    standard_scale: bool = True,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
) -> pd.Series:
    """Gradient of a fit's objective w.r.t. parameters and initial values.

    The gradient gradient-based minimizers see, in the untransformed values.
    The runner computes it (see `mxlpy.runners`): with
    `derivatives="adjoint"` (or `"auto"` for large time courses) by one
    forward and one backward solve, else from the sensitivities.

    Examples
    --------
        >>> fit.loss_gradient(model, data, {"k1": 1.0, "k2": 0.5})
        k1    0.12
        k2   -0.03
        dtype: float64

    Parameters
    ----------
    model
        Model to evaluate, not changed
    data
        Steady-state (Series) or time-course (DataFrame) data
    parameters
        Values of the parameters / variable initial values to differentiate
        w.r.t.
    protocol
        Protocol of a protocol time course
    y0
        Initial conditions as {species_name: value}
    runner
        How the model is simulated and differentiated, see `mxlpy.runners`
    loss_fn
        Loss function, as for the fits
    postprocess_fn
        Function applied to the simulation before the loss, as for the fits
    standard_scale
        Whether to apply standard scale to data and prediction
    prior
        Priors whose negative log densities are added, as for the fits
    penalty
        Further penalties added, as for the fits

    Returns
    -------
    pd.Series
        Gradient per name of `parameters`, NaN if the simulation failed

    """
    _check_y0(model, y0)
    residual_fn, settings = _settings_for(
        deepcopy(model),
        data,
        protocol,
        y0=y0,
        runner=runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        standard_scale=standard_scale,
        names=list(parameters),
    )
    residual: Any = GradientResidual(residual_fn, settings)
    if penalties := collect_penalties(prior, penalty, list(parameters)):
        residual = penalised(residual, penalties)
    return pd.Series(residual.gradient(dict(parameters)), index=list(parameters))
