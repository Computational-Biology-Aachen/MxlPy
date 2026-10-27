# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Single fitting
# This is single-model, single-data fitting
###############################################################################

from copy import deepcopy
from functools import partial
from typing import Any, overload

import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit import losses
from mxlpy.fit.abstract import Fit, FitResidual, _Settings
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
)
from mxlpy.runners import DynamicRunner, SteadyStateRunner, default_runner
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

__all__ = ["protocol_time_course", "steady_state", "time_course"]


type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder


def _split_p0(
    model: _Model,
    p0: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Split initial guesses into parameter and variable names."""
    p_names = set(model.get_parameter_names())
    v_names = (
        set()
        if isinstance(model, SteadyStateModelBuilder)
        else set(model.get_variable_names())
    )
    return [i for i in p0 if i in p_names], [i for i in p0 if i in v_names]


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
) -> Result[Fit[M]]:
    match minimizer(
        partial(residual_fn, settings=settings), p0, {} if bounds is None else bounds
    ).value:
        case OptimisationState(parameters, residual):
            return Result(
                Fit(
                    model=settings.model,
                    best_pars=parameters,
                    loss=residual,
                )
            )
        case _ as e:
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.

    Returns
    -------
        Result object containing fit object

    """
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
            p_names=p_names,
            v_names=v_names,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.

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
            p_names=p_names,
            v_names=v_names,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
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
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
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
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.

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
            p_names=p_names,
            v_names=v_names,
            protocol=protocol,
            standard_scale=standard_scale,
        ),
        p0=p0,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
    )
