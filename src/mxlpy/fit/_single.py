###############################################################################
# Single fitting
# This is single-model, single-data fitting
###############################################################################

from copy import deepcopy
from functools import partial

import pandas as pd

from mxlpy.fit import losses
from mxlpy.fit.abstract import Fit, FitResidual, _Settings
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.integrators import IntegratorType
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
)
from mxlpy.minimizers.abstract import Residual as MinimizerResidual
from mxlpy.model import Model
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

__all__ = ["protocol_time_course", "steady_state", "time_course"]


def steady_state(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[Fit]:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction

    Returns
    -------
        Result object containing fit object

    """
    if as_deepcopy:
        model = deepcopy(model)

    p_names = model.get_parameter_names()
    v_names = model.get_variable_names()

    fn: MinimizerResidual = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in p0 if i in p_names],
            v_names=[i for i in p0 if i in v_names],
            standard_scale=standard_scale,
        ),
    )
    match minimizer(fn, p0, {} if bounds is None else bounds).value:
        case OptimisationState(parameters, residual):
            return Result(
                Fit(
                    model=model,
                    best_pars=parameters,
                    loss=residual,
                )
            )
        case _ as e:
            return Result(e)


def time_course(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[Fit]:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction

    Returns
    -------
        Result object containing fit object

    """
    if as_deepcopy:
        model = deepcopy(model)
    p_names = model.get_parameter_names()
    v_names = model.get_variable_names()

    fn: MinimizerResidual = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in p0 if i in p_names],
            v_names=[i for i in p0 if i in v_names],
            standard_scale=standard_scale,
        ),
    )

    match minimizer(fn, p0, {} if bounds is None else bounds).value:
        case OptimisationState(parameters, residual):
            return Result(
                Fit(
                    model=model,
                    best_pars=parameters,
                    loss=residual,
                )
            )
        case _ as e:
            return Result(e)


def protocol_time_course(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[Fit]:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    standard_scale
        Whether to apply standard scale to data and prediction

    Returns
    -------
        Result object containing fit object

    """
    if as_deepcopy:
        model = deepcopy(model)
    p_names = model.get_parameter_names()
    v_names = model.get_variable_names()
    protocol = _normalise_protocol_index(protocol)

    fn: MinimizerResidual = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in p0 if i in p_names],
            v_names=[i for i in p0 if i in v_names],
            protocol=protocol,
            standard_scale=standard_scale,
        ),
    )

    match minimizer(fn, p0, {} if bounds is None else bounds).value:
        case OptimisationState(parameters, residual):
            return Result(
                Fit(
                    model=model,
                    best_pars=parameters,
                    loss=residual,
                )
            )
        case _ as e:
            return Result(e)
