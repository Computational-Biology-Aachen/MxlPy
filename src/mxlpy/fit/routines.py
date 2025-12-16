"""Fitting routines.

Single model, single data routines
----------------------------------
- `steady_state`
- `time_course`
- `protocol_time_course`

Multiple model, single data routines
------------------------------------
- `ensemble_steady_state`
- `ensemble_time_course`
- `ensemble_protocol_time_course`

A carousel is a special case of an ensemble, where the general
structure (e.g. stoichiometries) is the same, while the reactions kinetics
can vary
- `carousel_steady_state`
- `carousel_time_course`
- `carousel_protocol_time_course`

Multiple model, multiple data
-----------------------------
- `joint_steady_state`
- `joint_time_course`
- `joint_protocol_time_course`

Multiple model, multiple data, multiple methods
-----------------------------------------------
Here we also allow to run different methods (e.g. steady-state vs time courses)
for each combination of model:data.

- `joint_mixed`

Minimizers
----------
- LocalScipyMinimizer, including common methods such as Nelder-Mead or L-BFGS-B
- GlobalScipyMinimizer, including common methods such as basin hopping or dual annealing

Loss functions
--------------
- rmse

"""

from __future__ import annotations

import logging
import multiprocessing
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pebble

from mxlpy import parallel
from mxlpy.fit import losses
from mxlpy.fit.abstract import (
    EnsembleFit,
    Fit,
    FitResidual,
    FitSettings,
    JointFit,
    MixedSettings,
    _Settings,
)
from mxlpy.minimizers.abstract import Bounds, LossFn, OptimisationState
from mxlpy.minimizers.abstract import Residual as MinimizerResidual
from mxlpy.model import Model
from mxlpy.simulation import Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import Result, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mxlpy.carousel import Carousel
    from mxlpy.integrators import IntegratorType
    from mxlpy.minimizers.abstract import MinimizerProtocol
    from mxlpy.model import Model

_LOGGER = logging.getLogger(__name__)


__all__ = [
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
    "joint_mixed",
    "joint_protocol_time_course",
    "joint_steady_state",
    "joint_time_course",
    "protocol_time_course",
    "protocol_time_course_residual",
    "steady_state",
    "steady_state_residual",
    "time_course",
    "time_course_residual",
]


###############################################################################
# Residual functions
###############################################################################


def steady_state_residual(
    updates: dict[str, float],
    settings: _Settings,
) -> float:
    """Calculate residual error between model steady state and experimental data."""
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_to_steady_state()
        .get_result()
    )
    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.index)],
            )
        case _:
            return cast(float, np.inf)


def time_course_residual(
    updates: dict[str, float],
    settings: _Settings,
) -> float:
    """Calculate residual error between model time course and experimental data."""
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_time_course(cast(list, settings.data.index))
        .get_result()
    )

    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)


def protocol_time_course_residual(
    updates: dict[str, float],
    settings: _Settings,
) -> float:
    """Calculate residual error between model time course and experimental data."""
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    if (protocol := settings.protocol) is None:
        msg = "No protocol supplied"
        raise ValueError(msg)

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_protocol_time_course(
            protocol=protocol,
            time_points=settings.data.index,
        )
        .get_result()
    )

    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)


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

    Examples:
        >>> fit.steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental steady state data as pandas Series
        p0: Initial guesses as {name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
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

    Examples:
        >>> fit.time_course(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental time course data
        p0: Initial guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
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

    Examples:
        >>> fit.protocol_time_course(
        ...     model_fn(),
        ...     p0={"k2": 1.87, "k3": 1.093},
        ...     data=res_protocol,
        ...     protocol=protocol,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
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


###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


def ensemble_steady_state(
    ensemble: list[Model],
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
) -> EnsembleFit:
    """Fit model ensemble parameters to steady-state experimental data.

    Examples:
        >>> fit.ensemble_steady_state(
        ...     [
        ...         model_fn(),
        ...         model_fn(),
        ...     ],
        ...     data=res.iloc[-1],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Args:
        ensemble: Ensemble to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    steady_state,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def carousel_steady_state(
    carousel: Carousel,
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
) -> EnsembleFit:
    """Fit model parameters to steady-state experimental data over a carousel.

    Examples:
        >>> fit.carousel_steady_state(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Args:
        carousel: Model carousel to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return ensemble_steady_state(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )


def ensemble_time_course(
    ensemble: list[Model],
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
) -> EnsembleFit:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples:
        >>> fit.ensemble_steady_state(
        ...     [
        ...         model1,
        ...         model2,
        ...     ],
        ...     data=res.iloc[-1],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Args:
        ensemble: Model ensemble to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    time_course,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def carousel_time_course(
    carousel: Carousel,
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
) -> EnsembleFit:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples:
        >>> fit.carousel_time_course(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Args:
        carousel: Model carousel to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return ensemble_time_course(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )


def ensemble_protocol_time_course(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
) -> EnsembleFit:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples:
        >>> fit.ensemble_protocol_time_course(
        ...     [
        ...         model_fn(),
        ...         model_fn(),
        ...     ],
        ...     data=res_protocol,
        ...     protocol=protocol,
        ...     p0={"k2": 1.87, "k3": 1.093},  # note that k1 is given by the protocol
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Args:
        ensemble: Model ensemble: value}
        p0: initial parameter guess
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    protocol_time_course,
                    p0=p0,
                    data=data,
                    protocol=protocol,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def carousel_protocol_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
) -> EnsembleFit:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples:
        >>> fit.carousel_protocol_time_course(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     protocol=protocol,
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Args:
        carousel: Model carousel to fit
        p0: Initial parameter guesses as {parameter_name: value}
        data: Experimental time course data
        protocol: Experimental protocol
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        time_points_per_step: Number of time points per step in the protocol
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state

    Returns:
        Ensemble fit object

    """
    return ensemble_protocol_time_course(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        protocol=protocol,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )


###############################################################################
# Joint fitting
# This is multi-model, multi-data fitting, where the models share some parameters
###############################################################################


def _unpacked[T1, T2, Tout](inp: tuple[T1, T2], fn: Callable[[T1, T2], Tout]) -> Tout:
    return fn(*inp)


def _sum_of_residuals(
    updates: dict[str, float],
    residual_fn: FitResidual,
    fits: list[_Settings],
    pool: pebble.ProcessPool,
) -> float:
    future = pool.map(
        partial(_unpacked, fn=residual_fn),
        [(updates, i) for i in fits],
        timeout=None,
    )
    error = 0.0
    it = future.result()
    while True:
        try:
            error += next(it)
        except StopIteration:
            break
        except TimeoutError:
            return np.inf
    return error


def joint_steady_state(
    to_fit: list[FitSettings],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[JointFit]:
    """Multi-model, multi-data fitting.

    Examples:
        >>> fit.joint_steady_state(
        ...     [
        ...         fit.FitSettings(model=model_fn(), data=res.iloc[-1]),
        ...         fit.FitSettings(model=model_fn(), data=res.iloc[-1]),
        ...     ],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Args:
        to_fit: Models and data to fit
        p0: Initial guesses as {name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        max_workers: maximal amount of workers in parallel
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
        Result object containing JointFit object

    """
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=deepcopy(i.model) if as_deepcopy else i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=steady_state_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    match min_result.value:
        case OptimisationState(parameters, residual):
            return Result(JointFit(parameters, loss=residual))
        case _ as e:
            return Result(e)


def joint_time_course(
    to_fit: list[FitSettings],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[JointFit]:
    """Multi-model, multi-data fitting.

    Examples:
        >>> fit.joint_steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        to_fit: Models and data to fit
        p0: Initial guesses as {name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        max_workers: maximal amount of workers in parallel
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
        Result object containing JointFit object

    """
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=deepcopy(i.model) if as_deepcopy else i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )

    match min_result.value:
        case OptimisationState(parameters, residual):
            return Result(JointFit(parameters, loss=residual))
        case _ as e:
            return Result(e)


def joint_protocol_time_course(
    to_fit: list[FitSettings],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[JointFit]:
    """Multi-model, multi-data fitting.

    Examples:
        >>> fit.joint_steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        to_fit: Models and data to fit
        p0: Initial guesses as {name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        max_workers: maximal amount of workers in parallel
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
        Result object containing JointFit object

    """
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=deepcopy(i.model) if as_deepcopy else i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                protocol=i.protocol,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=protocol_time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )

    match min_result.value:
        case OptimisationState(parameters, residual):
            return Result(JointFit(parameters, loss=residual))
        case _ as e:
            return Result(e)


###############################################################################
# Joint fitting
# This is multi-model, multi-data, multi-simulation fitting
# The models share some parameters here, everything else can be changed though
###############################################################################


def _execute(inp: tuple[dict[str, float], FitResidual, _Settings]) -> float:
    updates, residual_fn, settings = inp
    return residual_fn(updates, settings)


def _mixed_sum_of_residuals(
    updates: dict[str, float],
    fits: list[_Settings],
    pool: pebble.ProcessPool,
) -> float:
    future = pool.map(_execute, [(updates, i.residual_fn, i) for i in fits])
    error = 0.0
    it = future.result()
    while True:
        try:
            error += next(it)
        except StopIteration:
            break
        except TimeoutError:
            return np.inf
    return error


def joint_mixed(
    to_fit: list[MixedSettings],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[JointFit]:
    """Multi-model, multi-data, multi-simulation fitting.

    Examples:
        >>> fit.joint_mixed(
        ...     [
        ...         fit.MixedSettings(
        ...             model=model_fn(),
        ...             data=res.iloc[-1],
        ...             residual_fn=fit.steady_state_residual,
        ...         ),
        ...         fit.MixedSettings(
        ...             model=model_fn(),
        ...             data=res,
        ...             residual_fn=fit.time_course_residual,
        ...         ),
        ...     ],
        ...     p0={"k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Args:
        to_fit: models, data and residual fn to fit
        p0: Initial guesses as {name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter
        as_deepcopy: Whether to copy the model to avoid overwriting the state
        max_workers: maximal amount of workers in parallel
        standard_scale: Whether to apply standard scale to data and prediction

    Returns:
        Result object containing JointFit object

    """
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=deepcopy(i.model) if as_deepcopy else i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                protocol=i.protocol,
                residual_fn=i.residual_fn,
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _mixed_sum_of_residuals,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )

    match min_result.value:
        case OptimisationState(parameters, residual):
            return Result(JointFit(parameters, loss=residual))
        case _ as e:
            return Result(e)
