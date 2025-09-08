"""Fitting routines."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd
import pebble
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    direct,
    dual_annealing,
    minimize,
    shgo,
)
from wadler_lindig import pformat

from mxlpy import parallel
from mxlpy.model import Model
from mxlpy.simulator import Simulator
from mxlpy.types import Array, IntegratorType, cast

if TYPE_CHECKING:
    import pandas as pd
    from scipy.optimize._optimize import OptimizeResult

    from mxlpy.carousel import Carousel
    from mxlpy.model import Model

LOGGER = logging.getLogger(__name__)

type InitialGuess = dict[str, float]

type Bounds = dict[str, tuple[float | None, float | None]]
type ResFn = Callable[[Array], float]
type ResidualFn = Callable[[dict[str, float]], float]
type LossFn = Callable[
    [
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
    ],
    float,
]


type Minimizer = Callable[
    [
        ResidualFn,
        InitialGuess,
        Bounds,
    ],
    MinResult | None,
]


__all__ = [
    "Bounds",
    "EnsembleFitResult",
    "FitInputs",
    "FitResult",
    "FitSettings",
    "FullFitSettings",
    "GlobalScipyMinimizer",
    "InitialGuess",
    "JointFitResult",
    "LOGGER",
    "LocalScipyMinimizer",
    "LossFn",
    "MinResult",
    "Minimizer",
    "ProtocolResidualFn",
    "ResFn",
    "ResidualFn",
    "SteadyStateResidualFn",
    "TimeSeriesResidualFn",
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
    "joint_protocol_time_course",
    "joint_steady_state",
    "joint_time_course",
    "protocol_time_course",
    "rmse",
    "steady_state",
    "time_course",
]


@dataclass
class MinResult:
    """Result of a minimization operation."""

    parameters: dict[str, float]
    residual: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class FitResult:
    """Result of a fit operation."""

    model: Model
    best_pars: dict[str, float]
    loss: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class EnsembleFitResult:
    """Result of a carousel fit operation."""

    fits: list[FitResult]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_best_fit(self) -> FitResult:
        """Get the best fit from the carousel."""
        return min(self.fits, key=lambda x: x.loss)


@dataclass
class JointFitResult:
    """Result of joint fit operation."""

    best_pars: dict[str, float]
    loss: float


###############################################################################
# loss fns
###############################################################################


def rmse(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate root mean square error between model and data."""
    return cast(float, np.sqrt(np.mean(np.square(y_pred - y_true))))


###############################################################################
# Residual function types
###############################################################################


class SteadyStateResidualFn(Protocol):
    """Protocol for steady state residual functions."""

    def __call__(
        self,
        updates: dict[str, float],
        # This will be filled out by partial
        data: pd.Series,
        model: Model,
        y0: dict[str, float] | None,
        integrator: IntegratorType | None,
        loss_fn: LossFn,
    ) -> float:
        """Calculate residual error between model steady state and experimental data."""
        ...


class TimeSeriesResidualFn(Protocol):
    """Protocol for time series residual functions."""

    def __call__(
        self,
        updates: dict[str, float],
        # This will be filled out by partial
        data: pd.DataFrame,
        model: Model,
        y0: dict[str, float] | None,
        integrator: IntegratorType | None,
        loss_fn: LossFn,
    ) -> float:
        """Calculate residual error between model time course and experimental data."""
        ...


class ProtocolResidualFn(Protocol):
    """Protocol for time series residual functions."""

    def __call__(
        self,
        updates: dict[str, float],
        # This will be filled out by partial
        data: pd.DataFrame,
        model: Model,
        y0: dict[str, float] | None,
        integrator: IntegratorType | None,
        loss_fn: LossFn,
        protocol: pd.DataFrame,
    ) -> float:
        """Calculate residual error between model time course and experimental data."""
        ...


def _pack_updates(
    par_values: Array,
    par_names: list[str],
) -> dict[str, float]:
    return dict(
        zip(
            par_names,
            par_values,
            strict=True,
        )
    )


def _steady_state_residual(
    updates: dict[str, float],
    # This will be filled out by partial
    data: pd.Series,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
) -> float:
    """Calculate residual error between model steady state and experimental data.

    Args:
        updates: Parameter values to test
        data: Experimental steady state data
        model: Model instance to simulate
        y0: Initial conditions
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(updates),
            y0=y0,
            integrator=integrator,
        )
        .simulate_to_steady_state()
        .get_result()
    )
    if res is None:
        return cast(float, np.inf)

    return loss_fn(
        res.get_combined().loc[:, cast(list, data.index)],
        data,
    )


def _time_course_residual(
    updates: dict[str, float],
    # This will be filled out by partial
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        updates: Parameter values to test
        data: Experimental time course data
        model: Model instance to simulate
        y0: Initial conditions
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(updates),
            y0=y0,
            integrator=integrator,
        )
        .simulate_time_course(cast(list, data.index))
        .get_result()
    )
    if res is None:
        return cast(float, np.inf)
    results_ss = res.get_combined()

    return loss_fn(
        results_ss.loc[:, cast(list, data.columns)],
        data,
    )


def _protocol_time_course_residual(
    updates: dict[str, float],
    # This will be filled out by partial
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
    protocol: pd.DataFrame,
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        updates: Parameter values to test
        data: Experimental time course data
        model: Model instance to simulate
        y0: Initial conditions
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation
        protocol: Experimental protocol
        time_points_per_step: Number of time points per step in the protocol

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(updates),
            y0=y0,
            integrator=integrator,
        )
        .simulate_protocol_time_course(
            protocol=protocol,
            time_points=data.index,
        )
        .get_result()
    )
    if res is None:
        return cast(float, np.inf)
    results_ss = res.get_combined()

    return loss_fn(
        results_ss.loc[:, cast(list, data.columns)],
        data,
    )


def steady_state(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: SteadyStateResidualFn = _steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> FitResult | None:
    """Fit model parameters to steady-state experimental data.

    Examples:
        >>> steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental steady state data as pandas Series
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    # Copy to restore
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
        ),
    )
    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return None

    return FitResult(
        model=deepcopy(model).update_parameters(min_result.parameters),
        best_pars=min_result.parameters,
        loss=min_result.residual,
    )


def time_course(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: TimeSeriesResidualFn = _time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> FitResult | None:
    """Fit model parameters to time course of experimental data.

    Examples:
        >>> time_course(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Args:
        model: Model instance to fit
        data: Experimental time course data
        p0: Initial parameter guesses as {parameter_name: value}
        y0: Initial conditions as {species_name: value}
        minimizer: Function to minimize fitting error
        residual_fn: Function to calculate fitting error
        integrator: ODE integrator class
        loss_fn: Loss function to use for residual calculation
        bounds: Mapping of bounds per parameter

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return None

    return FitResult(
        model=deepcopy(model).update_parameters(min_result.parameters),
        best_pars=min_result.parameters,
        loss=min_result.residual,
    )


def protocol_time_course(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: ProtocolResidualFn = _protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> FitResult | None:
    """Fit model parameters to time course of experimental data.

    Time points of protocol time course are taken from the data.

    Examples:
        >>> time_course(model, p0, data)
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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            protocol=protocol,
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return None

    return FitResult(
        model=deepcopy(model).update_parameters(min_result.parameters),
        best_pars=min_result.parameters,
        loss=min_result.residual,
    )


###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


def _carousel_steady_state_worker(
    model: Model,
    p0: dict[str, float],
    data: pd.Series,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
    minimizer: Minimizer,
    residual_fn: SteadyStateResidualFn,
    bounds: Bounds | None,
) -> FitResult | None:
    model_pars = model.get_parameter_values()

    return steady_state(
        model,
        p0={k: v for k, v in p0.items() if k in model_pars},
        y0=y0,  # FIXME: also check if vars in y0 for multi-model, multi-data fitting
        data=data,
        minimizer=minimizer,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
    )


def _carousel_time_course_worker(
    model: Model,
    p0: dict[str, float],
    data: pd.DataFrame,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
    minimizer: Minimizer,
    residual_fn: TimeSeriesResidualFn,
    bounds: Bounds | None,
) -> FitResult | None:
    model_pars = model.get_parameter_values()
    return time_course(
        model,
        p0={k: v for k, v in p0.items() if k in model_pars},
        y0=y0,
        data=data,
        minimizer=minimizer,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
    )


def _carousel_protocol_worker(
    model: Model,
    p0: dict[str, float],
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
    minimizer: Minimizer,
    residual_fn: ProtocolResidualFn,
    bounds: Bounds | None,
) -> FitResult | None:
    model_pars = model.get_parameter_values()
    return protocol_time_course(
        model,
        p0={k: v for k, v in p0.items() if k in model_pars},
        y0=y0,
        protocol=protocol,
        data=data,
        minimizer=minimizer,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
    )


def ensemble_steady_state(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: SteadyStateResidualFn = _steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model ensemble parameters to steady-state experimental data.

    Examples:
        >>> carousel_steady_state(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    return EnsembleFitResult(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _carousel_steady_state_worker,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if (fit := i[1]) is not None
        ]
    )


def carousel_steady_state(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: SteadyStateResidualFn = _steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model parameters to steady-state experimental data over a carousel.

    Examples:
        >>> carousel_steady_state(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

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
    )


def ensemble_time_course(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: TimeSeriesResidualFn = _time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples:
        >>> carousel_time_course(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    return EnsembleFitResult(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _carousel_time_course_worker,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if (fit := i[1]) is not None
        ]
    )


def carousel_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: TimeSeriesResidualFn = _time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples:
        >>> carousel_time_course(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

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
    )


def ensemble_protocol_time_course(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: Minimizer,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: ProtocolResidualFn = _protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples:
        >>> carousel_steady_state(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

    """
    return EnsembleFitResult(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _carousel_protocol_worker,
                    p0=p0,
                    data=data,
                    protocol=protocol,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                ),
                inputs=list(enumerate(ensemble)),
            )
            if (fit := i[1]) is not None
        ]
    )


def carousel_protocol_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: Minimizer,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: ProtocolResidualFn = _protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> EnsembleFitResult:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples:
        >>> carousel_steady_state(carousel, p0=p0, data=data)

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

    Returns:
        dict[str, float]: Fitted parameters as {parameter_name: fitted_value}

    Note:
        Uses L-BFGS-B optimization with bounds [1e-6, 1e6] for all parameters

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
    )


###############################################################################
# Joint fitting
# This is multi-model, multi-data fitting, where the models share some parameters
###############################################################################


@dataclass
class FitSettings:
    model: Model
    data: pd.Series | pd.DataFrame
    p0: dict[str, float] | None = None
    y0: dict[str, float] | None = None
    integrator: IntegratorType | None = None
    loss_fn: LossFn | None = None


@dataclass
class FullFitSettings:
    model: Model
    data: pd.Series | pd.DataFrame
    p0: dict[str, float]
    y0: dict[str, float] | None
    integrator: IntegratorType | None
    loss_fn: LossFn


@dataclass
class FitInputs:
    p: dict[str, float]
    y: dict[str, float]

    @classmethod
    def from_model(cls, updates: dict[str, float], model: Model) -> FitInputs:
        model_pars = model.get_parameter_values()
        model_vars = set(model.get_variable_names())

        return FitInputs(
            p={k: v for k, v in updates.items() if k in model_pars},
            y={k: v for k, v in updates.items() if k in model_vars},
        )


def _single_ss_residual(inp: tuple[FullFitSettings, FitInputs]) -> float:
    settings, inputs = inp
    return _steady_state_residual(
        updates=inputs.p,
        data=settings.data,  # type: ignore
        model=settings.model,
        y0=settings.y0,
        integrator=settings.integrator,
        loss_fn=settings.loss_fn,
    )


def _single_tc_residual(inp: tuple[FullFitSettings, FitInputs]) -> float:
    settings, inputs = inp
    return _time_course_residual(
        updates=inputs.p,
        data=settings.data,  # type: ignore
        model=settings.model,
        y0=settings.y0,
        integrator=settings.integrator,
        loss_fn=settings.loss_fn,
    )


def _single_ptc_residual(
    inp: tuple[FullFitSettings, FitInputs],
    protocol: pd.DataFrame,
) -> float:
    settings, inputs = inp
    return _protocol_time_course_residual(
        updates=inputs.p,
        data=settings.data,  # type: ignore
        model=settings.model,
        y0=settings.y0,
        integrator=settings.integrator,
        loss_fn=settings.loss_fn,
        protocol=protocol,
    )


def _sum_of_residuals(
    updates: dict[str, float],
    residual_fn: Callable[[tuple[FullFitSettings, FitInputs]], float],
    fits: list[FullFitSettings],
    pool: pebble.ProcessPool,
) -> float:
    future = pool.map(
        residual_fn,
        [(i, FitInputs.from_model(updates, i.model)) for i in fits],
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
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
) -> JointFitResult | None:
    original_pars = [i.model.get_parameter_values() for i in to_fit]
    full_settings = [
        FullFitSettings(
            model=i.model,
            data=i.data,
            p0=i.p0 if i.p0 is not None else p0,
            y0=i.y0 if i.y0 is not None else y0,
            integrator=i.integrator if i.integrator is not None else integrator,
            loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
        )
        for i in to_fit
    ]
    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=_single_ss_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

    # Restore original model
    for i, p_orig in zip(to_fit, original_pars, strict=True):
        i.model.update_parameters(p_orig)

    return JointFitResult(min_result.parameters, loss=min_result.residual)


def joint_time_course(
    to_fit: list[FitSettings],
    *,
    p0: dict[str, float],
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
) -> JointFitResult | None:
    original_pars = [i.model.get_parameter_values() for i in to_fit]
    full_settings = [
        FullFitSettings(
            model=i.model,
            data=i.data,
            p0=i.p0 if i.p0 is not None else p0,
            y0=i.y0 if i.y0 is not None else y0,
            integrator=i.integrator if i.integrator is not None else integrator,
            loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
        )
        for i in to_fit
    ]
    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=_single_tc_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

    # Restore original model
    for i, p_orig in zip(to_fit, original_pars, strict=True):
        i.model.update_parameters(p_orig)

    return JointFitResult(min_result.parameters, loss=min_result.residual)


def joint_protocol_time_course(
    to_fit: list[FitSettings],
    *,
    p0: dict[str, float],
    protocol: pd.DataFrame,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
) -> JointFitResult | None:
    original_pars = [i.model.get_parameter_values() for i in to_fit]
    full_settings = [
        FullFitSettings(
            model=i.model,
            data=i.data,
            p0=i.p0 if i.p0 is not None else p0,
            y0=i.y0 if i.y0 is not None else y0,
            integrator=i.integrator if i.integrator is not None else integrator,
            loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
        )
        for i in to_fit
    ]
    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=partial(_single_ptc_residual, protocol=protocol),
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

    # Restore original model
    for i, p_orig in zip(to_fit, original_pars, strict=True):
        i.model.update_parameters(p_orig)

    return JointFitResult(min_result.parameters, loss=min_result.residual)


###############################################################################
# Minimizers
###############################################################################


@dataclass
class LocalScipyMinimizer:
    """Local multivariate minimization using scipy.optimize.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization

    """

    tol: float = 1e-6
    method: Literal[
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "COBYQA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ] = "L-BFGS-B"

    def __call__(
        self,
        residual_fn: ResidualFn,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> MinResult | None:
        """Call minimzer."""
        par_names = list(p0.keys())

        res = minimize(
            lambda par_values: residual_fn(_pack_updates(par_values, par_names)),
            x0=list(p0.values()),
            bounds=[bounds.get(name, (1e-6, 1e6)) for name in p0],
            method=self.method,
            tol=self.tol,
        )
        if res.success:
            return MinResult(
                parameters=dict(
                    zip(
                        p0,
                        res.x,
                        strict=True,
                    ),
                ),
                residual=res.fun,
            )

        LOGGER.warning("Minimisation failed due to %s", res.message)
        return None


@dataclass
class GlobalScipyMinimizer:
    """Global iate minimization using scipy.optimize.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization

    """

    tol: float = 1e-6
    method: Literal[
        "basinhopping",
        "differential_evolution",
        "shgo",
        "dual_annealing",
        "direct",
    ] = "basinhopping"

    def __call__(
        self,
        residual_fn: ResidualFn,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> MinResult | None:
        """Minimize residual fn."""
        res: OptimizeResult
        par_names = list(p0.keys())
        res_fn: ResFn = lambda par_values: residual_fn(  # noqa: E731
            _pack_updates(par_values, par_names)
        )

        if self.method == "basinhopping":
            res = basinhopping(
                res_fn,
                x0=list(p0.values()),
            )
        elif self.method == "differential_evolution":
            res = differential_evolution(res_fn, bounds)
        elif self.method == "shgo":
            res = shgo(res_fn, bounds)
        elif self.method == "dual_annealing":
            res = dual_annealing(res_fn, bounds)
        elif self.method == "direct":
            res = direct(res_fn, bounds)
        else:
            msg = f"Unknown method {self.method}"
            raise NotImplementedError(msg)
        if res.success:
            return MinResult(
                parameters=dict(
                    zip(
                        p0,
                        res.x,
                        strict=True,
                    ),
                ),
                residual=res.fun,
            )

        LOGGER.warning("Minimisation failed.")
        return None
