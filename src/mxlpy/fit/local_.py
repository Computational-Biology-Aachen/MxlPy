"""Parameter local fitting Module for Metabolic Models.

This module provides functions for fitting model parameters to experimental data,
including both steadyd-state and time-series data fitting capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from scipy.optimize import minimize

from mxlpy import parallel
from mxlpy.types import IntegratorType, cast

from .common import (
    Bounds,
    CarouselFit,
    FitResult,
    InitialGuess,
    LossFn,
    MinResult,
    ProtocolResidualFn,
    ResidualFn,
    SteadyStateResidualFn,
    TimeSeriesResidualFn,
    _protocol_time_course_residual,
    _steady_state_residual,
    _time_course_residual,
    rmse,
)

if TYPE_CHECKING:
    import pandas as pd

    from mxlpy.carousel import Carousel
    from mxlpy.model import Model

LOGGER = logging.getLogger(__name__)

__all__ = [
    "LOGGER",
    "Minimizer",
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
    "protocol_time_course",
    "steady_state",
    "time_course",
]


type Minimizer = Callable[
    [
        ResidualFn,
        InitialGuess,
        Bounds,
    ],
    MinResult | None,
]


@dataclass
class ScipyMinimizer:
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
        res = minimize(
            residual_fn,
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


_default_minimizer = ScipyMinimizer()


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
        y0=y0,
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


def steady_state(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.Series,
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
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
    par_names = list(p0.keys())

    # Copy to restore
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            par_names=par_names,
            integrator=integrator,
            loss_fn=loss_fn,
        ),
    )
    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return min_result

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
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
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
    par_names = list(p0.keys())
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            par_names=par_names,
            integrator=integrator,
            loss_fn=loss_fn,
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return min_result

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
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
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
    par_names = list(p0.keys())
    p_orig = model.get_parameter_values()

    fn = cast(
        ResidualFn,
        partial(
            residual_fn,
            data=data,
            model=model,
            y0=y0,
            par_names=par_names,
            integrator=integrator,
            loss_fn=loss_fn,
            protocol=protocol,
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    # Restore original model
    model.update_parameters(p_orig)
    if min_result is None:
        return min_result

    return FitResult(
        model=deepcopy(model).update_parameters(min_result.parameters),
        best_pars=min_result.parameters,
        loss=min_result.residual,
    )


def carousel_steady_state(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.Series,
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
    residual_fn: SteadyStateResidualFn = _steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> CarouselFit:
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
    return CarouselFit(
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
                inputs=list(enumerate(carousel.variants)),
            )
            if (fit := i[1]) is not None
        ]
    )


def carousel_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
    residual_fn: TimeSeriesResidualFn = _time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> CarouselFit:
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
    return CarouselFit(
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
                inputs=list(enumerate(carousel.variants)),
            )
            if (fit := i[1]) is not None
        ]
    )


def carousel_protocol_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    minimizer: Minimizer = _default_minimizer,
    residual_fn: ProtocolResidualFn = _protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
) -> CarouselFit:
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
    return CarouselFit(
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
                inputs=list(enumerate(carousel.variants)),
            )
            if (fit := i[1]) is not None
        ]
    )
