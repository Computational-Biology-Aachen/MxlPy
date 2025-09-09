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


__all__ = [
    "Bounds",
    "EnsembleFitResult",
    "FitResult",
    "FitSettings",
    "GlobalScipyMinimizer",
    "InitialGuess",
    "JointFitResult",
    "LOGGER",
    "LocalScipyMinimizer",
    "LossFn",
    "MinResult",
    "Minimizer",
    "MixedSettings",
    "ResFn",
    "ResidualFn",
    "ResidualProtocol",
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
    "rmse",
    "steady_state",
    "time_course",
]

type InitialGuess = dict[str, float]

type Bounds = dict[str, tuple[float | None, float | None]]
type ResFn = Callable[[Array], float]
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


class ResidualProtocol(Protocol):
    """Protocol for steady state residual functions.

    This is the user-facing variant, for stuff like
    - `fit.steady_state`
    - `fit.time_course`
    - `fit.protocol_time_course`

    The settings are later partially applied to yield ResidualFn
    """

    def __call__(
        self,
        updates: dict[str, float],
        settings: _Settings,
    ) -> float:
        """Calculate residual error between model steady state and experimental data."""
        ...


class ResidualFn(Protocol):
    """Protocol for steady state residual functions.

    This is the internal version, which is produced by partial
    application of `settings` of `ResidualProtocol`
    """

    def __call__(
        self,
        updates: dict[str, float],
    ) -> float:
        """Calculate residual error between model steady state and experimental data."""
        ...


@dataclass
class FitSettings:
    """Settings for a fit."""

    model: Model
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None = None
    integrator: IntegratorType | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class MixedSettings:
    """Settings for a fit."""

    model: Model
    data: pd.Series | pd.DataFrame
    residual_fn: ResidualFn
    y0: dict[str, float] | None = None
    integrator: IntegratorType | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class _Settings:
    """Non user-facing version of FitSettings."""

    model: Model
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None
    integrator: IntegratorType | None
    loss_fn: LossFn
    p_names: list[str]
    v_names: list[str]
    protocol: pd.DataFrame | None = None
    residual_fn: ResidualFn | None = None


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

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


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


###############################################################################
# Residual functions
###############################################################################


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
    if res is None:
        return cast(float, np.inf)

    return settings.loss_fn(
        res.get_combined().loc[:, cast(list, settings.data.index)],
        settings.data,
    )


def _time_course_residual(
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
    if res is None:
        return cast(float, np.inf)
    results_ss = res.get_combined()

    return settings.loss_fn(
        results_ss.loc[:, cast(list, settings.data.columns)],
        settings.data,
    )


def _protocol_time_course_residual(
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
        raise ValueError

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
    if res is None:
        return cast(float, np.inf)
    results_ss = res.get_combined()

    return settings.loss_fn(
        results_ss.loc[:, cast(list, settings.data.columns)],
        settings.data,
    )


def steady_state(
    model: Model,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: ResidualProtocol = _steady_state_residual,
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
        p0: Initial guesses as {name: value}
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
    model = deepcopy(model)

    fn: ResidualFn = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in model.get_parameter_names() if i in p0],
            v_names=[i for i in model.get_variable_names() if i in p0],
        ),
    )
    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    if min_result is None:
        return None

    return FitResult(
        model=model.update_parameters(min_result.parameters),
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
    residual_fn: ResidualProtocol = _time_course_residual,
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
        p0: Initial guesses as {parameter_name: value}
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
    model = deepcopy(model)

    fn: ResidualFn = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in model.get_parameter_names() if i in p0],
            v_names=[i for i in model.get_variable_names() if i in p0],
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    if min_result is None:
        return None

    return FitResult(
        model=model.update_parameters(min_result.parameters),
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
    residual_fn: ResidualProtocol = _protocol_time_course_residual,
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
    model = deepcopy(model)

    fn: ResidualFn = partial(
        residual_fn,
        settings=_Settings(
            model=model,
            data=data,
            y0=y0,
            integrator=integrator,
            loss_fn=loss_fn,
            p_names=[i for i in model.get_parameter_names() if i in p0],
            v_names=[i for i in model.get_variable_names() if i in p0],
            protocol=protocol,
        ),
    )

    min_result = minimizer(fn, p0, {} if bounds is None else bounds)
    if min_result is None:
        return None

    return FitResult(
        model=model.update_parameters(min_result.parameters),
        best_pars=min_result.parameters,
        loss=min_result.residual,
    )


###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


def ensemble_steady_state(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    residual_fn: ResidualProtocol = _steady_state_residual,
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
                    steady_state,
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
    residual_fn: ResidualProtocol = _steady_state_residual,
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
    residual_fn: ResidualProtocol = _time_course_residual,
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
                    time_course,
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
    residual_fn: ResidualProtocol = _time_course_residual,
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
    residual_fn: ResidualProtocol = _protocol_time_course_residual,
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
    residual_fn: ResidualProtocol = _protocol_time_course_residual,
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


def _unpacked[T1, T2, Tout](inp: tuple[T1, T2], fn: Callable[[T1, T2], Tout]) -> Tout:
    return fn(*inp)


def _sum_of_residuals(
    updates: dict[str, float],
    residual_fn: ResidualProtocol,
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
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
) -> JointFitResult | None:
    """Multi-model, multi-data fitting."""
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
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
                residual_fn=_steady_state_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

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
    """Multi-model, multi-data fitting."""
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
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
                residual_fn=_time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

    return JointFitResult(min_result.parameters, loss=min_result.residual)


def joint_protocol_time_course(
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
    """Multi-model, multi-data fitting."""
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
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
                residual_fn=_protocol_time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )
    if min_result is None:
        return None

    return JointFitResult(min_result.parameters, loss=min_result.residual)


###############################################################################
# Joint fitting
# This is multi-model, multi-data, multi-simulation fitting
# The models share some parameters here, everything else can be changed though
###############################################################################


def _execute(inp: tuple[dict[str, float], ResidualProtocol, _Settings]) -> float:
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
    minimizer: Minimizer,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = rmse,
    bounds: Bounds | None = None,
    max_workers: int | None = None,
) -> JointFitResult | None:
    """Multi-model, multi-data, multi-simulation fitting."""
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                residual_fn=i.residual_fn,
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
    if min_result is None:
        return None

    return JointFitResult(min_result.parameters, loss=min_result.residual)
