"""Common types and utilities between local and global fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy.model import Model
from mxlpy.simulator import Simulator
from mxlpy.types import Array, ArrayLike, Callable, IntegratorType, cast

if TYPE_CHECKING:
    import pandas as pd

    from mxlpy.model import Model

LOGGER = logging.getLogger(__name__)

type InitialGuess = dict[str, float]

type Bounds = dict[str, tuple[float | None, float | None]]
type ResidualFn = Callable[[Array], float]
type LossFn = Callable[
    [
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
    ],
    float,
]


__all__ = [
    "Bounds",
    "CarouselFit",
    "FitResult",
    "InitialGuess",
    "LOGGER",
    "LossFn",
    "MinResult",
    "ProtocolResidualFn",
    "ResidualFn",
    "SteadyStateResidualFn",
    "TimeSeriesResidualFn",
    "rmse",
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
class CarouselFit:
    """Result of a carousel fit operation."""

    fits: list[FitResult]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_best_fit(self) -> FitResult:
        """Get the best fit from the carousel."""
        return min(self.fits, key=lambda x: x.loss)


def rmse(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate root mean square error between model and data."""
    return cast(float, np.sqrt(np.mean(np.square(y_pred - y_true))))


class SteadyStateResidualFn(Protocol):
    """Protocol for steady state residual functions."""

    def __call__(
        self,
        par_values: Array,
        # This will be filled out by partial
        par_names: list[str],
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
        par_values: Array,
        # This will be filled out by partial
        par_names: list[str],
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
        par_values: Array,
        # This will be filled out by partial
        par_names: list[str],
        data: pd.DataFrame,
        model: Model,
        y0: dict[str, float] | None,
        integrator: IntegratorType | None,
        loss_fn: LossFn,
        protocol: pd.DataFrame,
    ) -> float:
        """Calculate residual error between model time course and experimental data."""
        ...


def _steady_state_residual(
    par_values: Array,
    # This will be filled out by partial
    par_names: list[str],
    data: pd.Series,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
) -> float:
    """Calculate residual error between model steady state and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental steady state data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(
                dict(
                    zip(
                        par_names,
                        par_values,
                        strict=True,
                    )
                )
            ),
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
    par_values: ArrayLike,
    # This will be filled out by partial
    par_names: list[str],
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental time course data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
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
    par_values: ArrayLike,
    # This will be filled out by partial
    par_names: list[str],
    data: pd.DataFrame,
    model: Model,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
    loss_fn: LossFn,
    protocol: pd.DataFrame,
) -> float:
    """Calculate residual error between model time course and experimental data.

    Args:
        par_values: Parameter values to test
        data: Experimental time course data
        model: Model instance to simulate
        y0: Initial conditions
        par_names: Names of parameters being fit
        integrator: ODE integrator class to use
        loss_fn: Loss function to use for residual calculation
        protocol: Experimental protocol
        time_points_per_step: Number of time points per step in the protocol

    Returns:
        float: Root mean square error between model and data

    """
    res = (
        Simulator(
            model.update_parameters(dict(zip(par_names, par_values, strict=True))),
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
