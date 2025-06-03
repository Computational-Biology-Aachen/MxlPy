"""Parameter Scanning Module.

This module provides functions and classes for performing parameter scans on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as protocol-based simulations.

Classes:
    TimePoint: Represents a single time point in a simulation.
    TimeCourse: Represents a time course in a simulation.

Functions:
    parameter_scan_ss: Get steady-state results over supplied parameters.
    parameter_scan_time_course: Get time course for each supplied parameter.
    parameter_scan_protocol: Get protocol course for each supplied parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import pandas as pd

from mxlpy.parallel import Cache, parallelise
from mxlpy.simulator import Result, Simulator
from mxlpy.types import IntegratorType, ProtocolByPars, SteadyStates, TimeCourseByPars

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model
    from mxlpy.types import Array


__all__ = [
    "ProtocolWorker",
    "SteadyStateWorker",
    "TimeCourse",
    "TimeCourseWorker",
    "TimePoint",
    "steady_state",
    "time_course",
    "time_course_over_protocol",
]


def _update_parameters_and_initial_conditions[T](
    pars: pd.Series,
    fn: Callable[[Model], T],
    model: Model,
) -> T:
    """Update model parameters and execute a function.

    Args:
        pars: Series containing parameter values to update.
        fn: Function to execute after updating parameters.
        model: Model instance to update.

    Returns:
        Result of the function execution.

    """
    pd = pars.to_dict()
    model.update_variables({k: v for k, v in pd.items() if k in model._variables})  # noqa: SLF001
    model.update_parameters({k: v for k, v in pd.items() if k in model._parameters})  # noqa: SLF001
    return fn(model)


def _empty_conc_series(model: Model) -> pd.Series:
    """Create an empty concentration series for the model.

    Args:
        model: Model instance to generate the series for.

    Returns:
        pd.Series: Series with NaN values for each model variable.

    """
    return pd.Series(
        data=np.full(shape=len(model.get_variable_names()), fill_value=np.nan),
        index=model.get_variable_names(),
    )


def _empty_flux_series(model: Model) -> pd.Series:
    """Create an empty flux series for the model.

    Args:
        model: Model instance to generate the series for.

    Returns:
        pd.Series: Series with NaN values for each model reaction.

    """
    return pd.Series(
        data=np.full(shape=len(model.get_reaction_names()), fill_value=np.nan),
        index=model.get_reaction_names(),
    )


def _empty_conc_df(model: Model, time_points: Array) -> pd.DataFrame:
    """Create an empty concentration DataFrame for the model over given time points.

    Args:
        model: Model instance to generate the DataFrame for.
        time_points: Array of time points.

    Returns:
        pd.DataFrame: DataFrame with NaN values for each model variable at each time point.

    """
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_variable_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_variable_names(),
    )


def _empty_flux_df(model: Model, time_points: Array) -> pd.DataFrame:
    """Create an empty concentration DataFrame for the model over given time points.

    Args:
        model: Model instance to generate the DataFrame for.
        time_points: Array of time points.

    Returns:
        pd.DataFrame: DataFrame with NaN values for each model variable at each time point.

    """
    return pd.DataFrame(
        data=np.full(
            shape=(len(time_points), len(model.get_reaction_names())),
            fill_value=np.nan,
        ),
        index=time_points,
        columns=model.get_reaction_names(),
    )


###############################################################################
# Single returns
###############################################################################


@dataclass(slots=True)
class TimePoint:
    """Represents a single time point in a simulation.

    Attributes:
        concs: Series of concentrations at the time point.
        fluxes: Series of fluxes at the time point.

    Args:
        model: Model instance to generate the time point for.
        concs: DataFrame of concentrations (default: None).
        fluxes: DataFrame of fluxes (default: None).
        idx: Index of the time point in the DataFrame (default: -1).

    """

    variables: pd.Series
    fluxes: pd.Series

    @classmethod
    def from_result(
        cls,
        *,
        model: Model,
        result: Result | None,
        idx: int = -1,
    ) -> Self:
        """Initialize the Scan object.

        Args:
            model: The model object.
            result: Result of the simulation
            idx: Index to select specific row from concs and fluxes DataFrames.

        """
        if result is None:
            return cls(
                variables=_empty_conc_series(model),
                fluxes=_empty_flux_series(model),
            )

        return cls(
            variables=result.variables.iloc[idx],
            fluxes=result.fluxes.iloc[idx],
        )

    @property
    def results(self) -> pd.Series:
        """Get the combined results of concentrations and fluxes.

        Example:
            >>> time_point.results
            x1    1.0
            x2    0.5
            v1    0.1
            v2    0.2

        Returns:
            pd.Series: Combined series of concentrations and fluxes.

        """
        return pd.concat((self.variables, self.fluxes), axis=0)


@dataclass(slots=True)
class TimeCourse:
    """Represents a time course in a simulation.

    Attributes:
        variables: DataFrame of concentrations over time.
        fluxes: DataFrame of fluxes over time.

    """

    variables: pd.DataFrame
    fluxes: pd.DataFrame

    @classmethod
    def from_scan(
        cls,
        *,
        model: Model,
        time_points: Array,
        result: Result | None,
    ) -> Self:
        """Initialize the Scan object.

        Args:
            model (Model): The model object.
            time_points (Array): Array of time points.
            result: Result of the simulation

        """
        if result is None:
            return cls(
                _empty_conc_df(model, time_points),
                _empty_flux_df(model, time_points),
            )
        return cls(
            result.variables,
            result.fluxes,
        )

    @property
    def results(self) -> pd.DataFrame:
        """Get the combined results of concentrations and fluxes over time.

        Examples:
            >>> time_course.results
            Time   x1     x2     v1     v2
            0.0   1.0   1.00   1.00   1.00
            0.1   0.9   0.99   0.99   0.99
            0.2   0.8   0.99   0.99   0.99

        Returns:
            pd.DataFrame: Combined DataFrame of concentrations and fluxes.

        """
        return pd.concat((self.variables, self.fluxes), axis=1)


###############################################################################
# Workers
###############################################################################


class SteadyStateWorker(Protocol):
    """Worker function for steady-state simulations."""

    def __call__(
        self,
        model: Model,
        *,
        rel_norm: bool,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
    ) -> TimePoint:
        """Call the worker function."""
        ...


class TimeCourseWorker(Protocol):
    """Worker function for time-course simulations."""

    def __call__(
        self,
        model: Model,
        time_points: Array,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
    ) -> TimeCourse:
        """Call the worker function."""
        ...


class ProtocolWorker(Protocol):
    """Worker function for protocol-based simulations."""

    def __call__(
        self,
        model: Model,
        protocol: pd.DataFrame,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
        time_points_per_step: int = 10,
    ) -> TimeCourse:
        """Call the worker function."""
        ...


def _steady_state_worker(
    model: Model,
    *,
    rel_norm: bool,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
) -> TimePoint:
    """Simulate the model to steady state and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        rel_norm: Whether to use relative normalization.
        integrator: Function producing an integrator for the simulation.

    Returns:
        TimePoint: Object containing steady-state concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_to_steady_state(rel_norm=rel_norm)
            .get_result()
        )
    except ZeroDivisionError:
        res = None
    return TimePoint.from_result(model=model, result=res)


def _time_course_worker(
    model: Model,
    time_points: Array,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
) -> TimeCourse:
    """Simulate the model to steady state and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        time_points: Array of time points for the simulation.
        integrator: Integrator function to use for steady state calculation

    Returns:
        TimePoint: Object containing steady-state concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_time_course(time_points=time_points)
            .get_result()
        )
    except ZeroDivisionError:
        res = None
    return TimeCourse.from_scan(
        model=model,
        time_points=time_points,
        result=res,
    )


def _protocol_worker(
    model: Model,
    protocol: pd.DataFrame,
    *,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
    time_points_per_step: int = 10,
) -> TimeCourse:
    """Simulate the model over a protocol and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        protocol: DataFrame containing the protocol steps.
        integrator: Integrator function to use for steady state calculation
        time_points_per_step: Number of time points per protocol step.

    Returns:
        TimeCourse: Object containing protocol series concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_protocol(
                protocol=protocol,
                time_points_per_step=time_points_per_step,
            )
            .get_result()
        )
    except ZeroDivisionError:
        res = None

    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    return TimeCourse.from_scan(
        model=model,
        time_points=time_points,
        result=res,
    )


def steady_state(
    model: Model,
    *,
    to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    rel_norm: bool = False,
    cache: Cache | None = None,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: IntegratorType | None = None,
) -> SteadyStates:
    """Get steady-state results over supplied values.

    Args:
        model: Model instance to simulate.
        to_scan: DataFrame containing parameter or initial values to scan.
        y0: Initial conditions as a dictionary {variable: value}.
        parallel: Whether to execute in parallel (default: True).
        rel_norm: Whether to use relative normalization (default: False).
        cache: Optional cache to store and retrieve results.
        worker: Worker function to use for the simulation.
        integrator: Integrator function to use for steady state calculation

    Returns:
        SteadyStates: Steady-state results for each parameter set.

    Examples:
        >>> steady_state(
        >>>     model,
        >>>     parameters=pd.DataFrame({"k1": np.linspace(1, 2, 3)})
        >>> ).variables
        idx      x      y
        1.0   0.50   1.00
        1.5   0.75   1.50
        2.0   1.00   2.00

        >>> steady_state(
        >>>     model,
        >>>     parameters=cartesian_product({"k1": [1, 2], "k2": [3, 4]})
        >>> ).variables

        | idx    |    x |   y |
        | (1, 3) | 0.33 |   1 |
        | (1, 4) | 0.25 |   1 |
        | (2, 3) | 0.66 |   2 |
        | (2, 4) | 0.5  |   2 |

    """
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                rel_norm=rel_norm,
                integrator=integrator,
                y0=None,
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = pd.DataFrame({k: v.variables.T for k, v in res}).T
    fluxes = pd.DataFrame({k: v.fluxes.T for k, v in res}).T
    idx = (
        pd.Index(to_scan.iloc[:, 0])
        if to_scan.shape[1] == 1
        else pd.MultiIndex.from_frame(to_scan)
    )
    concs.index = idx
    fluxes.index = idx
    return SteadyStates(variables=concs, fluxes=fluxes, parameters=to_scan)


def time_course(
    model: Model,
    *,
    to_scan: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    integrator: IntegratorType | None = None,
    worker: TimeCourseWorker = _time_course_worker,
) -> TimeCourseByPars:
    """Get time course for each supplied parameter.

    Examples:
        >>> time_course(
        >>>     model,
        >>>     to_scan=pd.DataFrame({"k1": [1, 1.5, 2]}),
        >>>     time_points=np.linspace(0, 1, 3)
        >>> ).variables

        | (n, time) |        x |       y |
        |:----------|---------:|--------:|
        | (0, 0.0)  | 1        | 1       |
        | (0, 0.5)  | 0.68394  | 1.23865 |
        | (0, 1.0)  | 0.567668 | 1.23254 |
        | (1, 0.0)  | 1        | 1       |
        | (1, 0.5)  | 0.84197  | 1.31606 |
        | (1, 1.0)  | 0.783834 | 1.43233 |
        | (2, 0.0)  | 1        | 1       |
        | (2, 0.5)  | 1        | 1.39347 |
        | (2, 1.0)  | 1        | 1.63212 |

        >>> time_course(
        >>>     model,
        >>>     to_scan=cartesian_product({"k1": [1, 2], "k2": [3, 4]}),
        >>>     time_points=[0.0, 0.5, 1.0],
        >>> ).variables

        | (n, time) |        x |      y |
        |:----------|---------:|-------:|
        | (0, 0.0)  | 1        | 1      |
        | (0, 0.5)  | 0.482087 | 1.3834 |
        | (1, 0.0)  | 1        | 1      |
        | (1, 0.5)  | 0.351501 | 1.4712 |
        | (2, 0.0)  | 1        | 1      |

    Args:
        model: Model instance to simulate.
        to_scan: DataFrame containing parameter or initial values to scan.
        time_points: Array of time points for the simulation.
        y0: Initial conditions as a dictionary {variable: value}.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).
        worker: Worker function to use for the simulation.
        integrator: Integrator function to use for steady state calculation

    Returns:
        TimeCourseByPars: Time series results for each parameter set.


    """
    # We update the initial conditions separately here, because `to_scan` might also
    # contain initial conditions.
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                time_points=time_points,
                integrator=integrator,
                y0=None,  # See comment above
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = cast(dict, {k: v.variables for k, v in res})
    fluxes = cast(dict, {k: v.fluxes for k, v in res})
    return TimeCourseByPars(
        parameters=to_scan,
        variables=pd.concat(concs, names=["n", "time"]),
        fluxes=pd.concat(fluxes, names=["n", "time"]),
    )


def time_course_over_protocol(
    model: Model,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolByPars:
    """Get protocol series for each supplied parameter.

    Examples:
        >>> scan.time_course_over_protocol(
        ...     model,
        ...     parameters=pd.DataFrame({"k2": np.linspace(1, 2, 11)}),
        ...     protocol=make_protocol(
        ...         {
        ...             1: {"k1": 1},
        ...             2: {"k1": 2},
        ...         }
        ...     ),
        ... )

    Args:
        model: Model instance to simulate.
        to_scan: DataFrame containing parameter or initial values to scan.
        protocol: Protocol to follow for the simulation.
        time_points_per_step: Number of time points per protocol step (default: 10).
        y0: Initial conditions as a dictionary {variable: value}.
        parallel: Whether to execute in parallel (default: True).
        cache: Optional cache to store and retrieve results.
        worker: Worker function to use for the simulation.
        integrator: Integrator function to use for steady state calculation

    Returns:
        TimeCourseByPars: Protocol series results for each parameter set.

    """
    # We update the initial conditions separately here, because `to_scan` might also
    # contain initial conditions.
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                protocol=protocol,
                time_points_per_step=time_points_per_step,
                integrator=integrator,
                y0=None,
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    concs = cast(dict, {k: v.variables for k, v in res})
    fluxes = cast(dict, {k: v.fluxes for k, v in res})
    return ProtocolByPars(
        parameters=to_scan,
        protocol=protocol,
        variables=pd.concat(concs, names=["n", "time"]),
        fluxes=pd.concat(fluxes, names=["n", "time"]),
    )
