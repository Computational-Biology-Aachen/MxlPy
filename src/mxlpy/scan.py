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

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy.parallel import Cache, parallelise
from mxlpy.simulation import Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator

    from mxlpy.integrators import IntegratorType
    from mxlpy.model import Model
    from mxlpy.types import Array


__all__ = [
    "ProtocolScan",
    "ProtocolTimeCourseWorker",
    "ProtocolWorker",
    "SteadyStateScan",
    "SteadyStateWorker",
    "TimeCourseScan",
    "TimeCourseWorker",
    "protocol",
    "protocol_time_course",
    "steady_state",
    "time_course",
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
    ) -> Simulation:
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
    ) -> Simulation:
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
    ) -> Simulation:
        """Call the worker function."""
        ...


class ProtocolTimeCourseWorker(Protocol):
    """Worker function for protocol-based simulations."""

    def __call__(
        self,
        model: Model,
        protocol: pd.DataFrame,
        time_points: Array,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
    ) -> Simulation:
        """Call the worker function."""
        ...


def _steady_state_worker(
    model: Model,
    *,
    rel_norm: bool,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
) -> Simulation:
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
        res = Result(Exception())

    return res.default(
        lambda: Simulation.default(model=model, time_points=np.array([0.0]))
    )


def _time_course_worker(
    model: Model,
    time_points: Array,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
) -> Simulation:
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
        res = Result(Exception())

    return res.default(lambda: Simulation.default(model=model, time_points=time_points))


def _protocol_worker(
    model: Model,
    protocol: pd.DataFrame,
    *,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
    time_points_per_step: int = 10,
) -> Simulation:
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
        res = Result(Exception())

    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    return res.default(lambda: Simulation.default(model=model, time_points=time_points))


def _protocol_time_course_worker(
    model: Model,
    protocol: pd.DataFrame,
    time_points: Array,
    *,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
) -> Simulation:
    """Simulate the model over a protocol and return concentrations and fluxes.

    Args:
        model: Model instance to simulate.
        y0: Initial conditions as a dictionary {species: value}.
        protocol: DataFrame containing the protocol steps.
        time_points: Time points where to return the simulation
        integrator: Integrator function to use for steady state calculation

    Returns:
        TimeCourse: Object containing protocol series concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_protocol_time_course(
                protocol=protocol,
                time_points=time_points,
            )
            .get_result()
        )
    except ZeroDivisionError:
        res = Result(Exception())

    return res.default(lambda: Simulation.default(model=model, time_points=time_points))


@dataclass(kw_only=True, slots=True)
class SteadyStateScan:
    """Container for steady states by scanned values."""

    to_scan: pd.DataFrame
    raw_index: pd.Index | pd.MultiIndex
    raw_results: list[Simulation]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(self) -> pd.DataFrame:
        """Return steady-state variables by scan."""
        return pd.DataFrame(
            [i.variables.iloc[-1].T for i in self.raw_results], index=self.raw_index
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return steady-state fluxes by scan."""
        return pd.DataFrame(
            [i.fluxes.iloc[-1].T for i in self.raw_results], index=self.raw_index
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return steady-state args by scan."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return steady-state args by scan."""
        return pd.DataFrame(
            [
                i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                .iloc[-1]
                .T
                for i in self.raw_results
            ],
            index=self.raw_index,
        )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))


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
) -> SteadyStateScan:
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

    return SteadyStateScan(
        raw_index=(
            pd.Index(to_scan.iloc[:, 0])
            if to_scan.shape[1] == 1
            else pd.MultiIndex.from_frame(to_scan)
        ),
        raw_results=[i[1] for i in res],
        to_scan=to_scan,
    )


@dataclass(kw_only=True, slots=True)
class TimeCourseScan:
    """Container for time courses by scanned values."""

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    to_scan: pd.DataFrame
    raw_results: dict[Hashable, Simulation]

    @property
    def variables(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.variables for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {
                k: i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                for k, i in self.raw_results.items()
            },
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get time courses by name."""
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux time courses."""
        return iter((self.variables, self.fluxes))


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
) -> TimeCourseScan:
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
    return TimeCourseScan(
        to_scan=to_scan,
        raw_results=dict(res),
    )


@dataclass(kw_only=True, slots=True)
class ProtocolScan:
    """Container for protocols by scanned values."""

    to_scan: pd.DataFrame
    protocol: pd.DataFrame
    raw_results: dict[Hashable, Simulation]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.variables for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {
                k: i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                for k, i in self.raw_results.items()
            },
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get concentration or flux by name."""
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux protocols."""
        return iter((self.variables, self.fluxes))


def protocol(
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
) -> ProtocolScan:
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
    return ProtocolScan(
        to_scan=to_scan,
        protocol=protocol,
        raw_results=dict(res),
    )


def protocol_time_course(
    model: Model,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolTimeCourseWorker = _protocol_time_course_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan:
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
        time_points: Time points where to return simulation results
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
                time_points=time_points,
                integrator=integrator,
                y0=None,
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )
    return ProtocolScan(
        to_scan=to_scan,
        protocol=protocol,
        raw_results=dict(res),
    )
