# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

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
from typing import TYPE_CHECKING, Generic, Protocol, cast, overload

import numpy as np
import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.parallel import Cache, parallelise
from mxlpy.simulation import OdeSimulation, Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.types import Array


__all__ = [
    "EvaluationScan",
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


# Defaults to Simulation, so a bare scan result type keeps its meaning
S = TypeVar("S", Simulation, OdeSimulation, default=Simulation)


def _update_parameters_and_initial_conditions[
    M: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
    T,
](
    pars: pd.Series,
    fn: Callable[[M], T],
    model: M,
) -> T:
    """Update model parameters and execute a function.

    Parameters
    ----------
    pars
        Series containing parameter values to update.
    fn
        Function to execute after updating parameters.
    model
        Model instance to update.

    Returns
    -------
        Result of the function execution.

    """
    pd: dict[str, float] = cast(dict, pars.to_dict())
    if not isinstance(model, SteadyStateModelBuilder):
        variables = set(model.get_variable_names())
        model.update_variables({k: v for k, v in pd.items() if k in variables})
    parameters = set(model.get_parameter_names())
    model.update_parameters({k: v for k, v in pd.items() if k in parameters})
    return fn(model)


def _raw_index(to_scan: pd.DataFrame) -> pd.Index | pd.MultiIndex:
    if to_scan.shape[1] == 1:
        return pd.Index(to_scan.iloc[:, 0])
    return pd.MultiIndex.from_frame(to_scan)


def _require_fluxes(results: Iterable[Simulation | OdeSimulation]) -> None:
    """Raise a helpful error instead of an AttributeError for ODE results."""
    if any(isinstance(i, OdeSimulation) for i in results):
        msg = "Scans of an OdeModelBuilder have no fluxes and can't be unpacked into (variables, fluxes) - use .variables or .combined instead"
        raise TypeError(msg)


def _unwrap_or_default(
    res: Result[Simulation] | Result[OdeSimulation] | Result[Exception],
    model: KineticModelBuilder | OdeModelBuilder,
    time_points: Array,
) -> Simulation | OdeSimulation:
    """Unwrap a simulation result, falling back to NaNs on failure."""
    if isinstance(val := res.value, Simulation | OdeSimulation):
        return val
    if isinstance(model, OdeModelBuilder):
        return OdeSimulation.default(model=model, time_points=time_points)
    return Simulation.default(model=model, time_points=time_points)


###############################################################################
# Workers
###############################################################################


class SteadyStateWorker(Protocol):
    """Worker function for steady-state simulations."""

    def __call__(
        self,
        model: KineticModelBuilder | OdeModelBuilder,
        *,
        tolerance: float = 1e-6,
        rel_norm: bool,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
        oscillation_detector: OscillationDetector | None,
    ) -> Simulation | OdeSimulation:
        """Call the worker function."""
        ...


class TimeCourseWorker(Protocol):
    """Worker function for time-course simulations."""

    def __call__(
        self,
        model: KineticModelBuilder | OdeModelBuilder,
        time_points: Array,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
    ) -> Simulation | OdeSimulation:
        """Call the worker function."""
        ...


class ProtocolWorker(Protocol):
    """Worker function for protocol-based simulations."""

    def __call__(
        self,
        model: KineticModelBuilder | OdeModelBuilder,
        protocol: pd.DataFrame,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
        time_points_per_step: int = 10,
    ) -> Simulation | OdeSimulation:
        """Call the worker function."""
        ...


class ProtocolTimeCourseWorker(Protocol):
    """Worker function for protocol-based simulations."""

    def __call__(
        self,
        model: KineticModelBuilder | OdeModelBuilder,
        protocol: pd.DataFrame,
        time_points: Array,
        *,
        integrator: IntegratorType | None,
        y0: dict[str, float] | None,
    ) -> Simulation | OdeSimulation:
        """Call the worker function."""
        ...


def _steady_state_worker(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    tolerance: float = 1e-6,
    rel_norm: bool,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> Simulation | OdeSimulation:
    """Simulate the model to steady state and return concentrations and fluxes.

    Parameters
    ----------
    model
        Model instance to simulate.
    tolerance
        Tolerance for the steady-state calculation.
    y0
        Initial conditions as a dictionary {species: value}.
    rel_norm
        Whether to use relative normalization.
    integrator
        Function producing an integrator for the simulation.
    oscillation_detector
        Callable for oscillation detection, or ``None`` to disable.

    Returns
    -------
    TimePoint
        Object containing steady-state concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_to_steady_state(
                tolerance=tolerance,
                rel_norm=rel_norm,
                oscillation_detector=oscillation_detector,
            )
            .get_result()
        )
    except ZeroDivisionError:
        res = Result(Exception())

    return _unwrap_or_default(res, model, np.array([0.0]))


def _time_course_worker(
    model: KineticModelBuilder | OdeModelBuilder,
    time_points: Array,
    y0: dict[str, float] | None,
    integrator: IntegratorType | None,
) -> Simulation | OdeSimulation:
    """Simulate the model to steady state and return concentrations and fluxes.

    Parameters
    ----------
    model
        Model instance to simulate.
    y0
        Initial conditions as a dictionary {species: value}.
    time_points
        Array of time points for the simulation.
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    TimePoint
        Object containing steady-state concentrations and fluxes.

    """
    try:
        res = (
            Simulator(model, integrator=integrator, y0=y0)
            .simulate_time_course(time_points=time_points)
            .get_result()
        )
    except ZeroDivisionError:
        res = Result(Exception())

    return _unwrap_or_default(res, model, time_points)


def _protocol_worker(
    model: KineticModelBuilder | OdeModelBuilder,
    protocol: pd.DataFrame,
    *,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
    time_points_per_step: int = 10,
) -> Simulation | OdeSimulation:
    """Simulate the model over a protocol and return concentrations and fluxes.

    Parameters
    ----------
    model
        Model instance to simulate.
    y0
        Initial conditions as a dictionary {species: value}.
    protocol
        DataFrame containing the protocol steps.
    integrator
        Integrator function to use for steady state calculation
    time_points_per_step
        Number of time points per protocol step.

    Returns
    -------
    TimeCourse
        Object containing protocol series concentrations and fluxes.

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
    return _unwrap_or_default(res, model, time_points)


def _protocol_time_course_worker(
    model: KineticModelBuilder | OdeModelBuilder,
    protocol: pd.DataFrame,
    time_points: Array,
    *,
    integrator: IntegratorType | None,
    y0: dict[str, float] | None,
) -> Simulation | OdeSimulation:
    """Simulate the model over a protocol and return concentrations and fluxes.

    Parameters
    ----------
    model
        Model instance to simulate.
    y0
        Initial conditions as a dictionary {species: value}.
    protocol
        DataFrame containing the protocol steps.
    time_points
        Time points where to return the simulation
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    TimeCourse
        Object containing protocol series concentrations and fluxes.

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

    return _unwrap_or_default(res, model, time_points)


@dataclass(kw_only=True, slots=True)
class SteadyStateScan(Generic[S]):
    """Container for steady states by scanned values."""

    to_scan: pd.DataFrame
    raw_index: pd.Index | pd.MultiIndex
    raw_results: list[S]

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
    def fluxes(self: SteadyStateScan[Simulation]) -> pd.DataFrame:
        """Return steady-state fluxes by scan."""
        _require_fluxes(self.raw_results)
        return pd.DataFrame(
            [i.fluxes.iloc[-1].T for i in self.raw_results], index=self.raw_index
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return steady-state args by scan."""
        return self._get_args({})

    @overload
    def get_args(
        self: SteadyStateScan[Simulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self: SteadyStateScan[OdeSimulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    def get_args(self, **kwargs: bool) -> pd.DataFrame:
        """Return args by scan.

        Accepts the `include_*` flags of `Simulation.get_args` or
        `OdeSimulation.get_args` respectively, but defaults to including readouts.

        Returns
        -------
        pd.DataFrame
            Steady-state args indexed by scan values.

        """
        return self._get_args(kwargs)

    def _get_args(self, kwargs: dict[str, bool]) -> pd.DataFrame:
        kwargs = {"include_readouts": True} | kwargs
        return pd.DataFrame(
            [i.get_args(**kwargs).iloc[-1].T for i in self.raw_results],
            index=self.raw_index,
        )

    def __iter__(self: SteadyStateScan[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class EvaluationScan:
    """Container for `SteadyStateModelBuilder` evaluations by scanned values."""

    to_scan: pd.DataFrame
    raw_index: pd.Index | pd.MultiIndex
    raw_results: list[pd.Series]
    parameter_names: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return evaluated values by scan."""
        return self.get_args()

    def get_args(self, *, include_parameters: bool = False) -> pd.DataFrame:
        """Return evaluated values by scan.

        Parameters
        ----------
        include_parameters
            Include model parameters.

        Returns
        -------
        pd.DataFrame
            Evaluated values indexed by scan values.

        """
        args = pd.DataFrame(self.raw_results, index=self.raw_index)
        if include_parameters:
            return args
        return args.drop(columns=self.parameter_names)


def _evaluate_worker(model: SteadyStateModelBuilder) -> pd.Series:
    """Evaluate the model, returning NaNs on arithmetic or domain errors."""
    try:
        return model.evaluate(include_parameters=True)
    except (ArithmeticError, ValueError):
        return pd.Series(
            np.nan,
            index=model.get_parameter_names()
            + list(model.get_raw_derived(as_copy=False)),
            dtype=float,
        )


@overload
def steady_state(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    tolerance: float = 1e-6,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    rel_norm: bool = False,
    cache: Cache | None = None,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: IntegratorType | None = None,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> SteadyStateScan[Simulation]: ...


@overload
def steady_state(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    tolerance: float = 1e-6,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    rel_norm: bool = False,
    cache: Cache | None = None,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: IntegratorType | None = None,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> SteadyStateScan[OdeSimulation]: ...


@overload
def steady_state(
    model: SteadyStateModelBuilder,
    *,
    to_scan: pd.DataFrame,
    parallel: bool = False,
    cache: Cache | None = None,
) -> EvaluationScan: ...


def steady_state(
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
    *,
    to_scan: pd.DataFrame,
    tolerance: float = 1e-6,
    y0: dict[str, float] | None = None,
    parallel: bool | None = None,
    rel_norm: bool = False,
    cache: Cache | None = None,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: IntegratorType | None = None,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> SteadyStateScan[Simulation] | SteadyStateScan[OdeSimulation] | EvaluationScan:
    """Get steady-state results over supplied values.

    Parameters
    ----------
    model
        Model instance to simulate.
    to_scan
        DataFrame containing parameter or initial values to scan.
    tolerance
        Tolerance for the steady-state calculation.
    y0
        Initial conditions as a dictionary {variable: value}.
    parallel
        Whether to execute in parallel. Defaults to True, except for a
        `SteadyStateModelBuilder`, whose cheap evaluations don't make up for
        the process pool's overhead.
    rel_norm
        Whether to use relative normalization (default: False).
    cache
        Optional cache to store and retrieve results.
    worker
        Worker function to use for the simulation.
    integrator
        Integrator function to use for steady state calculation.
    oscillation_detector
        Callable for oscillation detection passed to each worker.  Pass
        :func:`~mxlpy.integrators.utils.no_oscillation_detection` to
        disable.  Default: :func:`~mxlpy.integrators.utils.detect_oscillations`.

    `SteadyStateModelBuilder` models are evaluated directly, so the
    simulation specific arguments don't apply to them.

    Returns
    -------
    SteadyStateScan | EvaluationScan
        Steady-state results for each parameter set.

    Examples
    --------
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
    if parallel is None:
        parallel = not isinstance(model, SteadyStateModelBuilder)

    if isinstance(model, SteadyStateModelBuilder):
        evaluated = parallelise(
            partial(
                _update_parameters_and_initial_conditions,
                fn=_evaluate_worker,
                model=model,
            ),
            inputs=list(to_scan.iterrows()),
            cache=cache,
            parallel=parallel,
        )
        return EvaluationScan(
            to_scan=to_scan,
            raw_index=_raw_index(to_scan),
            raw_results=[i[1] for i in evaluated],
            parameter_names=model.get_parameter_names(),
        )

    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                tolerance=tolerance,
                rel_norm=rel_norm,
                integrator=integrator,
                y0=None,
                oscillation_detector=oscillation_detector,
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )

    return SteadyStateScan(
        raw_index=_raw_index(to_scan),
        raw_results=[i[1] for i in res],  # type: ignore[misc]
        to_scan=to_scan,
    )


@dataclass(kw_only=True, slots=True)
class TimeCourseScan(Generic[S]):
    """Container for time courses by scanned values."""

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    to_scan: pd.DataFrame
    raw_results: dict[Hashable, S]

    @property
    def variables(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.variables for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def fluxes(self: TimeCourseScan[Simulation]) -> pd.DataFrame:
        """Return all args of the time courses."""
        _require_fluxes(self.raw_results.values())
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self._get_args({})

    @overload
    def get_args(
        self: TimeCourseScan[Simulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self: TimeCourseScan[OdeSimulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    def get_args(self, **kwargs: bool) -> pd.DataFrame:
        """Return args by scan.

        Accepts the `include_*` flags of `Simulation.get_args` or
        `OdeSimulation.get_args` respectively, but defaults to including readouts.

        Returns
        -------
        pd.DataFrame
            Combined args with multi-index (n, time).

        """
        return self._get_args(kwargs)

    def _get_args(self, kwargs: dict[str, bool]) -> pd.DataFrame:
        kwargs = {"include_readouts": True} | kwargs
        return pd.concat(
            {k: i.get_args(**kwargs) for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get time courses by name.

        Parameters
        ----------
        name
            Name of the variable or flux to retrieve.

        Returns
        -------
        pd.DataFrame
            Time course for the given name, with runs as columns.

        """
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses.

        Parameters
        ----------
        agg
            Aggregation function or name (e.g. "mean", "std").

        Returns
        -------
        pd.DataFrame
            Aggregated values across runs for each time point.

        """
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses.

        Parameters
        ----------
        agg
            Aggregation function or name (e.g. "mean", "std").

        Returns
        -------
        pd.DataFrame
            Aggregated values across time points for each run.

        """
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self: TimeCourseScan[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux time courses."""
        return iter((self.variables, self.fluxes))


@overload
def time_course(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    integrator: IntegratorType | None = None,
    worker: TimeCourseWorker = _time_course_worker,
) -> TimeCourseScan[Simulation]: ...


@overload
def time_course(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    integrator: IntegratorType | None = None,
    worker: TimeCourseWorker = _time_course_worker,
) -> TimeCourseScan[OdeSimulation]: ...


def time_course(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    integrator: IntegratorType | None = None,
    worker: TimeCourseWorker = _time_course_worker,
) -> TimeCourseScan[Simulation] | TimeCourseScan[OdeSimulation]:
    """Get time course for each supplied parameter.

    Examples
    --------
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

    Parameters
    ----------
    model
        Model instance to simulate.
    to_scan
        DataFrame containing parameter or initial values to scan.
    time_points
        Array of time points for the simulation.
    y0
        Initial conditions as a dictionary {variable: value}.
    cache
        Optional cache to store and retrieve results.
    parallel
        Whether to execute in parallel (default: True).
    worker
        Worker function to use for the simulation.
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    TimeCourseByPars
        Time series results for each parameter set.


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
        raw_results=dict(res),  # type: ignore[misc]
    )


@dataclass(kw_only=True, slots=True)
class ProtocolScan(Generic[S]):
    """Container for protocols by scanned values."""

    to_scan: pd.DataFrame
    protocol: pd.DataFrame
    raw_results: dict[Hashable, S]

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
    def fluxes(self: ProtocolScan[Simulation]) -> pd.DataFrame:
        """Return all args of the time courses."""
        _require_fluxes(self.raw_results.values())
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self._get_args({})

    @overload
    def get_args(
        self: ProtocolScan[Simulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self: ProtocolScan[OdeSimulation],
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = True,
    ) -> pd.DataFrame: ...

    def get_args(self, **kwargs: bool) -> pd.DataFrame:
        """Return args by scan.

        Accepts the `include_*` flags of `Simulation.get_args` or
        `OdeSimulation.get_args` respectively, but defaults to including readouts.

        Returns
        -------
        pd.DataFrame
            Combined args with multi-index (n, time).

        """
        return self._get_args(kwargs)

    def _get_args(self, kwargs: dict[str, bool]) -> pd.DataFrame:
        kwargs = {"include_readouts": True} | kwargs
        return pd.concat(
            {k: i.get_args(**kwargs) for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get concentration or flux by name.

        Parameters
        ----------
        name
            Name of the variable or flux to retrieve.

        Returns
        -------
        pd.DataFrame
            Time course for the given name, with runs as columns.

        """
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux.

        Parameters
        ----------
        agg
            Aggregation function or name (e.g. "mean", "std").

        Returns
        -------
        pd.DataFrame
            Aggregated values across runs for each time point.

        """
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux.

        Parameters
        ----------
        agg
            Aggregation function or name (e.g. "mean", "std").

        Returns
        -------
        pd.DataFrame
            Aggregated values across time points for each run.

        """
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self: ProtocolScan[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux protocols."""
        return iter((self.variables, self.fluxes))


@overload
def protocol(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[Simulation]: ...


@overload
def protocol(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[OdeSimulation]: ...


def protocol(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points_per_step: int = 10,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[Simulation] | ProtocolScan[OdeSimulation]:
    """Get protocol series for each supplied parameter.

    Examples
    --------
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

    Parameters
    ----------
    model
        Model instance to simulate.
    to_scan
        DataFrame containing parameter or initial values to scan.
    protocol
        Protocol to follow for the simulation.
    time_points_per_step
        Number of time points per protocol step (default: 10).
    y0
        Initial conditions as a dictionary {variable: value}.
    parallel
        Whether to execute in parallel (default: True).
    cache
        Optional cache to store and retrieve results.
    worker
        Worker function to use for the simulation.
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    TimeCourseByPars
        Protocol series results for each parameter set.

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
        raw_results=dict(res),  # type: ignore[misc]
    )


@overload
def protocol_time_course(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolTimeCourseWorker = _protocol_time_course_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[Simulation]: ...


@overload
def protocol_time_course(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolTimeCourseWorker = _protocol_time_course_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[OdeSimulation]: ...


def protocol_time_course(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    protocol: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    worker: ProtocolTimeCourseWorker = _protocol_time_course_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan[Simulation] | ProtocolScan[OdeSimulation]:
    """Get protocol series for each supplied parameter.

    Examples
    --------
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

    Parameters
    ----------
    model
        Model instance to simulate.
    to_scan
        DataFrame containing parameter or initial values to scan.
    protocol
        Protocol to follow for the simulation.
    time_points
        Time points where to return simulation results
    y0
        Initial conditions as a dictionary {variable: value}.
    parallel
        Whether to execute in parallel (default: True).
    cache
        Optional cache to store and retrieve results.
    worker
        Worker function to use for the simulation.
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    TimeCourseByPars
        Protocol series results for each parameter set.

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
        raw_results=dict(res),  # type: ignore[misc]
    )
