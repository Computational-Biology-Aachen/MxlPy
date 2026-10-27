# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Parameter Scanning Module.

This module provides functions and classes for performing parameter scans on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as protocol-based simulations.

How each model is simulated is configured by a runner, see `mxlpy.runners`.

Functions:
    steady_state: Get steady-state results over supplied parameters.
    time_course: Get time course for each supplied parameter.
    protocol: Get protocol course for each supplied parameter.
    protocol_time_course: Get protocol time course for each supplied parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Generic, cast, overload

import numpy as np
import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.parallel import Cache, parallelise
from mxlpy.runners import DynamicRunner, SteadyStateRunner, default_runner
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator

    from mxlpy.types import Array


__all__ = [
    "ProtocolScan",
    "SteadyStateScan",
    "TimeCourseScan",
    "protocol",
    "protocol_time_course",
    "steady_state",
    "time_course",
]


# Default to Simulation, so a bare scan result type keeps its meaning
S = TypeVar("S", Simulation, OdeSimulation, default=Simulation)
SE = TypeVar("SE", Simulation, OdeSimulation, Evaluation, default=Simulation)

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder


def _update_parameters_and_initial_conditions[M: _Model, T](
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


def _copy_with_y0[M: _Model](model: M, y0: dict[str, float] | None) -> M:
    """Copy the model, so routines never mutate the caller's model."""
    model = deepcopy(model)
    if y0 is not None:
        if isinstance(model, SteadyStateModelBuilder):
            msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
            raise TypeError(msg)
        model.update_variables(y0)
    return model


def _raw_index(to_scan: pd.DataFrame) -> pd.Index | pd.MultiIndex:
    if to_scan.shape[1] == 1:
        return pd.Index(to_scan.iloc[:, 0])
    return pd.MultiIndex.from_frame(to_scan)


def _require_fluxes(results: Iterable[Simulation | OdeSimulation | Evaluation]) -> None:
    """Raise a helpful error instead of an AttributeError for flux-less results."""
    if any(not isinstance(i, Simulation) for i in results):
        msg = "Only scans of a KineticModelBuilder have fluxes and can be unpacked into (variables, fluxes) - use .variables or .combined instead"
        raise TypeError(msg)


def _require_variables(
    results: Iterable[Simulation | OdeSimulation | Evaluation],
) -> None:
    """Raise a helpful error instead of an AttributeError for evaluations."""
    if any(isinstance(i, Evaluation) for i in results):
        msg = "Scans of a SteadyStateModelBuilder have no variables - use .combined instead"
        raise TypeError(msg)


###############################################################################
# Workers
###############################################################################


def _default_result(
    model: _Model, time_points: Array
) -> Simulation | OdeSimulation | Evaluation:
    """Result filled with NaNs, used if a simulation fails."""
    if isinstance(model, SteadyStateModelBuilder):
        return Evaluation.default(model)
    if isinstance(model, OdeModelBuilder):
        return OdeSimulation.default(model=model, time_points=time_points)
    return Simulation.default(model=model, time_points=time_points)


def _steady_state_worker[M: _Model, R](
    model: M,
    *,
    runner: SteadyStateRunner[M, R],
) -> R:
    return runner.steady_state(model).default(
        lambda: cast("R", _default_result(model, np.array([0.0])))
    )


def _time_course_worker[M: _Model, R](
    model: M,
    *,
    time_points: Array,
    runner: DynamicRunner[M, R],
) -> R:
    return runner.time_course(model, time_points).default(
        lambda: cast("R", _default_result(model, time_points))
    )


def _protocol_worker[M: _Model, R](
    model: M,
    *,
    protocol: pd.DataFrame,
    time_points_per_step: int,
    runner: DynamicRunner[M, R],
) -> R:
    time_points = np.linspace(
        0,
        protocol.index[-1].total_seconds(),
        len(protocol) * time_points_per_step,
    )
    return runner.protocol(
        model, protocol, time_points_per_step=time_points_per_step
    ).default(lambda: cast("R", _default_result(model, time_points)))


def _protocol_time_course_worker[M: _Model, R](
    model: M,
    *,
    protocol: pd.DataFrame,
    time_points: Array,
    runner: DynamicRunner[M, R],
) -> R:
    return runner.protocol_time_course(model, protocol, time_points).default(
        lambda: cast("R", _default_result(model, time_points))
    )


@dataclass(kw_only=True, slots=True)
class SteadyStateScan(Generic[SE]):
    """Container for steady states by scanned values."""

    to_scan: pd.DataFrame
    raw_index: pd.Index | pd.MultiIndex
    raw_results: list[SE]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(
        self: SteadyStateScan[Simulation] | SteadyStateScan[OdeSimulation],
    ) -> pd.DataFrame:
        """Return steady-state variables by scan."""
        _require_variables(self.raw_results)
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

    @overload
    def get_args(
        self: SteadyStateScan[Evaluation],
        *,
        include_parameters: bool = False,
    ) -> pd.DataFrame: ...

    def get_args(self, **kwargs: bool) -> pd.DataFrame:
        """Return args by scan.

        Accepts the `include_*` flags of `Simulation.get_args`,
        `OdeSimulation.get_args` or `Evaluation.get_args` respectively, but
        defaults to including readouts.

        Returns
        -------
        pd.DataFrame
            Steady-state args indexed by scan values.

        """
        return self._get_args(kwargs)

    def _get_args(self, kwargs: dict[str, bool]) -> pd.DataFrame:
        if not any(isinstance(i, Evaluation) for i in self.raw_results):
            kwargs = {"include_readouts": True} | kwargs
        return pd.DataFrame(
            [i.get_args(**kwargs).iloc[-1].T for i in self.raw_results],
            index=self.raw_index,
        )

    def __iter__(self: SteadyStateScan[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))


@overload
def steady_state(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> SteadyStateScan[Simulation]: ...


@overload
def steady_state(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> SteadyStateScan[OdeSimulation]: ...


@overload
def steady_state(
    model: SteadyStateModelBuilder,
    *,
    to_scan: pd.DataFrame,
    parallel: bool = False,
    cache: Cache | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
) -> SteadyStateScan[Evaluation]: ...


def steady_state(
    model: _Model,
    *,
    to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    parallel: bool | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner | None = None,
) -> (
    SteadyStateScan[Simulation]
    | SteadyStateScan[OdeSimulation]
    | SteadyStateScan[Evaluation]
):
    """Get steady-state results over supplied values.

    Parameters
    ----------
    model
        Model instance to simulate. It is copied, so it isn't changed.
    to_scan
        DataFrame containing parameter or initial values to scan.
    y0
        Initial conditions as a dictionary {variable: value}.
    parallel
        Whether to execute in parallel. Defaults to True, except for a
        `SteadyStateModelBuilder`, whose cheap evaluations don't make up for
        the process pool's overhead.
    cache
        Optional cache to store and retrieve results.
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.

    Returns
    -------
    SteadyStateScan
        Steady-state results for each parameter set.

    Examples
    --------
        >>> steady_state(
        >>>     model,
        >>>     to_scan=pd.DataFrame({"k1": np.linspace(1, 2, 3)})
        >>> ).variables
        idx      x      y
        1.0   0.50   1.00
        1.5   0.75   1.50
        2.0   1.00   2.00

        >>> steady_state(
        >>>     model,
        >>>     to_scan=cartesian_product({"k1": [1, 2], "k2": [3, 4]}),
        >>>     runner=KineticRunner(tolerance=1e-8),
        >>> ).variables

        | idx    |    x |   y |
        | (1, 3) | 0.33 |   1 |
        | (1, 4) | 0.25 |   1 |
        | (2, 3) | 0.66 |   2 |
        | (2, 4) | 0.5  |   2 |

    """
    return _steady_state(
        model,
        to_scan=to_scan,
        y0=y0,
        parallel=parallel,
        cache=cache,
        runner=runner,
    )


def _steady_state(
    model: _Model,
    *,
    to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    parallel: bool | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner | None = None,
) -> (
    SteadyStateScan[Simulation]
    | SteadyStateScan[OdeSimulation]
    | SteadyStateScan[Evaluation]
):
    """Non-overloaded implementation of `steady_state`, for internal callers."""
    if parallel is None:
        parallel = not isinstance(model, SteadyStateModelBuilder)
    model = _copy_with_y0(model, y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _steady_state_worker,
                runner=default_runner(model) if runner is None else runner,
            ),
            model=model,
        ),
        inputs=list(to_scan.iterrows()),
        cache=cache,
        parallel=parallel,
    )

    return SteadyStateScan(
        raw_index=_raw_index(to_scan),
        raw_results=[i[1] for i in res],
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
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
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
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> TimeCourseScan[OdeSimulation]: ...


def time_course(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    time_points: Array,
    y0: dict[str, float] | None = None,
    parallel: bool = True,
    cache: Cache | None = None,
    runner: DynamicRunner | None = None,
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
        Model instance to simulate. It is copied, so it isn't changed.
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
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.

    Returns
    -------
    TimeCourseByPars
        Time series results for each parameter set.


    """
    model = _copy_with_y0(model, y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _time_course_worker,
                time_points=time_points,
                runner=default_runner(model) if runner is None else runner,
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
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
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
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
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
    runner: DynamicRunner | None = None,
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
        Model instance to simulate. It is copied, so it isn't changed.
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
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.

    Returns
    -------
    TimeCourseByPars
        Protocol series results for each parameter set.

    """
    model = _copy_with_y0(model, y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _protocol_worker,
                protocol=protocol,
                time_points_per_step=time_points_per_step,
                runner=default_runner(model) if runner is None else runner,
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
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
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
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
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
    runner: DynamicRunner | None = None,
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
        Model instance to simulate. It is copied, so it isn't changed.
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
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.

    Returns
    -------
    TimeCourseByPars
        Protocol series results for each parameter set.

    """
    model = _copy_with_y0(model, y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _protocol_time_course_worker,
                protocol=protocol,
                time_points=time_points,
                runner=default_runner(model) if runner is None else runner,
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
