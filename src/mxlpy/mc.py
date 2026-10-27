# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Monte Carlo Analysis (MC) Module for Metabolic Models.

This module provides functions for performing Monte Carlo analysis on metabolic models.
It includes functionality for steady-state and time-course simulations, as well as
response coefficient calculations.

Functions:
    steady_state: Perform Monte Carlo analysis for steady-state simulations
    time_course: Perform Monte Carlo analysis for time-course simulations
    time_course_over_protocol: Perform Monte Carlo analysis for time-course simulations over a protocol
    parameter_scan_ss: Perform Monte Carlo analysis for steady-state parameter scans
    compound_elasticities: Calculate compound elasticities using Monte Carlo analysis
    parameter_elasticities: Calculate parameter elasticities using Monte Carlo analysis
    response_coefficients: Calculate response coefficients using Monte Carlo analysis
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, cast, overload

import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy import mca, scan
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.mca import (
    ResponseCoefficientsByPars,
    _require_names,
    _split_outputs,
)
from mxlpy.parallel import Cache, parallelise
from mxlpy.runners import (
    DynamicRunner,
    SteadyStateRunner,
    default_runner,
)
from mxlpy.scan import (
    ProtocolScan,
    SteadyStateScan,
    TimeCourseScan,
    _protocol_time_course_worker,
    _protocol_worker,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and_initial_conditions,
)
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mxlpy.types import Array


type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

# Defaults to Simulation, so bare result types keep their meaning
SE = TypeVar("SE", Simulation, OdeSimulation, Evaluation, default=Simulation)

__all__ = [
    "McSteadyStates",
    "parameter_elasticities",
    "protocol",
    "protocol_time_course",
    "response_coefficients",
    "scan_steady_state",
    "steady_state",
    "time_course",
    "variable_elasticities",
]


def _parameter_scan_worker[M: _Model](
    model: M,
    *,
    parameters: pd.DataFrame,
    runner: SteadyStateRunner[M, Any],
) -> SteadyStateScan[Any]:
    """Scan the steady states of one Monte Carlo sample."""
    return scan._steady_state(  # noqa: SLF001
        model,
        to_scan=parameters,
        parallel=False,
        runner=runner,
    )


@overload
def steady_state(
    model: KineticModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> SteadyStateScan[Simulation]: ...


@overload
def steady_state(
    model: OdeModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> SteadyStateScan[OdeSimulation]: ...


@overload
def steady_state(
    model: SteadyStateModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
) -> SteadyStateScan[Evaluation]: ...


def steady_state(
    model: _Model,
    *,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
) -> (
    SteadyStateScan[Simulation]
    | SteadyStateScan[OdeSimulation]
    | SteadyStateScan[Evaluation]
):
    """Monte-carlo scan of steady states.

    Parameters
    ----------
    model
        Model instance to simulate.
    mc_to_scan
        DataFrame of parameter/initial condition samples for Monte Carlo.
    y0
        Initial conditions as a dictionary {variable: value}.
    max_workers
        Maximum number of parallel workers. None uses all available cores.
    cache
        Optional cache to store and retrieve results.
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Examples
    --------
        >>> steady_state(model, mc_to_scan)
        p    t     x      y
        0    0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1    0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns
    -------
    SteadyStates
        Object containing the steady state solutions for the given parameter

    """
    model = scan._copy_with_y0(model, y0)  # noqa: SLF001

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _steady_state_worker,
                runner=default_runner(model) if runner is None else runner,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    return SteadyStateScan(
        raw_index=(
            pd.Index(mc_to_scan.iloc[:, 0])
            if mc_to_scan.shape[1] == 1
            else pd.MultiIndex.from_frame(mc_to_scan)
        ),
        raw_results=[i[1] for i in res],
        to_scan=mc_to_scan,
    )


@overload
def time_course(
    model: KineticModelBuilder,
    *,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> TimeCourseScan[Simulation]: ...


@overload
def time_course(
    model: OdeModelBuilder,
    *,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> TimeCourseScan[OdeSimulation]: ...


def time_course(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
) -> TimeCourseScan[Simulation] | TimeCourseScan[OdeSimulation]:
    """MC time course.

    Parameters
    ----------
    model
        Model instance to simulate.
    time_points
        Array of time points for the simulation.
    mc_to_scan
        DataFrame of parameter/initial condition samples for Monte Carlo.
    y0
        Initial conditions as a dictionary {variable: value}.
    max_workers
        Maximum number of parallel workers. None uses all available cores.
    cache
        Optional cache to store and retrieve results.
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Examples
    --------
        >>> time_course(model, time_points, mc_to_scan)
        p    t     x      y
        0   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns
    -------
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
    model = scan._copy_with_y0(model, y0)  # noqa: SLF001

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
        inputs=list(mc_to_scan.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )

    return TimeCourseScan(
        to_scan=mc_to_scan,
        raw_results=dict(res),
    )


@overload
def protocol(
    model: KineticModelBuilder,
    *,
    protocol: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> ProtocolScan[Simulation]: ...


@overload
def protocol(
    model: OdeModelBuilder,
    *,
    protocol: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> ProtocolScan[OdeSimulation]: ...


def protocol(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    protocol: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
) -> ProtocolScan[Simulation] | ProtocolScan[OdeSimulation]:
    """MC time course.

    Parameters
    ----------
    model
        Model instance to simulate.
    protocol
        DataFrame containing the protocol steps.
    mc_to_scan
        DataFrame of parameter/initial condition samples for Monte Carlo.
    y0
        Initial conditions as a dictionary {variable: value}.
    time_points_per_step
        Number of time points per protocol step.
    max_workers
        Maximum number of parallel workers. None uses all available cores.
    cache
        Optional cache to store and retrieve results.
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Examples
    --------
        >>> time_course_over_protocol(model, protocol, mc_to_scan)
        p    t     x      y
        0   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns
    -------
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
    model = scan._copy_with_y0(model, y0)  # noqa: SLF001

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
        inputs=list(mc_to_scan.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    return ProtocolScan(
        to_scan=mc_to_scan,
        protocol=protocol,
        raw_results=dict(res),
    )


@overload
def protocol_time_course(
    model: KineticModelBuilder,
    *,
    protocol: pd.DataFrame,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> ProtocolScan[Simulation]: ...


@overload
def protocol_time_course(
    model: OdeModelBuilder,
    *,
    protocol: pd.DataFrame,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> ProtocolScan[OdeSimulation]: ...


def protocol_time_course(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    protocol: pd.DataFrame,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
) -> ProtocolScan[Simulation] | ProtocolScan[OdeSimulation]:
    """MC time course.

    Parameters
    ----------
    model
        Model instance to simulate.
    protocol
        DataFrame containing the protocol steps.
    time_points
        Array of time points for the simulation.
    mc_to_scan
        DataFrame of parameter/initial condition samples for Monte Carlo.
    y0
        Initial conditions as a dictionary {variable: value}.
    max_workers
        Maximum number of parallel workers. None uses all available cores.
    cache
        Optional cache to store and retrieve results.
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Examples
    --------
        >>> protocol_time_course(model, protocol, time_points, mc_to_scan)
        p    t     x      y
        0   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03
            ...   ...    ...
        1   0.0   0.1    0.00
            1.0   0.2    0.01
            2.0   0.3    0.02
            3.0   0.4    0.03

    Returns
    -------
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
    model = scan._copy_with_y0(model, y0)  # noqa: SLF001

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
        inputs=list(mc_to_scan.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    return ProtocolScan(
        to_scan=mc_to_scan,
        protocol=protocol,
        raw_results=dict(res),
    )


@dataclass(kw_only=True, slots=True)
class McSteadyStates(Generic[SE]):
    """Container for Monte Carlo steady states.

    `combined` holds every model output (columns) by Monte Carlo sample and
    scanned value (rows). The split into `variables` and `fluxes` only exists
    for builders having them.
    """

    combined: pd.DataFrame
    parameters: pd.DataFrame
    mc_to_scan: pd.DataFrame
    variable_names: list[str] | None = None
    flux_names: list[str] | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(
        self: McSteadyStates[Simulation] | McSteadyStates[OdeSimulation],
    ) -> pd.DataFrame:
        """Return the steady-state variables."""
        return self.combined.loc[:, _require_names(self.variable_names, "variables")]

    @property
    def fluxes(self: McSteadyStates[Simulation]) -> pd.DataFrame:
        """Return the steady-state fluxes."""
        return self.combined.loc[:, _require_names(self.flux_names, "fluxes")]

    def __iter__(self: McSteadyStates[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))


@overload
def scan_steady_state(
    model: KineticModelBuilder,
    *,
    to_scan: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> McSteadyStates[Simulation]: ...


@overload
def scan_steady_state(
    model: OdeModelBuilder,
    *,
    to_scan: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> McSteadyStates[OdeSimulation]: ...


@overload
def scan_steady_state(
    model: SteadyStateModelBuilder,
    *,
    to_scan: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
) -> McSteadyStates[Evaluation]: ...


def scan_steady_state(
    model: _Model,
    *,
    to_scan: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
) -> (
    McSteadyStates[Simulation]
    | McSteadyStates[OdeSimulation]
    | McSteadyStates[Evaluation]
):
    """Parameter scan of mc distributed steady states.

    Examples
    --------
        >>> scan_steady_state(
        ...     model,
        ...     parameters=pd.DataFrame({"k1": np.linspace(0, 1, 3)}),
        ...     mc_to_scan=mc_to_scan,
        ... ).variables
                  x     y
          k1
        0 0.0 -0.00 -0.00
          0.5  0.44  0.20
          1.0  0.88  0.40
        1 0.0 -0.00 -0.00
          0.5  0.45  0.14
          1.0  0.90  0.28



    Parameters
    ----------
    model
        The model to analyze
    to_scan
        DataFrame containing parameter and initial values to scan over
    mc_to_scan
        DataFrame containing Monte Carlo parameter sets
    y0
        Initial conditions for the solver
    max_workers
        Maximum number of workers for parallel processing
    cache
        Cache object for storing results
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Returns
    -------
    McSteadyStates
        Object containing the steady state solutions for the given parameter

    """
    model = scan._copy_with_y0(model, y0)  # noqa: SLF001

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                _parameter_scan_worker,
                parameters=to_scan,
                runner=default_runner(model) if runner is None else runner,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    combined = cast(
        pd.DataFrame,
        pd.concat(
            {
                k: pd.DataFrame(
                    [i.get_combined().iloc[-1] for i in v.raw_results],
                    index=v.raw_index,
                )
                for k, v in res
            }
        ),
    )
    variable_names, flux_names = _split_outputs(model, list(combined.columns))
    return McSteadyStates(
        combined=combined,
        parameters=to_scan,
        mc_to_scan=mc_to_scan,
        variable_names=variable_names,
        flux_names=flux_names,
    )


###############################################################################
# MCA
###############################################################################


def variable_elasticities(
    model: KineticModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    time: float = 0,
    cache: Cache | None = None,
    max_workers: int | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate variable elasticities using Monte Carlo analysis.

    Examples
    --------
        >>> variable_elasticities(
        ...     model,
        ...     variables=["x1", "x2"],
        ...     concs={"x1": 1, "x2": 2},
        ...     mc_to_scan=mc_to_scan
        ... )
                 x1     x2
        0   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4
        1   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4

    Parameters
    ----------
    model
        The model to analyze
    to_scan
        List of variables for which to calculate elasticities
    variables
        Custom variable values. Defaults to initial conditions.
    mc_to_scan
        DataFrame containing Monte Carlo parameter sets
    time
        Time point for the analysis
    cache
        Cache object for storing results
    max_workers
        Maximum number of workers for parallel processing
    normalized
        Whether to use normalized elasticities
    displacement
        Displacement for finite difference calculations

    Returns
    -------
    pd.DataFrame
        DataFrame containing the compound elasticities for the given variables

    """
    model = deepcopy(model)
    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                mca.variable_elasticities,
                variables=variables,
                to_scan=to_scan,
                time=time,
                displacement=displacement,
                normalized=normalized,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    return cast(pd.DataFrame, pd.concat(dict(res)))


def parameter_elasticities(
    model: KineticModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    variables: dict[str, float],
    time: float = 0,
    cache: Cache | None = None,
    max_workers: int | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate parameter elasticities using Monte Carlo analysis.

    Examples
    --------
        >>> parameter_elasticities(
        ...     model,
        ...     parameters=["p1", "p2"],
        ...     concs={"x1": 1, "x2": 2},
        ...     mc_to_scan=mc_to_scan
        ... )
                 p1     p2
        0   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4
        1   v1  0.0    0.0
            v2  1.0    0.0
            v3  0.0   -1.4

    Parameters
    ----------
    model
        The model to analyze
    to_scan
        List of parameters for which to calculate elasticities
    variables
        Custom variable values. Defaults to initial conditions.
    mc_to_scan
        DataFrame containing Monte Carlo parameter sets
    time
        Time point for the analysis
    cache
        Cache object for storing results
    max_workers
        Maximum number of workers for parallel processing
    normalized
        Whether to use normalized elasticities
    displacement
        Displacement for finite difference calculations

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parameter elasticities for the given variables

    """
    model = deepcopy(model)
    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                mca.parameter_elasticities,
                to_scan=to_scan,
                variables=variables,
                time=time,
                displacement=displacement,
                normalized=normalized,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    return cast(pd.DataFrame, pd.concat(dict(res)))


@overload
def response_coefficients(
    model: KineticModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    variables: dict[str, float] | None = None,
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> ResponseCoefficientsByPars[Simulation]: ...


@overload
def response_coefficients(
    model: OdeModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    variables: dict[str, float] | None = None,
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> ResponseCoefficientsByPars[OdeSimulation]: ...


@overload
def response_coefficients(
    model: SteadyStateModelBuilder,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
) -> ResponseCoefficientsByPars[Evaluation]: ...


def response_coefficients(
    model: _Model,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    variables: dict[str, float] | None = None,
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
) -> (
    ResponseCoefficientsByPars[Simulation]
    | ResponseCoefficientsByPars[OdeSimulation]
    | ResponseCoefficientsByPars[Evaluation]
):
    """Calculate response coefficients using Monte Carlo analysis.

    Examples
    --------
        >>> response_coefficients(
        ...     model,
        ...     parameters=["vmax1", "vmax2"],
        ...     mc_to_scan=mc_to_scan,
        ... ).variables
                    x1    x2
        0 vmax_1  0.01  0.01
          vmax_2  0.02  0.02
        1 vmax_1  0.03  0.03
          vmax_2  0.04  0.04

    Parameters
    ----------
    model
        The model to analyze
    mc_to_scan
        DataFrame containing Monte Carlo parameter sets
    to_scan
        List of parameters for which to calculate elasticities
    variables
        Custom variable values. Defaults to initial conditions.
    cache
        Cache object for storing results
    normalized
        Whether to use normalized elasticities
    displacement
        Displacement for finite difference calculations
    disable_tqdm
        Whether to disable the tqdm progress bar
    max_workers
        Maximum number of workers for parallel processing
    runner
        How each model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Returns
    -------
    ResponseCoefficientsByPars
        Object containing the response coefficients for the given parameters

    """
    model = scan._copy_with_y0(model, variables)  # noqa: SLF001

    res = parallelise(
        fn=partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                mca._response_coefficients,  # noqa: SLF001
                to_scan=to_scan,
                normalized=normalized,
                displacement=displacement,
                disable_tqdm=disable_tqdm,
                parallel=False,
                runner=runner,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )

    first = res[0][1]
    return ResponseCoefficientsByPars(
        combined=cast(pd.DataFrame, pd.concat({k: v.combined for k, v in res})),
        parameters=mc_to_scan,
        variable_names=first.variable_names,
        flux_names=first.flux_names,
    )
