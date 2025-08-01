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

from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from mxlpy import mca, scan
from mxlpy.parallel import Cache, parallelise
from mxlpy.scan import (
    ProtocolTimeCourseWorker,
    ProtocolWorker,
    SteadyStateWorker,
    TimeCourseWorker,
    _protocol_time_course_worker,
    _protocol_worker,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and_initial_conditions,
)
from mxlpy.types import (
    IntegratorType,
    McSteadyStates,
    ProtocolScan,
    ResponseCoefficientsByPars,
    SteadyStateScan,
    TimeCourseScan,
)

if TYPE_CHECKING:
    from mxlpy.model import Model
    from mxlpy.types import Array


__all__ = [
    "ParameterScanWorker",
    "parameter_elasticities",
    "protocol",
    "protocol_time_course",
    "response_coefficients",
    "scan_steady_state",
    "steady_state",
    "time_course",
    "variable_elasticities",
]


class ParameterScanWorker(Protocol):
    """Protocol for the parameter scan worker function."""

    def __call__(
        self,
        model: Model,
        *,
        parameters: pd.DataFrame,
        y0: dict[str, float] | None,
        rel_norm: bool,
        integrator: IntegratorType,
    ) -> SteadyStateScan:
        """Call the worker function."""
        ...


def _parameter_scan_worker(
    model: Model,
    *,
    parameters: pd.DataFrame,
    y0: dict[str, float] | None,
    rel_norm: bool,
    integrator: IntegratorType,
) -> SteadyStateScan:
    """Worker function for parallel steady state scanning across parameter sets.

    This function executes a parameter scan for steady state solutions for a
    given model and parameter combinations. It's designed to be used as a worker
    in parallel processing.

    Args: model : Model
        The model object to analyze
    y0 : dict[str, float] | None
        Initial conditions for the solver. If None, default initial conditions
        are used.
    parameters : pd.DataFrame
        DataFrame containing parameter combinations to scan over. Each row
        represents one parameter set.
    rel_norm : bool
        Whether to use relative normalization in the steady state calculations

    Returns:
        SteadyStates
            Object containing the steady state solutions for the given parameter
            combinations

    """
    return scan.steady_state(
        model,
        to_scan=parameters,
        parallel=False,
        rel_norm=rel_norm,
        integrator=integrator,
        y0=y0,
    )


def steady_state(
    model: Model,
    *,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = False,
    worker: SteadyStateWorker = _steady_state_worker,
    integrator: IntegratorType | None = None,
) -> SteadyStateScan:
    """Monte-carlo scan of steady states.

    Examples:
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

    Returns:
        SteadyStates: Object containing the steady state solutions for the given parameter

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


def time_course(
    model: Model,
    *,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    worker: TimeCourseWorker = _time_course_worker,
    integrator: IntegratorType | None = None,
) -> TimeCourseScan:
    """MC time course.

    Examples:
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
    Returns:
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                time_points=time_points,
                integrator=integrator,
                y0=None,
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


def protocol(
    model: Model,
    *,
    protocol: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    time_points_per_step: int = 10,
    max_workers: int | None = None,
    cache: Cache | None = None,
    worker: ProtocolWorker = _protocol_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan:
    """MC time course.

    Examples:
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

    Returns:
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                protocol=protocol,
                integrator=integrator,
                y0=None,
                time_points_per_step=time_points_per_step,
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


def protocol_time_course(
    model: Model,
    *,
    protocol: pd.DataFrame,
    time_points: Array,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    worker: ProtocolTimeCourseWorker = _protocol_time_course_worker,
    integrator: IntegratorType | None = None,
) -> ProtocolScan:
    """MC time course.

    Examples:
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

    Returns:
        tuple[concentrations, fluxes] using pandas multiindex
        Both dataframes are of shape (#time_points * #mc_to_scan, #variables)

    """
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
        inputs=list(mc_to_scan.iterrows()),
        max_workers=max_workers,
        cache=cache,
    )
    return ProtocolScan(
        to_scan=mc_to_scan,
        protocol=protocol,
        raw_results=dict(res),
    )


def scan_steady_state(
    model: Model,
    *,
    to_scan: pd.DataFrame,
    mc_to_scan: pd.DataFrame,
    y0: dict[str, float] | None = None,
    max_workers: int | None = None,
    cache: Cache | None = None,
    rel_norm: bool = False,
    worker: ParameterScanWorker = _parameter_scan_worker,
    integrator: IntegratorType | None = None,
) -> McSteadyStates:
    """Parameter scan of mc distributed steady states.

    Examples:
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



    Args:
        model: The model to analyze
        to_scan: DataFrame containing parameter and initial values to scan over
        mc_to_scan: DataFrame containing Monte Carlo parameter sets
        y0: Initial conditions for the solver
        max_workers: Maximum number of workers for parallel processing
        cache: Cache object for storing results
        rel_norm: Whether to use relative normalization in the steady state calculations
        worker: Worker function for parallel steady state scanning across parameter sets
        integrator: Function producing an integrator for the simulation.

    Returns:
        McSteadyStates: Object containing the steady state solutions for the given parameter

    """
    if y0 is not None:
        model.update_variables(y0)

    res = parallelise(
        partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                worker,
                parameters=to_scan,
                rel_norm=rel_norm,
                integrator=integrator,
                y0=None,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )
    concs = {k: v.variables.T for k, v in res}
    fluxes = {k: v.fluxes.T for k, v in res}
    return McSteadyStates(
        variables=pd.concat(concs, axis=1).T,
        fluxes=pd.concat(fluxes, axis=1).T,
        parameters=to_scan,
        mc_to_scan=mc_to_scan,
    )


###############################################################################
# MCA
###############################################################################


def variable_elasticities(
    model: Model,
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

    Examples:
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

    Args:
        model: The model to analyze
        to_scan: List of variables for which to calculate elasticities
        variables: Custom variable values. Defaults to initial conditions.
        mc_to_scan: DataFrame containing Monte Carlo parameter sets
        time: Time point for the analysis
        cache: Cache object for storing results
        max_workers: Maximum number of workers for parallel processing
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations

    Returns:
        pd.DataFrame: DataFrame containing the compound elasticities for the given variables

    """
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
    model: Model,
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

    Examples:
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

    Args:
        model: The model to analyze
        to_scan: List of parameters for which to calculate elasticities
        variables: Custom variable values. Defaults to initial conditions.
        mc_to_scan: DataFrame containing Monte Carlo parameter sets
        time: Time point for the analysis
        cache: Cache object for storing results
        max_workers: Maximum number of workers for parallel processing
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations

    Returns:
        pd.DataFrame: DataFrame containing the parameter elasticities for the given variables

    """
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


def response_coefficients(
    model: Model,
    *,
    mc_to_scan: pd.DataFrame,
    to_scan: list[str],
    variables: dict[str, float] | None = None,
    cache: Cache | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    max_workers: int | None = None,
    rel_norm: bool = False,
    integrator: IntegratorType | None = None,
) -> ResponseCoefficientsByPars:
    """Calculate response coefficients using Monte Carlo analysis.

    Examples:
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

    Args:
        model: The model to analyze
        mc_to_scan: DataFrame containing Monte Carlo parameter sets
        to_scan: List of parameters for which to calculate elasticities
        variables: Custom variable values. Defaults to initial conditions.
        cache: Cache object for storing results
        normalized: Whether to use normalized elasticities
        displacement: Displacement for finite difference calculations
        disable_tqdm: Whether to disable the tqdm progress bar
        max_workers: Maximum number of workers for parallel processing
        rel_norm: Whether to use relative normalization in the steady state calculations
        integrator: Function producing an integrator for the simulation.

    Returns:
        ResponseCoefficientsByPars: Object containing the response coefficients for the given parameters

    """
    if variables is not None:
        model.update_variables(variables)

    res = parallelise(
        fn=partial(
            _update_parameters_and_initial_conditions,
            fn=partial(
                mca.response_coefficients,
                to_scan=to_scan,
                normalized=normalized,
                displacement=displacement,
                rel_norm=rel_norm,
                disable_tqdm=disable_tqdm,
                parallel=False,
                integrator=integrator,
            ),
            model=model,
        ),
        inputs=list(mc_to_scan.iterrows()),
        cache=cache,
        max_workers=max_workers,
    )

    return ResponseCoefficientsByPars(
        variables=cast(pd.DataFrame, pd.concat({k: v.variables for k, v in res})),
        fluxes=cast(pd.DataFrame, pd.concat({k: v.fluxes for k, v in res})),
        parameters=mc_to_scan,
    )
