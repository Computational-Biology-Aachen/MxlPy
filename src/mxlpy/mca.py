# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Metabolic Control Analysis (MCA) Module.

Provides functions for analyzing control and regulation in metabolic networks through:
- Elasticity coefficients (variable and parameter)
- Response coefficients

Main Functions:
    variable_elasticities: Calculate non-steady state variable elasticities
    parameter_elasticities: Calculate non-steady state parameter elasticities
    response_coefficients: Calculate response coefficients for steady state

Mathematical Background:
    MCA quantifies how changes in system parameters affect metabolic variables
    through elasticity and control coefficients. These describe the sensitivity
    of reaction rates and steady state variables to perturbations.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, cast, overload

import numpy as np
import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.forward_sensitivities._core import (
    _check_parameters,
    _Derivatives,
    _free_parameters,
    _setup,
    _step,
)
from mxlpy.parallel import parallelise
from mxlpy.plot import _default_fig_ax, _default_fig_axs
from mxlpy.runners import (
    DynamicRunner,
    KineticRunner,
    SteadyStateRunner,
    default_runner,
    precompile,
)
from mxlpy.scan import (
    _copy_with_y0,
    _steady_state_worker,
    _time_course_worker,
    steady_state,
)
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation, _align_time
from mxlpy.symbolic.jacobian import compile_derivatives
from mxlpy.symbolic.symbolic_model import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from mxlpy.plot import FigAx, FigAxs
    from mxlpy.types import Array, ArrayLike

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

_LOGGER = logging.getLogger(__name__)

# Defaults to Simulation, so bare result types keep their meaning
SE = TypeVar("SE", Simulation, OdeSimulation, Evaluation, default=Simulation)

__all__ = [
    "RateCharacteristics",
    "ResponseCoefficients",
    "ResponseCoefficientsByPars",
    "SteadyStateStability",
    "parameter_elasticities",
    "rate_characteristics",
    "response_coefficients",
    "steady_state_stability",
    "variable_elasticities",
]


def _unique_times(time_points: ArrayLike) -> Array:
    times = np.asarray(time_points, dtype=float)
    if len(np.unique(times)) != len(times):
        msg = "time_points must not contain duplicates"
        raise ValueError(msg)
    return times


def _outputs[M: _Model](
    model: M,
    runner: SteadyStateRunner[M, Any],
    time_points: Array | None,
) -> pd.Series:
    """Steady-state values, or the values at `time_points` indexed by (time, name)."""
    if time_points is None:
        return _steady_state_worker(model, runner=runner).get_combined().iloc[-1]
    result = _time_course_worker(
        model, time_points=time_points, runner=cast(DynamicRunner[M, Any], runner)
    )
    return cast(
        pd.Series,
        _align_time(result.get_combined(), time_points).stack(),  # noqa: PD013
    )


def _response_coefficient_worker[M: _Model](
    parameter: str,
    *,
    model: M,
    baseline: pd.Series | None,
    runner: SteadyStateRunner[M, Any],
    time_points: Array | None = None,
) -> pd.Series:
    """Response coefficients of a single parameter by central finite differences.

    Perturbs the parameter up and down by the runner's relative
    `displacement` (default 1e-3, absolute for a parameter at 0), computes
    the steady states (or the time courses at `time_points`) of both and
    their difference quotient. Normalized by the unperturbed values
    `baseline` if given, which are computed once by the caller.
    """
    old = model.get_parameter_values()[parameter]
    h = _step(old, float(getattr(runner, "displacement", 1e-3)))

    model.update_parameters({parameter: old + h})
    upper = _outputs(model, runner, time_points)

    model.update_parameters({parameter: old - h})
    lower = _outputs(model, runner, time_points)

    resp = (upper - lower) / (2 * h)
    # Reset
    model.update_parameters({parameter: old})
    if baseline is not None:
        resp *= old / baseline
    return resp


def _forward_response_coefficients(
    model: _Model,
    to_scan: list[str],
    runner: SteadyStateRunner[Any, Any],
    *,
    normalized: bool,
    time_points: Array | None,
) -> pd.DataFrame | None:
    """Response coefficients from the runner's forward sensitivities.

    At steady state one simulation and a linear solve for all parameters
    (implicit function theorem), for time courses one integration of the
    sensitivity equations. None if the runner can't provide forward
    sensitivities, so the caller falls back to parallel finite differences:
    runners without sensitivities or set to finite differences, models that
    can't be differentiated symbolically and failed simulations.
    """
    method = (
        "steady_state_sensitivities"
        if time_points is None
        else "time_course_sensitivities"
    )
    derivatives = getattr(runner, "derivatives", None)
    if (
        not callable(getattr(runner, method, None))
        or derivatives == "finite_difference"
    ):
        return None
    if derivatives == "auto" and dataclasses.is_dataclass(runner):
        # Fail instead of silently using serial finite differences, which the
        # caller computes in parallel
        runner = dataclasses.replace(runner, derivatives="forward")  # type: ignore[type-var]
    try:
        res = (
            cast(Any, runner).steady_state_sensitivities(model, to_scan)
            if time_points is None
            else cast(Any, runner).time_course_sensitivities(
                model, time_points, to_scan
            )
        )
    except Exception as e:  # noqa: BLE001
        _LOGGER.debug("Response coefficients by finite differences, as %s", e)
        return None
    if isinstance(sens := res.value, Exception):
        return None

    if time_points is None:
        values = sens.simulated.iloc[-1]
        resp = pd.DataFrame({k: v.iloc[-1] for k, v in sens.derivatives.items()})
    else:
        values = _align_time(sens.simulated, time_points).stack()  # noqa: PD013
        resp = pd.DataFrame(
            {
                k: _align_time(v, time_points).stack()  # noqa: PD013
                for k, v in sens.derivatives.items()
            }
        )
    if normalized:
        parameters = model.get_parameter_values()
        resp = resp * pd.Series({k: parameters[k] for k in to_scan})
        resp = resp.div(values, axis=0)
    return resp


def _parameters_to_scan(model: _Model, to_scan: list[str] | None) -> list[str]:
    """Default to all parameters not computed from others, reject computed ones."""
    if to_scan is None:
        return _free_parameters(model)
    _check_parameters(model, to_scan)
    return to_scan


def _ordered(model: _Model, combined: pd.DataFrame) -> pd.DataFrame:
    """Rows ordered like `get_combined`, fluxes last (per time point)."""
    names = list(dict.fromkeys(combined.index.get_level_values(-1)))
    variables, fluxes = _split_outputs(model, names)
    if variables is None:
        return combined
    order = variables + (fluxes or [])
    if isinstance(combined.index, pd.MultiIndex):
        times = combined.index.get_level_values(0).unique()
        return combined.reindex(
            pd.MultiIndex.from_product([times, order], names=["time", None])
        )
    return combined.loc[order]


def _split_outputs(
    model: _Model, names: list[str]
) -> tuple[list[str] | None, list[str] | None]:
    """Split output names into variables and fluxes, if the builder has them."""
    if isinstance(model, SteadyStateModelBuilder):
        return None, None
    if isinstance(model, OdeModelBuilder):
        return names, None
    fluxes = set(
        model.get_arg_names(
            include_time=False,
            include_variables=False,
            include_parameters=False,
            include_derived_parameters=False,
            include_derived_variables=False,
            include_reactions=True,
            include_surrogate_variables=False,
            include_surrogate_fluxes=True,
            include_readouts=False,
        )
    )
    return [i for i in names if i not in fluxes], [i for i in names if i in fluxes]


def _select_rows(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Select rows by the last index level, which holds the output names."""
    return df.loc[df.index.get_level_values(-1).isin(names)]


def _require_names(names: list[str] | None, kind: str) -> list[str]:
    if names is None:
        msg = f"These results have no {kind} - use .combined instead"
        raise TypeError(msg)
    return names


###############################################################################
# Non-steady state
###############################################################################


def _fd_variable_elasticities(
    model: KineticModelBuilder,
    *,
    to_scan: list[str],
    variables: dict[str, float],
    time: float,
    normalized: bool,
    displacement: float,
) -> pd.DataFrame:
    """Variable elasticities at one state by central finite differences."""
    elasticities = {}
    for var in to_scan:
        old = variables[var]
        h = _step(old, displacement)

        upper = model.get_fluxes(variables=variables | {var: old + h}, time=time)
        lower = model.get_fluxes(variables=variables | {var: old - h}, time=time)

        elasticity_coef = (upper - lower) / (2 * h)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=variables, time=time)
        elasticities[var] = elasticity_coef

    return pd.DataFrame(data=elasticities)


def _fd_parameter_elasticities(
    model: KineticModelBuilder,
    *,
    to_scan: list[str],
    variables: dict[str, float],
    time: float,
    normalized: bool,
    displacement: float,
) -> pd.DataFrame:
    """Parameter elasticities at one state by central finite differences."""
    elasticities = {}
    for par in to_scan:
        old = model.get_parameter_values()[par]
        h = _step(old, displacement)

        model.update_parameters({par: old + h})
        upper = model.get_fluxes(variables=variables, time=time)

        model.update_parameters({par: old - h})
        lower = model.get_fluxes(variables=variables, time=time)

        # Reset
        model.update_parameters({par: old})
        elasticity_coef = (upper - lower) / (2 * h)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=variables, time=time)
        elasticities[par] = elasticity_coef

    return pd.DataFrame(data=elasticities)


def _exact_elasticities(
    model: KineticModelBuilder,
    *,
    to_scan: list[str],
    of_parameters: bool,
    states: pd.DataFrame,
    fluxes: pd.DataFrame,
    normalized: bool,
    displacement: float,
) -> pd.DataFrame | None:
    """Elasticities along a trajectory from the compiled derivatives.

    `states` and `fluxes` hold the variables and fluxes (time x name). The
    derivatives are total ones through derived quantities, like those of
    the finite differences. None if the model can't be differentiated
    symbolically.
    """
    reactions = list(fluxes.columns)
    var_names = model.get_variable_names()
    try:
        if of_parameters:
            # Parameters also act through initial assignments of other
            # parameters, which `chain` propagates
            setup = _setup(
                model,
                wrt=to_scan,
                outputs=reactions,
                y0_from_model=False,
                displacement=displacement,
            )
            derivatives, chain, columns = setup.derivs, setup.chain, []
        else:
            derivatives = _Derivatives(
                model, compile_derivatives(model, wrt=[], outputs=reactions)
            )
            chain, columns = None, [var_names.index(k) for k in to_scan]
        rows = [derivatives.compiled.outputs.index(k) for k in reactions]
    except Exception as e:  # noqa: BLE001
        _LOGGER.debug("Elasticities by finite differences, as %s", e)
        return None

    frames = {}
    for t, x in zip(
        states.index, states.loc[:, var_names].to_numpy(dtype=float), strict=True
    ):
        gx, gp = derivatives.outputs(float(t), x)
        full = gx[:, columns] if chain is None else gp @ chain
        frames[t] = pd.DataFrame(full[rows], index=reactions, columns=to_scan)
    elasticities = pd.concat(frames, names=["time", None])
    if normalized:
        if of_parameters:
            values = model.get_parameter_values()
            scale = pd.DataFrame(
                [[values[k] for k in to_scan]] * len(elasticities),
                index=elasticities.index,
                columns=to_scan,
            )
        else:
            scale = (
                states.loc[:, to_scan]
                .reindex(elasticities.index.get_level_values(0))
                .set_axis(elasticities.index)
            )
        rates = fluxes.loc[:, reactions].stack()  # noqa: PD013
        elasticities = (elasticities * scale).div(rates, axis=0)
    return elasticities


def _elasticities_over_time(
    model: KineticModelBuilder,
    *,
    to_scan: list[str],
    of_parameters: bool,
    variables: dict[str, float] | None,
    normalized: bool,
    displacement: float,
    time_points: ArrayLike,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None,
) -> pd.DataFrame:
    """Elasticities at every time point of the trajectory starting at `variables`."""
    times = _unique_times(time_points)
    model = _copy_with_y0(model, variables)
    runner = default_runner(model) if runner is None else runner
    result = runner.time_course(model, times).unwrap_or_err()
    states = _align_time(
        result.get_variables(
            include_derived_variables=False,
            include_surrogate_variables=False,
            include_readouts=False,
        ),
        times,
    )
    fluxes = _align_time(result.fluxes, times)

    if (
        getattr(runner, "derivatives", None) != "finite_difference"
        and (
            exact := _exact_elasticities(
                model,
                to_scan=to_scan,
                of_parameters=of_parameters,
                states=states,
                fluxes=fluxes,
                normalized=normalized,
                displacement=displacement,
            )
        )
        is not None
    ):
        return exact

    fd = _fd_parameter_elasticities if of_parameters else _fd_variable_elasticities
    return pd.concat(
        {
            t: fd(
                model,
                to_scan=to_scan,
                variables=cast(dict[str, float], row.to_dict()),
                time=float(cast(float, t)),
                normalized=normalized,
                displacement=displacement,
            )
            for t, row in states.iterrows()
        },
        names=["time", None],
    )


def _check_time_args(
    time: float | None, time_points: ArrayLike | None, runner: object
) -> float:
    if time is not None and time_points is not None:
        msg = "Pass either the time of a single state or time_points of a trajectory"
        raise ValueError(msg)
    if runner is not None and time_points is None:
        msg = "A runner only applies to time_points, a single state isn't simulated"
        raise TypeError(msg)
    return 0.0 if time is None else time


def variable_elasticities(
    model: KineticModelBuilder,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    time: float | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    time_points: ArrayLike | None = None,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> pd.DataFrame:
    """Calculate non-steady state elasticity coefficients.

    Computes the sensitivity of reaction rates to changes in metabolite
    concentrations (ε-elasticities), at one state or along a trajectory.

    Examples
    --------
        >>> variable_elasticities(model, variables={"A": 1.0, "B": 2.0})
        Rxn     A     B
         v1   0.0   0.0
         v2   1.0   0.0
         v3   0.0   5.0

        >>> elas = variable_elasticities(model, time_points=np.linspace(0, 10, 101))
        >>> elas.loc[2.0]               # reactions x variables at t = 2
        >>> elas.xs("v1", level=1)      # time x variables of v1

    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        List of variables to analyze. Uses all if None
    variables
        Custom variable values. Defaults to initial conditions. With
        `time_points`, the initial values of the trajectory
    time
        Time point for evaluation of a single state, defaults to 0
    normalized
        Whether to normalize coefficients
    displacement
        Relative perturbation size of the finite differences (absolute for
        values at 0). The runner's `displacement` doesn't apply
    time_points
        Simulate the time course from `variables` and compute the
        elasticities at every time point. Rows are then indexed by
        (time, reaction). These are exact, from the compiled derivatives of
        the model, unless it can't be differentiated symbolically or the
        runner uses `derivatives="finite_difference"`
    runner
        How the trajectory of `time_points` is simulated, see `mxlpy.runners`.
        `derivatives="finite_difference"` forces finite differences

    Returns
    -------
        DataFrame with elasticity coefficients (reactions x metabolites)

    """
    time = _check_time_args(time, time_points, runner)
    to_scan = model.get_variable_names() if to_scan is None else to_scan
    if time_points is not None:
        return _elasticities_over_time(
            model,
            to_scan=to_scan,
            of_parameters=False,
            variables=variables,
            normalized=normalized,
            displacement=displacement,
            time_points=time_points,
            runner=runner,
        )
    return _fd_variable_elasticities(
        model,
        to_scan=to_scan,
        variables=model.get_initial_conditions() if variables is None else variables,
        time=time,
        normalized=normalized,
        displacement=displacement,
    )


def parameter_elasticities(
    model: KineticModelBuilder,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    time: float | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    time_points: ArrayLike | None = None,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> pd.DataFrame:
    """Calculate parameter elasticity coefficients, at one state or along a trajectory.

    Examples
    --------
        >>> parameter_elasticities(model)
        Rxn    k1    k2
         v1   1.0   0.0
         v2   0.0   1.0
         v3   0.0   0.0

        >>> elas = parameter_elasticities(model, time_points=np.linspace(0, 10, 101))
        >>> elas.xs("v1", level=1)      # time x parameters of v1

    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        List of parameters to analyze. Defaults to all parameters except
        those set by initial assignments, which can't be perturbed (a
        ValueError names the parameters they are computed from)
    variables
        Custom variable values. Defaults to initial conditions. With
        `time_points`, the initial values of the trajectory
    time
        Time point for evaluation of a single state, defaults to 0
    normalized
        Whether to normalize coefficients
    displacement
        Relative perturbation size of the finite differences (absolute for
        values at 0), also used for the (constant) derivatives of initial
        assignments with `time_points`. The runner's `displacement` doesn't apply
    time_points
        Simulate the time course from `variables` and compute the
        elasticities at every time point. Rows are then indexed by
        (time, reaction). These are exact, from the compiled derivatives of
        the model (effects through initial assignments of other parameters
        by the chain rule), unless it can't be differentiated symbolically
        or the runner uses `derivatives="finite_difference"`
    runner
        How the trajectory of `time_points` is simulated, see `mxlpy.runners`.
        `derivatives="finite_difference"` forces finite differences

    Returns
    -------
        DataFrame with parameter elasticities (reactions x parameters)

    """
    time = _check_time_args(time, time_points, runner)
    to_scan = _parameters_to_scan(model, to_scan)
    if time_points is not None:
        return _elasticities_over_time(
            model,
            to_scan=to_scan,
            of_parameters=True,
            variables=variables,
            normalized=normalized,
            displacement=displacement,
            time_points=time_points,
            runner=runner,
        )
    return _fd_parameter_elasticities(
        model,
        to_scan=to_scan,
        variables=model.get_initial_conditions() if variables is None else variables,
        time=time,
        normalized=normalized,
        displacement=displacement,
    )


@dataclass(kw_only=True, slots=True)
class ResponseCoefficients(Generic[SE]):
    """Container for response coefficients.

    `combined` holds the response of every model output (rows) to every
    scanned parameter (columns). For time courses the rows are indexed by
    (time, output). The split into `variables` and `fluxes` only exists for
    builders having them.
    """

    combined: pd.DataFrame
    variable_names: list[str] | None = None
    flux_names: list[str] | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(
        self: ResponseCoefficients[Simulation] | ResponseCoefficients[OdeSimulation],
    ) -> pd.DataFrame:
        """Return the response coefficients of the variables."""
        return _select_rows(
            self.combined, _require_names(self.variable_names, "variables")
        )

    @property
    def fluxes(self: ResponseCoefficients[Simulation]) -> pd.DataFrame:
        """Return the response coefficients of the fluxes."""
        return _select_rows(self.combined, _require_names(self.flux_names, "fluxes"))

    def __iter__(self: ResponseCoefficients[Simulation]) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))

    def for_parameter(self, name: str) -> pd.DataFrame | pd.Series:
        """Response of every output to one parameter.

        A time x output DataFrame for time courses, e.g. for `plot.lines`,
        else a Series indexed by output.

        Examples
        --------
            >>> rc = mca.response_coefficients(model, time_points=np.linspace(0, 10, 101))
            >>> plot.lines(rc.for_parameter("k1"))

        """
        column = self.combined[name]
        if not isinstance(column.index, pd.MultiIndex):
            return column
        order = list(dict.fromkeys(column.index.get_level_values(-1)))
        return column.unstack().loc[:, order]


# ###############################################################################
# # Steady state
# ###############################################################################


@overload
def response_coefficients(
    model: KineticModelBuilder,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    normalized: bool = True,
    time_points: ArrayLike | None = None,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> ResponseCoefficients[Simulation]: ...


@overload
def response_coefficients(
    model: OdeModelBuilder,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    normalized: bool = True,
    time_points: ArrayLike | None = None,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
) -> ResponseCoefficients[OdeSimulation]: ...


@overload
def response_coefficients(
    model: SteadyStateModelBuilder,
    *,
    to_scan: list[str] | None = None,
    normalized: bool = True,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
) -> ResponseCoefficients[Evaluation]: ...


def response_coefficients(
    model: _Model,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    normalized: bool = True,
    time_points: ArrayLike | None = None,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
) -> (
    ResponseCoefficients[Simulation]
    | ResponseCoefficients[OdeSimulation]
    | ResponseCoefficients[Evaluation]
):
    """Calculate response coefficients of the steady state or of a time course.

    The response coefficient of an output y to a parameter p is dy/dp,
    normalized to (dy/dp) * (p / y). With `time_points`, these are the
    time-dependent response coefficients of the trajectory starting from
    `variables`, i.e. how every output at every time point responds to a
    change of the parameter (Ingalls & Sauro 2003). Normalized coefficients
    are undefined where an output is 0, e.g. variables starting at 0.

    Examples
    --------
        >>> response_coefficients(model, to_scan=["k1", "k2"]).variables
        p    x1    x2
        k1  1.4  1.31
        k2 -1.0 -2.49

        >>> rc = response_coefficients(model, time_points=np.linspace(0, 10, 101))
        >>> rc.variables.loc[2.0]       # all variables at t = 2
        >>> rc.for_parameter("k1")      # time x outputs

    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        Parameters to analyze. Defaults to all parameters except those set
        by initial assignments, which can't be perturbed (a ValueError names
        the parameters they are computed from)
    variables
        Custom variable values. Defaults to initial conditions.
    normalized
        Whether to normalize coefficients
    time_points
        Time points of a time course to analyse instead of the steady state.
        Rows of the result are then indexed by (time, output)
    disable_tqdm
        Disable progress bar
    parallel
        Whether to parallelize the computation
    max_workers
        Maximum number of workers
    runner
        How the model is simulated and differentiated, see `mxlpy.runners`.
        Defaults to the model's default runner. Runners providing
        sensitivities (`KineticRunner`, `OdeRunner`) compute all coefficients
        from a single steady state with the implicit function theorem (or a
        single integration of the sensitivity equations for time courses),
        unless `derivatives="finite_difference"` or the model can't be
        differentiated symbolically. Otherwise every parameter is perturbed
        by the runner's relative `displacement` (central finite differences
        of two simulations each, in parallel)

    Returns
    -------
        ResponseCoefficients object containing:
        - Flux response coefficients
        - Concentration response coefficients

    """
    return _response_coefficients(
        model,
        to_scan=to_scan,
        variables=variables,
        normalized=normalized,
        time_points=time_points,
        disable_tqdm=disable_tqdm,
        parallel=parallel,
        max_workers=max_workers,
        runner=runner,
    )


def _response_coefficients(
    model: _Model,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    normalized: bool = True,
    time_points: ArrayLike | None = None,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
) -> (
    ResponseCoefficients[Simulation]
    | ResponseCoefficients[OdeSimulation]
    | ResponseCoefficients[Evaluation]
):
    """Non-overloaded implementation of `response_coefficients`, for internal callers."""
    if time_points is not None and isinstance(model, SteadyStateModelBuilder):
        msg = "A SteadyStateModelBuilder has no dynamics, so time_points don't apply"
        raise TypeError(msg)
    times = None if time_points is None else _unique_times(time_points)
    to_scan = _parameters_to_scan(model, to_scan)
    model = _copy_with_y0(model, variables)
    runner = default_runner(model) if runner is None else runner
    if times is not None and not callable(getattr(runner, "time_course", None)):
        msg = "time_points need a runner simulating time courses, see `mxlpy.runners`"
        raise TypeError(msg)

    combined = _forward_response_coefficients(
        model, to_scan, runner, normalized=normalized, time_points=times
    )
    if combined is None:
        precompile(runner, model)
        res = parallelise(
            partial(
                _response_coefficient_worker,
                model=model,
                baseline=_outputs(model, runner, times) if normalized else None,
                runner=runner,
                time_points=times,
            ),
            inputs=list(zip(to_scan, to_scan, strict=True)),
            cache=None,
            disable_tqdm=disable_tqdm,
            parallel=parallel,
            max_workers=max_workers,
        )
        combined = pd.DataFrame(dict(res))
    combined = _ordered(model, combined)
    names = list(dict.fromkeys(combined.index.get_level_values(-1)))
    variable_names, flux_names = _split_outputs(model, names)
    return ResponseCoefficients(
        combined=combined,
        variable_names=variable_names,
        flux_names=flux_names,
    )


@dataclass(kw_only=True, slots=True)
class ResponseCoefficientsByPars(Generic[SE]):
    """Container for response coefficients.

    `combined` holds the response of every model output (rows) to every
    scanned parameter (columns). The split into `variables` and `fluxes`
    only exists for builders having them.
    """

    combined: pd.DataFrame
    parameters: pd.DataFrame
    variable_names: list[str] | None = None
    flux_names: list[str] | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(
        self: ResponseCoefficientsByPars[Simulation]
        | ResponseCoefficientsByPars[OdeSimulation],
    ) -> pd.DataFrame:
        """Return the response coefficients of the variables."""
        return _select_rows(
            self.combined, _require_names(self.variable_names, "variables")
        )

    @property
    def fluxes(self: ResponseCoefficientsByPars[Simulation]) -> pd.DataFrame:
        """Return the response coefficients of the fluxes."""
        return _select_rows(self.combined, _require_names(self.flux_names, "fluxes"))

    def __iter__(
        self: ResponseCoefficientsByPars[Simulation],
    ) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


###############################################################################
# Stability analysis
###############################################################################


@dataclass(kw_only=True, slots=True)
class SteadyStateStability:
    """Local stability classification of a steady state."""

    eigenvalues: NDArray[np.complexfloating[Any, Any]]
    spectral_abscissa: float
    is_stable: bool
    has_oscillatory: bool
    classification: str

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


def steady_state_stability(
    model: KineticModelBuilder,
    steady_state: dict[str, float],
    *,
    threshold: float = 1e-10,
) -> SteadyStateStability:
    """Classify local stability of a steady state via Jacobian eigenvalues.

    Parameters
    ----------
    model
        Model instance.
    steady_state
        Steady-state variable concentrations {name: value}.
    threshold
        Tolerance for treating real/imaginary parts as zero.

    Returns
    -------
    SteadyStateStability
        Eigenvalues, spectral abscissa, stability flag, oscillatory flag,
        and a human-readable classification label.

    Examples
    --------
        >>> from mxlpy import Simulator
        >>> ss = Simulator(model).simulate_to_steady_state().get_new_y0()
        >>> steady_state_stability(model, ss).classification
        'stable node'

    """
    sym = to_symbolic_model(model)
    J_sym = sym.jacobian()

    subs: dict = {s: steady_state[name] for name, s in sym.variables.items()}
    subs |= {s: sym.parameter_values[name] for name, s in sym.parameters.items()}
    J_num = np.array(J_sym.subs(subs).tolist(), dtype=complex)
    eigenvalues = np.linalg.eigvals(J_num)

    real_parts = eigenvalues.real
    spectral_abscissa = float(real_parts.max())
    min_real = float(real_parts.min())
    has_oscillatory = bool(np.any(np.abs(eigenvalues.imag) > threshold))
    is_stable = spectral_abscissa < -threshold

    if spectral_abscissa < -threshold:
        classification = "stable spiral" if has_oscillatory else "stable node"
    elif min_real > threshold:
        classification = "unstable spiral" if has_oscillatory else "unstable node"
    elif spectral_abscissa > threshold and min_real < -threshold:
        classification = "saddle point"
    elif np.all(np.abs(real_parts) <= threshold) and has_oscillatory:
        classification = "centre"
    else:
        classification = "non-hyperbolic"

    return SteadyStateStability(
        eigenvalues=eigenvalues,  # pyright: ignore[reportArgumentType]
        spectral_abscissa=spectral_abscissa,
        is_stable=is_stable,
        has_oscillatory=has_oscillatory,
        classification=classification,
    )


###############################################################################
# Supply-demand rate characteristics
###############################################################################


@dataclass(kw_only=True, slots=True)
class RateCharacteristics:
    """Supply and demand rate characteristics for a single variable.

    Holds the supply and demand fluxes obtained by clamping a variable to a
    range of concentrations and re-computing the steady state at each point.
    The original steady-state concentration is retained so it can be marked on
    the characteristic plots.

    Attributes
    ----------
    supply_fluxes
        Fluxes of reactions that produce the variable, indexed by the scanned
        concentration. One column per supply reaction; empty if the variable is
        never produced.
    demand_fluxes
        Fluxes of reactions that consume the variable, indexed by the scanned
        concentration. One column per demand reaction; empty if the variable is
        never consumed.
    steady_state_conc
        Concentration of the variable at the original steady state.

    """

    supply_fluxes: pd.DataFrame
    demand_fluxes: pd.DataFrame
    steady_state_conc: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def total_supply(self) -> pd.Series:
        """Total supply flux (sum over all supply reactions)."""
        return self.supply_fluxes.sum(axis=1)

    @property
    def total_demand(self) -> pd.Series:
        """Total demand flux (sum over all demand reactions)."""
        return self.demand_fluxes.sum(axis=1)

    def plot_supply_demand(
        self,
        *,
        ax: Axes | None = None,
        grid: bool = True,
    ) -> FigAx:
        """Plot total supply and demand fluxes on a single log-log axis.

        Parameters
        ----------
        ax
            Axis to plot on. Created if None.
        grid
            Whether to add a grid.

        Returns
        -------
        FigAx
            Figure and Axes objects.

        """
        fig, ax = _default_fig_ax(ax=ax, grid=grid)
        ax.plot(self.total_supply.index, self.total_supply, label="supply")
        ax.plot(self.total_demand.index, self.total_demand, label="demand")
        ax.axvline(self.steady_state_conc, color="black", linestyle="--", alpha=0.5)
        ax.set(xscale="log", yscale="log", xlabel="concentration", ylabel="flux")
        ax.legend()
        return fig, ax

    def plot(
        self,
        *,
        figsize: tuple[float, float] | None = (12.0, 4.0),
        grid: bool = True,
    ) -> FigAxs:
        """Plot a multi-panel summary of the rate characteristics.

        The figure has three log-log panels: total supply versus total demand,
        the per-reaction supply fluxes, and the per-reaction demand fluxes. The
        original steady-state concentration is marked on each panel.

        Parameters
        ----------
        figsize
            Size of the figure.
        grid
            Whether to add a grid to each panel.

        Returns
        -------
        FigAxs
            Figure and Axes objects.

        """
        fig, axs = _default_fig_axs(
            ncols=3,
            nrows=1,
            figsize=figsize,
            grid=grid,
            sharex=True,
            sharey=False,
        )
        ax_total, ax_supply, ax_demand = axs

        self.plot_supply_demand(ax=ax_total, grid=grid)
        ax_total.set_title("Supply vs demand")

        for ax, fluxes, title in (
            (ax_supply, self.supply_fluxes, "Supply reactions"),
            (ax_demand, self.demand_fluxes, "Demand reactions"),
        ):
            for name in fluxes.columns:
                ax.plot(fluxes.index, fluxes[name], label=name)
            ax.axvline(self.steady_state_conc, color="black", linestyle="--", alpha=0.5)
            ax.set(xscale="log", yscale="log", xlabel="concentration", ylabel="flux")
            ax.set_title(title)
            if len(fluxes.columns) > 0:
                ax.legend()

        return fig, axs


def rate_characteristics(
    model: KineticModelBuilder,
    variable: str,
    *,
    min_factor: float = 100.0,
    max_factor: float = 100.0,
    n_points: int = 256,
    parallel: bool = True,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
) -> RateCharacteristics:
    """Compute supply-demand rate characteristics for a variable.

    The chosen variable is clamped to a logarithmically spaced range of
    concentrations around its steady-state value. At each concentration the
    steady state of the remaining variables is recomputed and the supply
    (producing) and demand (consuming) reaction fluxes are collected. Plotting
    these against the clamped concentration places the steady state in the
    context of its full operating curve.

    Reactions are classified structurally from the stoichiometry evaluated at
    the original steady state: a positive coefficient marks a supply reaction,
    a negative one a demand reaction.

    Parameters
    ----------
    model
        Metabolic model instance.
    variable
        Name of the variable to scan.
    min_factor
        Lower bound of the scan, as ``steady_state_conc / min_factor``.
    max_factor
        Upper bound of the scan, as ``steady_state_conc * max_factor``.
    n_points
        Number of concentrations to scan (log-spaced).
    parallel
        Whether to compute the steady states in parallel.
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    Returns
    -------
    RateCharacteristics
        Supply and demand fluxes and the original steady-state concentration.

    Raises
    ------
    ValueError
        If the variable does not participate in any reaction.

    Examples
    --------
        >>> rc = rate_characteristics(model, "ATP")
        >>> rc.plot()

    """
    runner = KineticRunner() if runner is None else runner
    ss = runner.steady_state(model).unwrap_or_err().get_new_y0()
    ss_conc = ss[variable]

    try:
        stoich = model.get_stoichiometries_of_variable(variable, variables=ss)
    except KeyError:
        stoich = {}
    supply_rxns = [rxn for rxn, coef in stoich.items() if coef > 0]
    demand_rxns = [rxn for rxn, coef in stoich.items() if coef < 0]
    if not supply_rxns and not demand_rxns:
        msg = f"Variable {variable!r} does not participate in any reaction"
        raise ValueError(msg)

    clamped = copy.deepcopy(model)
    clamped.make_variable_static(variable)

    concs = np.geomspace(ss_conc / min_factor, ss_conc * max_factor, n_points)
    res = steady_state(
        clamped,
        to_scan=pd.DataFrame({variable: concs}),
        parallel=parallel,
        runner=runner,
    )
    fluxes = res.fluxes

    return RateCharacteristics(
        supply_fluxes=fluxes.loc[:, supply_rxns],
        demand_fluxes=fluxes.loc[:, demand_rxns],
        steady_state_conc=ss_conc,
    )
