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
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy.parallel import parallelise
from mxlpy.plot import _default_fig_ax, _default_fig_axs
from mxlpy.scan import _steady_state_worker, steady_state
from mxlpy.simulator import Simulator
from mxlpy.symbolic.symbolic_model import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from mxlpy.integrators import IntegratorType
    from mxlpy.model import Model
    from mxlpy.plot import FigAx, FigAxs

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


def _response_coefficient_worker(
    parameter: str,
    *,
    model: Model,
    y0: dict[str, float] | None,
    normalized: bool,
    rel_norm: bool,
    displacement: float = 1e-4,
    integrator: IntegratorType | None,
) -> tuple[pd.Series, pd.Series]:
    """Calculate response coefficients for a single parameter.

    Internal helper function that computes concentration and flux response
    coefficients using finite differences. The function:
    1. Perturbs the parameter up and down by a small displacement
    2. Calculates steady states for each perturbation
    3. Computes response coefficients from the differences
    4. Optionally normalizes the results

    Parameters
    ----------
    parameter
        Name of the parameter to analyze
    model
        Metabolic model instance
    y0
        Initial conditions as a dictionary {species: value}
    normalized
        Whether to normalize the coefficients
    rel_norm
        Whether to use relative normalization
    displacement
        Relative perturbation size (default: 1e-4)
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Tuple containing:
        - Series of concentration response coefficients
        - Series of flux response coefficients

    """
    old = model.get_parameter_values()[parameter]
    if y0 is not None:
        model.update_variables(y0)

    model.update_parameters({parameter: old * (1 + displacement)})
    upper = _steady_state_worker(
        model,
        rel_norm=rel_norm,
        integrator=integrator,
        y0=None,
    )

    model.update_parameters({parameter: old * (1 - displacement)})
    lower = _steady_state_worker(
        model,
        rel_norm=rel_norm,
        integrator=integrator,
        y0=None,
    )

    conc_resp = (upper.variables.iloc[-1] - lower.variables.iloc[-1]) / (
        2 * displacement * old
    )  # pyright: ignore[reportOperatorIssue]
    flux_resp = (upper.fluxes.iloc[-1] - lower.fluxes.iloc[-1]) / (
        2 * displacement * old
    )  # pyright: ignore[reportOperatorIssue]
    # Reset
    model.update_parameters({parameter: old})
    if normalized:
        norm = _steady_state_worker(
            model,
            rel_norm=rel_norm,
            integrator=integrator,
            y0=None,
        )
        conc_resp *= old / norm.variables.iloc[-1]
        flux_resp *= old / norm.fluxes.iloc[-1]
    return conc_resp, flux_resp


###############################################################################
# Non-steady state
###############################################################################


def variable_elasticities(
    model: Model,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    time: float = 0,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate non-steady state elasticity coefficients.

    Computes the sensitivity of reaction rates to changes in metabolite
    concentrations (ε-elasticities).

    Examples
    --------
        >>> variable_elasticities(model, concs={"A": 1.0, "B": 2.0})
        Rxn     A     B
         v1   0.0   0.0
         v2   1.0   0.0
         v3   0.0   5.0


    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        List of variables to analyze. Uses all if None
    variables
        Custom variable values. Defaults to initial conditions.
    time
        Time point for evaluation
    normalized
        Whether to normalize coefficients
    displacement
        Relative perturbation size

    Returns
    -------
        DataFrame with elasticity coefficients (reactions x metabolites)

    """
    variables = model.get_initial_conditions() if variables is None else variables
    to_scan = model.get_variable_names() if to_scan is None else to_scan
    elasticities = {}

    for var in to_scan:
        old = variables[var]

        upper = model.get_fluxes(
            variables=variables | {var: old * (1 + displacement)}, time=time
        )
        lower = model.get_fluxes(
            variables=variables | {var: old * (1 - displacement)}, time=time
        )

        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=variables, time=time)
        elasticities[var] = elasticity_coef

    return pd.DataFrame(data=elasticities)


def parameter_elasticities(
    model: Model,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    time: float = 0,
    normalized: bool = True,
    displacement: float = 1e-4,
) -> pd.DataFrame:
    """Calculate parameter elasticity coefficients.

    Examples
    --------
        >>> parameter_elasticities(model)
        Rxn    k1    k2
         v1   1.0   0.0
         v2   0.0   1.0
         v3   0.0   0.0

    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        List of parameters to analyze. Uses all if None
    variables
        Custom variable values. Defaults to initial conditions.
    time
        Time point for evaluation
    normalized
        Whether to normalize coefficients
    displacement
        Relative perturbation size

    Returns
    -------
        DataFrame with parameter elasticities (reactions x parameters)

    """
    variables = model.get_initial_conditions() if variables is None else variables
    to_scan = model.get_parameter_names() if to_scan is None else to_scan

    elasticities = {}

    variables = model.get_initial_conditions() if variables is None else variables
    for par in to_scan:
        old = model.get_parameter_values()[par]

        model.update_parameters({par: old * (1 + displacement)})
        upper = model.get_fluxes(variables=variables, time=time)

        model.update_parameters({par: old * (1 - displacement)})
        lower = model.get_fluxes(variables=variables, time=time)

        # Reset
        model.update_parameters({par: old})
        elasticity_coef = (upper - lower) / (2 * displacement * old)
        if normalized:
            elasticity_coef *= old / model.get_fluxes(variables=variables, time=time)
        elasticities[par] = elasticity_coef

    return pd.DataFrame(data=elasticities)


@dataclass(kw_only=True, slots=True)
class ResponseCoefficients:
    """Container for response coefficients."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


# ###############################################################################
# # Steady state
# ###############################################################################


def response_coefficients(
    model: Model,
    *,
    to_scan: list[str] | None = None,
    variables: dict[str, float] | None = None,
    normalized: bool = True,
    displacement: float = 1e-4,
    disable_tqdm: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
    rel_norm: bool = False,
    integrator: IntegratorType | None = None,
) -> ResponseCoefficients:
    """Calculate response coefficients.

    Examples
    --------
        >>> response_coefficients(model, parameters=["k1", "k2"]).variables
        p    x1    x2
        k1  1.4  1.31
        k2 -1.0 -2.49

    Parameters
    ----------
    model
        Metabolic model instance
    to_scan
        Parameters to analyze. Uses all if None
    variables
        Custom variable values. Defaults to initial conditions.
    normalized
        Whether to normalize coefficients
    displacement
        Relative perturbation size
    disable_tqdm
        Disable progress bar
    parallel
        Whether to parallelize the computation
    max_workers
        Maximum number of workers
    rel_norm
        Whether to use relative normalization
    integrator
        Integrator function to use for steady state calculation

    Returns
    -------
        ResponseCoefficients object containing:
        - Flux response coefficients
        - Concentration response coefficients

    """
    to_scan = model.get_parameter_names() if to_scan is None else to_scan

    res = parallelise(
        partial(
            _response_coefficient_worker,
            model=model,
            y0=variables,
            normalized=normalized,
            displacement=displacement,
            rel_norm=rel_norm,
            integrator=integrator,
        ),
        inputs=list(zip(to_scan, to_scan, strict=True)),
        cache=None,
        disable_tqdm=disable_tqdm,
        parallel=parallel,
        max_workers=max_workers,
    )
    return ResponseCoefficients(
        variables=pd.DataFrame({k: v[0] for k, v in res}),
        fluxes=pd.DataFrame({k: v[1] for k, v in res}),
    )


@dataclass(kw_only=True, slots=True)
class ResponseCoefficientsByPars:
    """Container for response coefficients by parameter."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def __iter__(self) -> Iterator[pd.DataFrame]:
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
    model: Model,
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
    model: Model,
    variable: str,
    *,
    min_factor: float = 100.0,
    max_factor: float = 100.0,
    n_points: int = 256,
    parallel: bool = True,
    integrator: IntegratorType | None = None,
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
    integrator
        Integrator to use for steady-state calculation.

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
    ss = (
        Simulator(model, integrator=integrator)
        .simulate_to_steady_state()
        .get_result()
        .unwrap_or_err()
        .get_new_y0()
    )
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
        integrator=integrator,
    )
    fluxes = res.fluxes

    return RateCharacteristics(
        supply_fluxes=fluxes.loc[:, supply_rxns],
        demand_fluxes=fluxes.loc[:, demand_rxns],
        steady_state_conc=ss_conc,
    )
