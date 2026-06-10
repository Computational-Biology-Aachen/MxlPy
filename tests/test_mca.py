"""Tests for the MCA module."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mxlpy import Model, Simulator, fns
from mxlpy.mca import (
    RateCharacteristics,
    SteadyStateStability,
    parameter_elasticities,
    rate_characteristics,
    response_coefficients,
    steady_state_stability,
    variable_elasticities,
)


def linear_chain() -> Model:
    """A -> B -> C open linear chain with mass-action kinetics."""
    return (
        Model()
        .add_variables({"A": 5.0, "B": 2.0})
        .add_parameters({"k1": 1.0, "k2": 0.5, "k_in": 2.0})
        .add_reaction(
            "v_in",
            fn=fns.constant,
            args=["k_in"],
            stoichiometry={"A": 1.0},
        )
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["A", "k1"],
            stoichiometry={"A": -1.0, "B": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["B", "k2"],
            stoichiometry={"B": -1.0},
        )
    )


###############################################################################
# variable_elasticities
###############################################################################


def test_parameter_elasticities() -> None:
    # FIXME: implement this
    assert True


def test_parameter_elasticities_cross_terms_are_zero() -> None:
    """k1 does not appear in v2; elasticity must be zero."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v2", "k1"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc["v1", "k2"] == pytest.approx(0.0, abs=1e-4)


def test_parameter_elasticities_mass_action_normalized_equals_one() -> None:
    """For v = k*S, normalized elasticity w.r.t. k is exactly 1."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v1", "k1"] == pytest.approx(1.0, rel=1e-3)
    assert result.loc["v2", "k2"] == pytest.approx(1.0, rel=1e-3)


def test_parameter_elasticities_returns_dataframe() -> None:
    result = parameter_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"k1", "k2", "k_in"}
    assert set(result.index) == {"v_in", "v1", "v2"}


def test_response_coefficients() -> None:
    # FIXME: implement this
    assert True


def test_response_coefficients_finite() -> None:
    """All response coefficients must be finite for a well-behaved model."""
    result = response_coefficients(linear_chain(), parallel=False)
    assert np.isfinite(result.variables.values).all()
    assert np.isfinite(result.fluxes.values).all()


def test_response_coefficients_returns_named_result() -> None:
    result = response_coefficients(linear_chain(), parallel=False)
    assert result.variables is not None
    assert result.fluxes is not None
    assert set(result.variables.columns) == {"k1", "k2", "k_in"}


def test_response_coefficients_zero_parameter() -> None:
    """Parameter value of 0.0 causes division by zero in the finite-difference.

    ``_response_coefficient_worker`` divides by ``2 * displacement * old``
    where ``old = 0.0``.  For pandas Series this silently yields NaN rather
    than raising.  This test documents the current behavior.
    """
    model = (
        Model()
        .add_variable("S", 1.0)
        .add_parameters({"k_active": 1.0, "k_zero": 0.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k_active"],
            stoichiometry={"S": -1.0},
        )
    )
    result = response_coefficients(model, to_scan=["k_zero"], parallel=False)
    col = result.variables["k_zero"]
    # k_zero = 0 → displacement = 0 → NaN, not a crash
    assert col.isnull().any() or not np.isfinite(col.to_numpy()).all()


def test_responsecoefficients_combined() -> None:
    # FIXME: implement this
    assert True


def test_responsecoefficientsbypars_combined() -> None:
    # FIXME: implement this
    assert True


def test_variable_elasticities() -> None:
    # FIXME: implement this
    assert True


def test_variable_elasticities_constant_reaction_is_zero() -> None:
    """Constant influx v_in does not depend on any variable."""
    result = variable_elasticities(linear_chain(), normalized=False)
    assert result.loc["v_in", "A"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc["v_in", "B"] == pytest.approx(0.0, abs=1e-6)


def test_variable_elasticities_mass_action_equals_k_unnormalized() -> None:
    """For mass-action v = k*S, unnormalized elasticity ≈ k (the derivative)."""
    model = linear_chain()
    result = variable_elasticities(model, normalized=False)
    k1 = model.get_parameter_values()["k1"]
    assert result.loc["v1", "A"] == pytest.approx(k1, rel=1e-3)
    assert result.loc["v2", "B"] == pytest.approx(
        model.get_parameter_values()["k2"], rel=1e-3
    )


def test_variable_elasticities_normalized_mass_action_equals_one() -> None:
    """For mass-action v = k*S, normalized elasticity = 1."""
    result = variable_elasticities(linear_chain(), normalized=True)
    assert result.loc["v1", "A"] == pytest.approx(1.0, rel=1e-3)
    assert result.loc["v2", "B"] == pytest.approx(1.0, rel=1e-3)


def test_variable_elasticities_returns_dataframe() -> None:
    result = variable_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"A", "B"}
    assert set(result.index) == {"v_in", "v1", "v2"}


def test_variable_elasticities_zero_variable_produces_nan_or_inf() -> None:
    """Variable=0.0 → displacement=0 → division by zero → inf/NaN.

    The function computes ``(upper - lower) / (2 * displacement * old)`` where
    ``old = 0.0``.  For pandas Series this yields inf or NaN rather than raising.
    This test documents the current behavior.
    """
    model = (
        Model()
        .add_variable("S", 0.0)  # zero initial condition
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k"],
            stoichiometry={"S": -1.0},
        )
    )
    result = variable_elasticities(model, normalized=False)
    # Division by zero → inf or NaN in the S column
    assert not np.isfinite(result["S"].to_numpy()).all()


###############################################################################
# steady_state_stability
###############################################################################


def test_steady_state_stability_returns_correct_type() -> None:
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    assert isinstance(result, SteadyStateStability)
    assert isinstance(result.eigenvalues, np.ndarray)
    assert isinstance(result.spectral_abscissa, float)
    assert isinstance(result.is_stable, bool)
    assert isinstance(result.has_oscillatory, bool)
    assert isinstance(result.classification, str)


def test_steady_state_stability_linear_chain_is_stable() -> None:
    """Open linear chain A→B→sink is unconditionally stable."""
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    assert result.is_stable
    assert result.spectral_abscissa < 0
    assert result.classification in {"stable node", "stable spiral"}


def test_steady_state_stability_unstable_model() -> None:
    """Autocatalytic model dS/dt = k*S has positive eigenvalue → unstable."""
    model = (
        Model()
        .add_variable("S", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k"],
            stoichiometry={"S": 1.0},
        )
    )
    result = steady_state_stability(model, {"S": 1.0})
    assert not result.is_stable
    assert result.spectral_abscissa > 0
    assert result.classification in {"unstable node", "unstable spiral", "saddle point"}


def test_steady_state_stability_spectral_abscissa_matches_eigenvalues() -> None:
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    max_real = max(float(ev.real) for ev in result.eigenvalues)
    assert result.spectral_abscissa == pytest.approx(max_real)


###############################################################################
# rate_characteristics
###############################################################################


def test_rate_characteristics_returns_result_type() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert isinstance(result, RateCharacteristics)
    assert isinstance(result.steady_state_conc, float)


def test_rate_characteristics_supply_demand_columns() -> None:
    """B is produced by v1 (supply) and consumed by v2 (demand)."""
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert list(result.supply_fluxes.columns) == ["v1"]
    assert list(result.demand_fluxes.columns) == ["v2"]


def test_rate_characteristics_n_points() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=32, parallel=False)
    assert len(result.supply_fluxes) == 32
    assert len(result.demand_fluxes) == 32


def test_rate_characteristics_steady_state_conc() -> None:
    """At steady state k1*A = v_in and k2*B = k1*A, so A=2, B=4."""
    model = linear_chain()
    ss = (
        Simulator(model)
        .simulate_to_steady_state()
        .get_result()
        .unwrap_or_err()
        .get_new_y0()
    )
    result = rate_characteristics(model, "B", n_points=16, parallel=False)
    assert result.steady_state_conc == pytest.approx(ss["B"])
    assert result.steady_state_conc == pytest.approx(4.0, rel=1e-3)


def test_rate_characteristics_scan_bounds() -> None:
    """Scan spans [ss / min_factor, ss * max_factor] on a log scale."""
    result = rate_characteristics(
        linear_chain(),
        "B",
        min_factor=10.0,
        max_factor=100.0,
        n_points=16,
        parallel=False,
    )
    concs = result.demand_fluxes.index.to_numpy()
    ss = result.steady_state_conc
    assert concs[0] == pytest.approx(ss / 10.0)
    assert concs[-1] == pytest.approx(ss * 100.0)


def test_rate_characteristics_demand_increases_with_concentration() -> None:
    """Demand flux v2 = k2 * B is monotonically increasing in B."""
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    demand = result.total_demand.to_numpy()
    assert np.all(np.diff(demand) > 0)


def test_rate_characteristics_totals_are_series() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert result.total_supply.shape == (16,)
    assert result.total_demand.shape == (16,)


def test_rate_characteristics_no_reaction_raises() -> None:
    """A variable that participates in no reaction is rejected."""
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.constant,
            args=["k"],
            stoichiometry={},
        )
    )
    with pytest.raises(ValueError, match="does not participate"):
        rate_characteristics(model, "x", n_points=4, parallel=False)


def test_rate_characteristics_plot_supply_demand() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    fig, ax = result.plot_supply_demand()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_rate_characteristics_plot() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    fig, axs = result.plot()
    assert isinstance(fig, Figure)
    assert len(axs) == 3
    plt.close(fig)
