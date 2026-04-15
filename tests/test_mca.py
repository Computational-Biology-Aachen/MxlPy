"""Tests for the MCA module."""

from __future__ import annotations

import numpy as np
import pytest

from mxlpy import Model, fns
from mxlpy.mca import (
    parameter_elasticities,
    response_coefficients,
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


def test_variable_elasticities_returns_dataframe() -> None:
    result = variable_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"A", "B"}
    assert set(result.index) == {"v_in", "v1", "v2"}


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


def test_variable_elasticities_constant_reaction_is_zero() -> None:
    """Constant influx v_in does not depend on any variable."""
    result = variable_elasticities(linear_chain(), normalized=False)
    assert result.loc["v_in", "A"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc["v_in", "B"] == pytest.approx(0.0, abs=1e-6)


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
# parameter_elasticities
###############################################################################


def test_parameter_elasticities_returns_dataframe() -> None:
    result = parameter_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"k1", "k2", "k_in"}
    assert set(result.index) == {"v_in", "v1", "v2"}


def test_parameter_elasticities_mass_action_normalized_equals_one() -> None:
    """For v = k*S, normalized elasticity w.r.t. k is exactly 1."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v1", "k1"] == pytest.approx(1.0, rel=1e-3)
    assert result.loc["v2", "k2"] == pytest.approx(1.0, rel=1e-3)


def test_parameter_elasticities_cross_terms_are_zero() -> None:
    """k1 does not appear in v2; elasticity must be zero."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v2", "k1"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc["v1", "k2"] == pytest.approx(0.0, abs=1e-4)


###############################################################################
# response_coefficients
###############################################################################


def test_response_coefficients_returns_named_result() -> None:
    result = response_coefficients(linear_chain(), parallel=False)
    assert result.variables is not None
    assert result.fluxes is not None
    assert set(result.variables.columns) == {"k1", "k2", "k_in"}


def test_response_coefficients_finite() -> None:
    """All response coefficients must be finite for a well-behaved model."""
    result = response_coefficients(linear_chain(), parallel=False)
    assert np.isfinite(result.variables.values).all()
    assert np.isfinite(result.fluxes.values).all()


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
