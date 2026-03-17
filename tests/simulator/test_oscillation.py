"""Tests for oscillation detection in steady-state simulations."""

from __future__ import annotations

import numpy as np
import pytest

from mxlpy import Model, Simulator
from mxlpy.fns import mass_action_1s
from mxlpy.integrators import Scipy
from mxlpy.integrators.utils import detect_oscillations
from mxlpy.types import NoSteadyState, OscillationDetected

# ---------------------------------------------------------------------------
# Pure rate functions for an oscillating model (Lotka-Volterra)
# ---------------------------------------------------------------------------


def prey_growth(x: float, y: float, alpha: float, beta: float) -> float:
    """Prey growth rate: alpha*x - beta*x*y."""
    return alpha * x - beta * x * y


def predator_growth(x: float, y: float, delta: float, gamma: float) -> float:
    """Predator growth rate: delta*x*y - gamma*y."""
    return delta * x * y - gamma * y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lotka_volterra() -> Model:
    """Classic Lotka-Volterra predator-prey model (sustained oscillator)."""
    return (
        Model()
        .add_variables({"prey": 10.0, "predator": 5.0})
        .add_parameters({"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5})
        .add_reaction(
            "prey_dynamics",
            fn=prey_growth,
            args=["prey", "predator", "alpha", "beta"],
            stoichiometry={"prey": 1},
        )
        .add_reaction(
            "predator_dynamics",
            fn=predator_growth,
            args=["prey", "predator", "delta", "gamma"],
            stoichiometry={"predator": 1},
        )
    )


@pytest.fixture
def converging_model() -> Model:
    """Simple irreversible conversion with a well-defined steady state."""
    return (
        Model()
        .add_variables({"S": 10.0, "P": 0.0})
        .add_parameters({"k1": 1.0, "k2": 2.0})
        .add_reaction(
            "v1",
            fn=mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1, "P": 1},
        )
        .add_reaction(
            "v2",
            fn=mass_action_1s,
            args=["P", "k2"],
            stoichiometry={"P": -1},
        )
    )


# ---------------------------------------------------------------------------
# Unit tests for _detect_oscillations helper
# ---------------------------------------------------------------------------


def test_detect_oscillations_sine_wave() -> None:
    """Pure sine wave is detected as oscillating."""
    t = np.linspace(0, 200 * np.pi, 500)
    y = np.column_stack([np.sin(t), np.cos(t)])
    result = detect_oscillations(y, ["a", "b"], times=t)
    assert result is not None
    assert "a" in result.oscillating_species
    assert "b" in result.oscillating_species


def test_detect_oscillations_flat_signal() -> None:
    """Constant (flat) signal is NOT detected as oscillating."""
    y = np.ones((100, 2))
    result = detect_oscillations(y, ["a", "b"])
    assert result is None


def test_detect_oscillations_monotone_decay() -> None:
    """Monotonically decaying signal is NOT detected as oscillating."""
    t = np.linspace(0, 10, 200)
    y = np.column_stack([np.exp(-t), np.exp(-0.5 * t)])
    result = detect_oscillations(y, ["a", "b"], times=t)
    assert result is None


def test_detect_oscillations_too_few_samples() -> None:
    """History shorter than 10 samples always returns None."""
    t = np.linspace(0, 4 * np.pi, 5)
    y = np.column_stack([np.sin(t)])
    result = detect_oscillations(y, ["a"], times=t)
    assert result is None


def test_detect_oscillations_period_estimate() -> None:
    """Period estimate is in the right ballpark for a known sine wave."""
    period_true = 10.0
    t = np.linspace(0, 5 * period_true, 500)
    y = np.column_stack([np.sin(2 * np.pi * t / period_true)])
    result = detect_oscillations(y, ["a"], times=t)
    assert result is not None
    assert result.period is not None
    # Allow 30 % relative error from discretisation / lag resolution
    assert abs(result.period - period_true) / period_true < 0.3


def test_detect_oscillations_only_oscillating_species_listed() -> None:
    """Only the oscillating variable is reported; the flat one is not."""
    t = np.linspace(0, 20 * np.pi, 400)
    y = np.column_stack([np.sin(t), np.ones(len(t))])
    result = detect_oscillations(y, ["oscillating", "flat"], times=t)
    assert result is not None
    assert "oscillating" in result.oscillating_species
    assert "flat" not in result.oscillating_species


# ---------------------------------------------------------------------------
# Integration tests: Simulator + Scipy
# ---------------------------------------------------------------------------


def test_oscillating_model_returns_oscillation_detected(
    lotka_volterra: Model,
) -> None:
    """Lotka-Volterra returns OscillationDetected, not NoSteadyState."""
    result = (
        Simulator(lotka_volterra, integrator=Scipy)
        .simulate_to_steady_state(tolerance=1e-6)
        .get_result()
    )
    assert isinstance(result.value, OscillationDetected)


def test_oscillating_model_species_names_resolved(
    lotka_volterra: Model,
) -> None:
    """oscillating_species contains actual model variable names, not indices."""
    result = (
        Simulator(lotka_volterra, integrator=Scipy)
        .simulate_to_steady_state(tolerance=1e-6)
        .get_result()
    )
    assert isinstance(result.value, OscillationDetected)
    osc = result.value
    model_vars = set(lotka_volterra.get_variable_names())
    for species in osc.oscillating_species:
        assert species in model_vars, f"'{species}' is not a model variable name"


def test_oscillating_model_period_is_estimated(
    lotka_volterra: Model,
) -> None:
    """OscillationDetected carries a finite period estimate."""
    result = (
        Simulator(lotka_volterra, integrator=Scipy)
        .simulate_to_steady_state(tolerance=1e-6)
        .get_result()
    )
    assert isinstance(result.value, OscillationDetected)
    assert result.value.period is not None
    assert result.value.period > 0


def test_converging_model_not_affected(converging_model: Model) -> None:
    """A model that converges to a steady state is unaffected by the feature."""
    result = (
        Simulator(converging_model, integrator=Scipy)
        .simulate_to_steady_state(tolerance=1e-6)
        .get_result()
    )
    # Should succeed - no OscillationDetected, no NoSteadyState
    assert not isinstance(result.value, OscillationDetected)
    assert not isinstance(result.value, NoSteadyState)
    result.unwrap_or_err()  # must not raise


def test_oscillation_detected_exception_message() -> None:
    """OscillationDetected has the expected human-readable message."""
    exc = OscillationDetected(oscillating_species=["x", "y"], period=42.0)
    assert "oscillat" in str(exc).lower()
    assert exc.oscillating_species == ["x", "y"]
    assert exc.period == 42.0


def test_oscillation_detected_no_period() -> None:
    """OscillationDetected can be created without a period estimate."""
    exc = OscillationDetected(oscillating_species=["x"])
    assert exc.period is None


def test_unwrap_oscillation_detected_raises(lotka_volterra: Model) -> None:
    """unwrap_or_err on an oscillating result raises OscillationDetected."""
    result = (
        Simulator(lotka_volterra, integrator=Scipy)
        .simulate_to_steady_state(tolerance=1e-6)
        .get_result()
    )
    with pytest.raises(OscillationDetected):
        result.unwrap_or_err()
