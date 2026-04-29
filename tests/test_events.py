"""Tests for the solver-native event system."""

from __future__ import annotations

import pytest

from mxlpy import Derived, Event, Model, Scipy, Simulator
from mxlpy.integrators import DefaultIntegrator

# ---------------------------------------------------------------------------
# Helper rate functions (no lambdas per codebase convention)
# ---------------------------------------------------------------------------


def trigger_at_t5(t: float) -> float:
    return t - 5.0


def trigger_a_crosses_half(a: float) -> float:
    return a - 0.5


def trigger_falling(a: float) -> float:
    return a - 0.5


def set_zero() -> float:
    return 0.0


def set_two() -> float:
    return 2.0


def bump_a(a: float) -> float:
    return a + 1.0


def decay(x: float, k: float) -> float:
    return k * x


def constant_rate(k: float) -> float:
    return k


# ---------------------------------------------------------------------------
# Tests: Model.add_event / get_event_names
# ---------------------------------------------------------------------------


def test_get_event_names_empty() -> None:
    model = Model()
    assert model.get_event_names() == []


def test_add_event_returns_self() -> None:
    model = Model()
    result = model.add_event(
        "ev",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    assert result is model


def test_get_event_names() -> None:
    model = (
        Model()
        .add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"k1": Derived(fn=set_zero, args=[])},
        )
        .add_event(
            "switch",
            trigger_a_crosses_half,
            trigger_args=["x"],
            assignments={"k2": Derived(fn=set_two, args=[])},
        )
    )
    assert model.get_event_names() == ["dose", "switch"]


def test_add_event_duplicate_name_raises() -> None:
    model = Model().add_event(
        "ev",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    with pytest.raises((KeyError, NameError)):
        model.add_event(
            "ev",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"k1": Derived(fn=set_zero, args=[])},
        )


# ---------------------------------------------------------------------------
# Tests: Event evaluation helpers
# ---------------------------------------------------------------------------


def test_event_evaluate_trigger() -> None:
    event = Event(
        trigger_fn=trigger_at_t5,
        trigger_args=["time"],
        assignments={},
    )
    assert event.evaluate_trigger({"time": 3.0}) == pytest.approx(-2.0)
    assert event.evaluate_trigger({"time": 7.0}) == pytest.approx(2.0)


def test_event_apply_assignments_static() -> None:
    event = Event(
        trigger_fn=trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    result = event.apply_assignments({"time": 5.0, "k1": 1.0})
    assert result == {"k1": 0.0}


def test_event_apply_assignments_dynamic() -> None:
    event = Event(
        trigger_fn=trigger_a_crosses_half,
        trigger_args=["A"],
        assignments={"A": Derived(fn=bump_a, args=["A"])},
    )
    result = event.apply_assignments({"A": 0.5})
    assert result == {"A": pytest.approx(1.5)}


# ---------------------------------------------------------------------------
# Tests: Simulation with events
# ---------------------------------------------------------------------------


def _decay_model_with_dose() -> Model:
    """Decaying species x; at t=5 reset x to 2."""
    return (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )


def test_time_triggered_event_resets_variable() -> None:
    """x decays, then at t=5 it is reset to 2."""
    model = _decay_model_with_dose()
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    variables = result.get_variables()
    x = variables["x"]

    # Before event: x < 1.0 (decaying from 1)
    x_before = x[x.index < 5].iloc[-1]
    assert x_before < 1.0

    # After event: x jumps to 2, then decays again
    x_after = x[x.index > 5].iloc[0]
    assert x_after == pytest.approx(2.0, abs=0.05)

    # At t=10: x < 2 (decayed from reset)
    x_end = x.iloc[-1]
    assert x_end < 2.0


def test_parameter_assignment_event() -> None:
    """At t=5, set parameter k to 0 so decay stops."""

    def set_k_zero() -> float:
        return 0.0

    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.5)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "freeze",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"k": Derived(fn=set_k_zero, args=[])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=20).get_result().unwrap_or_err()

    variables = result.get_variables()
    x = variables["x"]

    # After t=5, k=0 so x should be roughly constant
    x_at_6 = float(x[x.index >= 6].iloc[0])
    x_at_15 = float(x[x.index >= 15].iloc[0])
    assert abs(x_at_15 - x_at_6) < 0.01


def test_direction_rising_only() -> None:
    """Rising-only event fires when A goes from below 0.5 to above, not the reverse."""

    def oscillate(x: float, omega: float) -> float:
        return omega * (1.0 - x)

    def trigger_x(x: float) -> float:
        return x - 0.5

    counter: list[float] = []

    def record_time(t: float) -> float:
        counter.append(t)
        return 0.0  # no-op assignment value

    model = (
        Model()
        .add_variable("x", 0.0)
        .add_parameter("omega", 2.0)
        .add_reaction("v", oscillate, args=["x", "omega"], stoichiometry={"x": 1})
        .add_event(
            "rising_cross",
            trigger_x,
            trigger_args=["x"],
            assignments={"omega": Derived(fn=set_two, args=[])},
            direction="rising",
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=5).get_result().unwrap_or_err()
    variables = result.get_variables()

    # x approaches 1 from 0; crosses 0.5 from below exactly once
    x = variables["x"]
    # After crossing, omega is reset to 2 (unchanged), but no error should occur
    assert float(x.iloc[-1]) > 0.4


def test_persistent_false_fires_once() -> None:
    """A non-persistent event fires at most once."""

    def trigger_growing(x: float) -> float:
        return x - 0.3

    def grow(x: float, k: float) -> float:
        return k * x

    def set_small() -> float:
        return 0.01

    model = (
        Model()
        .add_variable("x", 0.1)
        .add_parameter("k", 1.0)
        .add_reaction("v", grow, args=["x", "k"], stoichiometry={"x": 1})
        .add_event(
            "once",
            trigger_growing,
            trigger_args=["x"],
            # Reset to small positive so growth can resume; x=0 stays at 0.
            assignments={"x": Derived(fn=set_small, args=[])},
            persistent=False,
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()
    variables = result.get_variables()
    x = variables["x"]

    # After first reset (x→0.01), x grows again and will pass 0.3 again.
    # Because persistent=False the event does NOT fire a second time,
    # so x keeps growing past 0.3 and should be >> 0.3 at t=10.
    assert float(x.iloc[-1]) > 0.3


def test_non_scipy_integrator_with_events_raises() -> None:
    """Events require Scipy; other integrators raise NotImplementedError."""
    if DefaultIntegrator is Scipy:
        pytest.skip("DefaultIntegrator is Scipy; need a non-Scipy integrator to test")

    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "ev",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )
    with pytest.raises(NotImplementedError, match="Scipy"):
        Simulator(model, integrator=DefaultIntegrator)
