"""Tests for the solver-native event system."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mxlpy import Derived, Event, Model, Scipy, Simulator
from mxlpy._kinetic_builder import ArityMismatchError
from mxlpy.integrators.abstract import MockIntegrator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mxlpy.types import Array

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
    model = Model().add_parameter("k1", 0.1)
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
        .add_variable("x", 0.0)
        .add_parameter("k1", 0.1)
        .add_parameter("k2", 0.1)
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
    model = Model().add_parameter("k1", 0.1).add_event(
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


def test_event_apply_assignments_chains_through_earlier_targets() -> None:
    """A later assignment must see an earlier one's newly computed value.

    This matches the order the integrator applies tied/simultaneous
    assignments in during a real simulation (see Scipy._apply_tied_events).
    """

    def echo(v: float) -> float:
        return v

    event = Event(
        trigger_fn=trigger_a_crosses_half,
        trigger_args=["A"],
        assignments={
            "a": Derived(fn=set_two, args=[]),
            "b": Derived(fn=echo, args=["a"]),
        },
    )
    result = event.apply_assignments({"A": 0.5, "a": 0.0})
    assert result == {"a": pytest.approx(2.0), "b": pytest.approx(2.0)}


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
    """Events require Scipy; other integrators raise NotImplementedError.

    Uses MockIntegrator directly (rather than DefaultIntegrator) so this test
    exercises the guard regardless of whether Assimulo is installed - relying
    on DefaultIntegrator left this guard with zero coverage in CI, since
    DefaultIntegrator falls back to Scipy there.
    """
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
        Simulator(model, integrator=MockIntegrator)


def test_non_scipy_integrator_events_added_after_construction_raises() -> None:
    """The events-require-Scipy guard must also catch events added post-construction.

    Simulator._initialise_integrator() only runs at construction/reset time,
    but the model is a live reference the caller can keep mutating - so the
    guard must be re-checked at every simulate*() entry point too, not just
    once up front.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
    )
    sim = Simulator(model, integrator=MockIntegrator)

    model.add_event(
        "ev",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"x": Derived(fn=set_two, args=[])},
    )

    with pytest.raises(NotImplementedError, match="Scipy"):
        sim.simulate(t_end=10)
    with pytest.raises(NotImplementedError, match="Scipy"):
        sim.simulate_time_course([1.0, 2.0])
    with pytest.raises(NotImplementedError, match="Scipy"):
        sim.simulate_to_steady_state()


# ---------------------------------------------------------------------------
# Regression tests for review findings
# ---------------------------------------------------------------------------


def echo_k(k: float) -> float:
    return k


def test_event_sees_live_parameter_after_update_parameter() -> None:
    """update_parameter() before simulate() must not leave events on a stale snapshot."""
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "capture",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=echo_k, args=["k"])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    sim.update_parameter("k", 42.0)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    x = result.get_variables()["x"]
    x_at_event = x[x.index >= 5].iloc[0]
    assert x_at_event == pytest.approx(42.0)


def const_rate(k: float) -> float:
    return k


def test_simulate_to_steady_state_applies_events() -> None:
    """A shutoff event mid-search must be applied, not silently ignored."""
    model = (
        Model()
        .add_variable("x", 0.0)
        .add_parameter("k", 1.0)
        .add_reaction("v", const_rate, args=["k"], stoichiometry={"x": 1})
        .add_event(
            "shutoff",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"k": Derived(fn=set_zero, args=[])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate_to_steady_state(tolerance=1e-6).get_result().unwrap_or_err()

    x = result.get_variables()["x"]
    assert float(x.iloc[-1]) == pytest.approx(5.0, abs=1e-3)


def grow(k: float) -> float:
    return k


def trigger_half(v: float) -> float:
    return v - 0.5


def jump_high() -> float:
    return 99.0


def mark_hit() -> float:
    return 1.0


def test_tied_events_both_apply() -> None:
    """Two events crossing zero at the same instant must both fire, not just the first."""
    model = (
        Model()
        .add_variable("x", 0.0)
        .add_variable("y", 0.0)
        .add_variable("hit_b", 0.0)
        .add_parameter("k", 0.1)
        .add_reaction("vx", grow, args=["k"], stoichiometry={"x": 1})
        .add_reaction("vy", grow, args=["k"], stoichiometry={"y": 1})
        .add_event(
            "ev_a_jump",
            trigger_half,
            trigger_args=["x"],
            assignments={"x": Derived(fn=jump_high, args=[])},
        )
        .add_event(
            "ev_b_marks",
            trigger_half,
            trigger_args=["y"],
            assignments={"hit_b": Derived(fn=mark_hit, args=[])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    hit_b = result.get_variables()["hit_b"]
    assert float(hit_b.iloc[-1]) == pytest.approx(1.0)


def trigger_five(v: float) -> float:
    return v - 5.0


def test_tied_event_does_not_fire_candidate_in_wrong_direction() -> None:
    """A same-instant near-zero trigger must not force-fire a wrong-direction event.

    x(t) = t grows through 5 (a rising crossing). y(t) starts at 10 and
    decays through 5 at the same instant (a falling crossing). ev_b is
    declared direction="rising", so even though its trigger is tied with
    ev_a's at t=5, it must not fire - the actual crossing there is falling.
    """
    model = (
        Model()
        .add_variable("x", 0.0)
        .add_variable("y", 10.0)
        .add_variable("hit_b", 0.0)
        .add_parameter("k", 1.0)
        .add_reaction("vx", grow, args=["k"], stoichiometry={"x": 1})
        .add_reaction("vy", grow, args=["k"], stoichiometry={"y": -1})
        .add_event(
            "ev_a_jump",
            trigger_five,
            trigger_args=["x"],
            assignments={"x": Derived(fn=jump_high, args=[])},
        )
        .add_event(
            "ev_b_marks",
            trigger_five,
            trigger_args=["y"],
            assignments={"hit_b": Derived(fn=mark_hit, args=[])},
            direction="rising",
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    hit_b = result.get_variables()["hit_b"]
    assert float(hit_b.iloc[-1]) == pytest.approx(0.0)


def always_zero() -> float:
    return 0.0


def set_ten() -> float:
    return 10.0


def echo_shared(shared: float) -> float:
    return shared


def test_apply_tied_events_applies_in_declaration_order() -> None:
    """Tied events must apply in declaration order, not scipy's winning-root order.

    Directly exercises Scipy._apply_tied_events with a fake solve_ivp result
    where the *second*-declared event ("second") is reported as the winning
    root (index 1) and the *first*-declared event ("first") is only found to
    be tied afterwards. "first" sets `shared`; "second" reads `shared` into
    `result`. If tied events were applied in scipy's report order (second,
    then first) "second" would read the stale pre-event value of `shared`.
    """
    events = {
        "first": Event(
            trigger_fn=always_zero,
            trigger_args=[],
            assignments={"shared": Derived(fn=set_ten, args=[])},
        ),
        "second": Event(
            trigger_fn=always_zero,
            trigger_args=[],
            assignments={"result": Derived(fn=echo_shared, args=["shared"])},
        ),
    }

    def dummy_rhs(t: float, y: Iterable[float]) -> tuple[float, ...]:  # noqa: ARG001
        return (0.0, 0.0)

    def get_dependent(t: float, y: Array) -> dict[str, float]:
        return {"time": t, "shared": float(y[0]), "result": float(y[1])}

    integrator = Scipy(rhs=dummy_rhs, y0=(0.0, 0.0))
    integrator._var_names = ["shared", "result"]
    integrator._get_dependent = get_dependent

    class _FakeRes:
        def __init__(self) -> None:
            # "second" (index 1) is reported as the winning root; "first"
            # (index 0) has no root of its own - it is only found via the
            # tie-detection scan against the winning event's state.
            self.t_events = [np.array([]), np.array([5.0])]
            self.y_events = [np.array([]), [np.array([0.0, 0.0])]]

    fired = integrator._apply_tied_events(
        active=events,
        res=_FakeRes(),
        fired_names=set(),
        suppressed_once=set(),
        t_seg_start=0.0,
        y_seg_start=np.array([0.0, 0.0]),
    )

    assert fired is not None
    _, y_post = fired
    result_idx = integrator._var_names.index("result")
    assert float(y_post[result_idx]) == pytest.approx(10.0)


def test_get_raw_events() -> None:
    model = Model().add_parameter("k1", 0.1).add_event(
        "dose",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    raw = model.get_raw_events()
    assert list(raw) == ["dose"]
    assert isinstance(raw["dose"], Event)


def test_remove_event() -> None:
    model = Model().add_parameter("k1", 0.1).add_event(
        "dose",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    model.remove_event("dose")
    assert model.get_event_names() == []

    # Name is freed for reuse after removal.
    model.add_event(
        "dose",
        trigger_at_t5,
        trigger_args=["time"],
        assignments={"k1": Derived(fn=set_zero, args=[])},
    )
    assert model.get_event_names() == ["dose"]


def test_remove_event_missing_raises() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_event("missing")


def test_add_event_unknown_assignment_target_raises() -> None:
    """An assignment target that isn't a raw variable or parameter must be
    rejected at add_event() time, not surface as a mid-simulation KeyError
    from update_parameter()/update_variable().
    """
    model = Model().add_variable("x", 1.0).add_parameter("k", 0.1)
    with pytest.raises(KeyError, match="k2"):
        model.add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"k2": Derived(fn=set_zero, args=[])},
        )
    # The rejected event must not have been partially registered.
    assert model.get_event_names() == []


def test_add_events_batch() -> None:
    model = Model().add_variable("x", 1.0).add_parameter("k", 0.1)
    model.add_events(
        {
            "dose": Event(
                trigger_fn=trigger_at_t5,
                trigger_args=["time"],
                assignments={"x": Derived(fn=set_two, args=[])},
            ),
            "shutoff": Event(
                trigger_fn=trigger_at_t5,
                trigger_args=["time"],
                assignments={"k": Derived(fn=set_zero, args=[])},
            ),
        }
    )
    assert set(model.get_event_names()) == {"dose", "shutoff"}


def test_remove_events_batch() -> None:
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
        .add_event(
            "shutoff",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_zero, args=[])},
        )
    )
    model.remove_events(["dose", "shutoff"])
    assert model.get_event_names() == []


# ---------------------------------------------------------------------------
# Regression tests: events referencing derived (non-raw) quantities
# ---------------------------------------------------------------------------


def double_k(k: float) -> float:
    return 2 * k


def echo_k2(k2: float) -> float:
    return k2


def test_event_assignment_can_reference_derived_parameter() -> None:
    """Assignments must resolve the full arg namespace, not just base params/vars."""
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_derived("k2", double_k, args=["k"])
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=echo_k2, args=["k2"])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    x = result.get_variables()["x"]
    x_at_event = x[x.index >= 5].iloc[0]
    assert x_at_event == pytest.approx(0.2)


def ratio(x: float, k: float) -> float:
    return x / k


def trigger_ratio_above_5(r: float) -> float:
    return r - 5.0


def set_half() -> float:
    return 0.5


def test_event_trigger_can_reference_dynamic_derived_variable() -> None:
    """Triggers must see derived variables that depend on state, not just raw vars."""
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_derived("r", ratio, args=["x", "k"])
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "trip",
            trigger_ratio_above_5,
            trigger_args=["r"],
            assignments={"x": Derived(fn=set_half, args=[])},
            persistent=False,
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    x = result.get_variables()["x"]
    # r = x/k crosses 5 (i.e. x crosses 0.5) on the way down from x=1;
    # the event then resets x to 0.5.
    assert float(x.iloc[-1]) < 0.5


# ---------------------------------------------------------------------------
# Regression tests: persistent refiring and cross-call one-shot suppression
# ---------------------------------------------------------------------------


def cos_2t(t: float) -> float:
    return math.cos(2 * t)


def trigger_x_zero(x: float) -> float:
    return x


def increment(count: float) -> float:
    return count + 1.0


def test_persistent_event_refires_as_sole_active_event() -> None:
    """A persistent event must keep refiring even when it is the only event.

    x(t) = sin(2t)/2 has rising zero-crossings at t = 0, pi, 2*pi, 3*pi, ...
    Over [0, 10] that is 3 rising crossings after t=0 (~3.14, ~6.28, ~9.42).
    """
    model = (
        Model()
        .add_variable("x", 0.0)
        .add_variable("fire_count", 0.0)
        .add_reaction("vx", cos_2t, args=["time"], stoichiometry={"x": 1})
        .add_event(
            "cross",
            trigger_x_zero,
            trigger_args=["x"],
            assignments={"fire_count": Derived(fn=increment, args=["fire_count"])},
            direction="rising",
        )
    )
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    fire_count = result.get_variables()["fire_count"]
    assert float(fire_count.iloc[-1]) >= 3


def test_persistent_false_stays_fired_across_simulate_calls() -> None:
    """A one-shot event fired in one simulate() call must stay fired in the next."""

    def trigger_growing(x: float) -> float:
        return x - 0.3

    def grow(x: float, k: float) -> float:
        return k * x

    def set_small() -> float:
        return 0.01

    def make_model() -> Model:
        return (
            Model()
            .add_variable("x", 0.1)
            .add_parameter("k", 1.0)
            .add_reaction("v", grow, args=["x", "k"], stoichiometry={"x": 1})
            .add_event(
                "once",
                trigger_growing,
                trigger_args=["x"],
                assignments={"x": Derived(fn=set_small, args=[])},
                persistent=False,
            )
        )

    single_call = Simulator(make_model(), integrator=Scipy)
    single_result = single_call.simulate(t_end=10).get_result().unwrap_or_err()
    x_single = float(single_result.get_variables()["x"].iloc[-1])

    split_calls = Simulator(make_model(), integrator=Scipy)
    split_calls.simulate(t_end=2).simulate(t_end=10)
    split_result = split_calls.get_result().unwrap_or_err()
    x_split = float(split_result.get_variables()["x"].iloc[-1])

    # The event must fire exactly once regardless of how the simulation is
    # chunked into simulate() calls, so both runs must land on the same
    # trajectory.
    assert x_split == pytest.approx(x_single, rel=1e-4)


def test_requested_point_inside_post_event_buffer_window_is_kept() -> None:
    """A requested output point landing inside the tiny post-event suppression
    buffer window must appear in the results, not be silently dropped.

    After "capture" fires at t=5 (persistent=True, the default), the
    segment-restart loop takes a short buffer sub-step of size
    ~1e-6 * t_event before re-arming the event. A time point requested just
    inside that window used to be neither part of the buffer sub-step's
    (previously unrecorded) trajectory nor part of the next full segment,
    since t_current had already advanced past it.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.0)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "capture",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )
    sim = Simulator(model, integrator=Scipy)
    inside_buffer_window = 5.000001
    result = (
        sim.simulate_time_course([1.0, 3.0, inside_buffer_window, 7.0, 10.0])
        .get_result()
        .unwrap_or_err()
    )

    x = result.get_variables()["x"]
    assert inside_buffer_window in x.index
    assert float(x[inside_buffer_window]) == pytest.approx(2.0)


def test_update_variable_preserves_fired_names_across_calls() -> None:
    """update_variable()/update_variables() rebuild the integrator to pick up
    the new y0, but that rebuild must not be treated as a reset: a
    persistent=False event that already fired must stay fired afterwards.
    """

    def trigger_growing(x: float) -> float:
        return x - 0.3

    def grow(x: float, k: float) -> float:
        return k * x

    def set_small() -> float:
        return 0.01

    model = (
        Model()
        .add_variable("x", 0.1)
        .add_variable("y", 0.0)
        .add_parameter("k", 1.0)
        .add_reaction("v", grow, args=["x", "k"], stoichiometry={"x": 1})
        .add_event(
            "once",
            trigger_growing,
            trigger_args=["x"],
            assignments={"x": Derived(fn=set_small, args=[])},
            persistent=False,
        )
    )
    sim = Simulator(model, integrator=Scipy)
    sim.simulate(t_end=2)
    # Unrelated mid-simulation state tweak - must not resurrect the event.
    sim.update_variable("y", 5.0)
    result = sim.simulate(t_end=10).get_result().unwrap_or_err()

    x = result.get_variables()["x"]
    # If the event refired, x would be reset back down near 0.01 a second
    # time instead of growing unboundedly past 0.3 again.
    assert float(x.iloc[-1]) > 0.3


def test_rename_event_updates_registry() -> None:
    """Renaming an event itself must move its entry in the event registry,
    not just its id, otherwise get_event_names() goes stale and the freed
    old name can silently collide with a later add_event().
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_event(
            "dose",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )
    model.rename("dose", "drug_dose")

    assert model.get_event_names() == ["drug_dose"]
    raw = model.get_raw_events()
    assert "drug_dose" in raw
    assert "dose" not in raw


def test_rename_referenced_name_updates_event_args() -> None:
    """Renaming a variable referenced by an event's trigger/assignment must
    update the event's trigger_args and assignment target, not leave it
    pointing at the now-nonexistent old name.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", decay, args=["x", "k"], stoichiometry={"x": -1})
        .add_event(
            "dose",
            trigger_x_zero,
            trigger_args=["x"],
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )
    model.rename("x", "conc")

    event = model.get_raw_events()["dose"]
    assert event.trigger_args == ["conc"]
    assert set(event.assignments) == {"conc"}

    # Must actually be simulatable afterwards, not just internally consistent.
    sim = Simulator(model, integrator=Scipy)
    result = sim.simulate(t_end=1).get_result().unwrap_or_err()
    assert "conc" in result.get_variables().columns


def test_get_unused_parameters_excludes_event_only_parameter() -> None:
    """A parameter only referenced from an event's trigger/assignment must
    not be reported as unused - it is load-bearing for the event.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("threshold", 0.5)
        .add_parameter("truly_unused", 1.0)
        .add_event(
            "dose",
            trigger_a_crosses_half,
            trigger_args=["x"],
            assignments={"x": Derived(fn=bump_a, args=["threshold"])},
        )
    )
    assert model.get_unused_parameters() == {"truly_unused"}


def test_add_event_trigger_arity_mismatch_raises() -> None:
    """A trigger_fn/trigger_args arity mismatch must raise ArityMismatchError
    eagerly (like every other model component), not a bare TypeError deep
    inside solve_ivp the first time the event is actually evaluated.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_event(
            "bad",
            trigger_at_t5,
            trigger_args=[],  # trigger_at_t5 expects one arg
            assignments={"x": Derived(fn=set_two, args=[])},
        )
    )
    with pytest.raises(ArityMismatchError):
        Simulator(model, integrator=Scipy)


def test_add_event_assignment_arity_mismatch_raises() -> None:
    """An assignment Derived's fn/args arity mismatch must also raise
    ArityMismatchError eagerly, not silently at simulation time.
    """
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_event(
            "bad",
            trigger_at_t5,
            trigger_args=["time"],
            assignments={"x": Derived(fn=bump_a, args=[])},  # bump_a needs "a"
        )
    )
    with pytest.raises(ArityMismatchError):
        Simulator(model, integrator=Scipy)
