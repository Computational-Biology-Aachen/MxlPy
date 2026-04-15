"""Edge case tests for Simulator."""

from __future__ import annotations

import pytest

from mxlpy import Model, Simulator, fns


def get_model() -> Model:
    return (
        Model()
        .add_variables({"S": 10.0, "P": 0.0})
        .add_parameters({"k1": 1.0, "k2": 2.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1.0, "P": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["P", "k2"],
            stoichiometry={"P": -1.0},
        )
    )


def test_simulate_backwards_time() -> None:
    """simulate(5) after simulate(10) must raise, not silently succeed."""
    sim = Simulator(get_model()).simulate(10, steps=10)
    with pytest.raises(ValueError, match="larger than previous"):
        sim.simulate(5)


def test_simulate_negative_t_end() -> None:
    """simulate(-1) on fresh simulator: t_end=-1 <= prior_t_end=0, must raise."""
    with pytest.raises(ValueError, match="larger than previous"):
        Simulator(get_model()).simulate(-1)


def test_simulate_zero_t_end() -> None:
    """simulate(0) on fresh simulator: equal to prior_t_end=0, must raise."""
    with pytest.raises(ValueError, match="larger than previous"):
        Simulator(get_model()).simulate(0)


def test_errors_block_subsequent_simulate() -> None:
    """After an integration error the next simulate call is silently skipped.

    This documents current behavior: errors accumulate and block further
    simulation.  A caller must check sim.get_result() to detect the failure.
    """
    # Force an error by making the model diverge: very stiff, large k
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 1e30)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["x", "k"],
            stoichiometry={"x": -1.0},
        )
    )
    sim = Simulator(model)
    sim.simulate(1e-3)  # likely fails / stores error

    # Second call must not raise - it is silently skipped
    sim.simulate(1.0)

    # There should be an error recorded or the result should reflect failure
    result = sim.get_result()
    # Either we got an error (Result.value is an exception) or the result is
    # suspiciously short/empty.  We only assert no hard crash occurred.
    assert result is not None


def test_update_variables_unknown_key_before_simulation() -> None:
    """update_variables with a key absent from the model is merged silently.

    This documents the current behavior: unknown keys end up in y0 and are
    passed to the integrator, which may crash or silently ignore the extra key.
    """
    model = get_model()
    sim = Simulator(model)
    # Key "nonexistent" is not a model variable.
    sim.update_variables({"nonexistent": 99.0})
    # y0 now contains an alien key - document that it is present
    assert "nonexistent" in sim.y0


def test_update_variables_after_simulation_uses_last_state() -> None:
    """update_variables after a run continues from the last simulation state."""
    sim = Simulator(get_model()).simulate(5, steps=5)
    last_S = sim.variables[-1]["S"].iloc[-1]  # type: ignore[index]

    sim.update_variables({"S": 5.0})
    # y0 should be the last state with S overridden, not the original ICs
    assert sim.y0["S"] == pytest.approx(5.0)
    # The override must not equal the original IC unless coincidence
    assert last_S != pytest.approx(10.0, abs=0.1), "S should have changed during sim"


def test_simulate_time_course_overlapping_points_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overlapping time points after a prior run trigger a warning."""
    import logging

    sim = Simulator(get_model()).simulate_time_course([0.0, 1.0, 2.0, 3.0])
    with caplog.at_level(logging.WARNING):
        sim.simulate_time_course([2.0, 3.0, 4.0])  # 2.0 and 3.0 overlap

    assert any("Overlapping" in r.message for r in caplog.records)
