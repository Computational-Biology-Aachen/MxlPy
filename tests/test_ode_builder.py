# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Regression coverage for OdeModelBuilder's dx/dt computation.

This class had zero test coverage before this file and, until fixed here,
its core evaluation path (`__call__`/`_get_args`) never actually called a
diff_eq's rate law at all — `ModelCache.dyn_order` only ever included a
diff_eq's own name when that variable's *initial value* happened to be
dynamic (an unrelated case), so `_get_right_hand_side` silently returned
the input concentration unchanged instead of a derivative. A second,
related bug in an intermediate fix attempt: naively evaluating diff_eqs
via the same in-place-mutating `calculate_inpl` pattern `Derived` uses is
wrong for diff_eqs specifically, since one diff_eq's evaluation would
overwrite `args[var]` with *its* derivative — and a second diff_eq that
legitimately needs `var`'s *concentration* as an input would then read the
wrong value. The fix evaluates every diff_eq from one frozen
pre-evaluation snapshot instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mxlpy import OdeModelBuilder, Simulator, SteadyStateModelBuilder
from mxlpy.simulation import OdeSimulation
from mxlpy.surrogates.abstract_ode import AbstractOdeSurrogate


@dataclass(kw_only=True)
class _MockOdeSurrogate(AbstractOdeSurrogate):
    """A minimal, deterministic `AbstractOdeSurrogate` for testing `OdeModelBuilder.add_surrogate`.

    Every output gets the same constant `correction` value, regardless of
    `args` — enough to test wiring (which outputs land in `args`, which
    ones additionally sum into a target's dx/dt), not prediction logic.
    """

    correction: float

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],  # noqa: ARG002, for API compatibility
    ) -> dict[str, float]:
        return dict.fromkeys(self.outputs, self.correction)


def test_single_variable_dxdt_is_actually_computed() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0)
    )
    dxdt = model(0.0, [2.0])
    assert dxdt == (-1.0,)


def test_coupled_system_reads_concentrations_not_derivatives() -> None:
    # dy/dt depends on x's *concentration* — if x's own diff_eq ran first
    # and overwrote args["x"] with x's derivative, dy/dt would silently
    # use the wrong value here.
    model = (
        OdeModelBuilder()
        .add_parameter("k1", 0.5)
        .add_parameter("k2", 0.3)
        .add_diff_eq("x", fn=lambda x, k1: -k1 * x, args=["x", "k1"], initial_value=2.0)
        .add_diff_eq(
            "y",
            fn=lambda x, y, k1, k2: k1 * x - k2 * y,
            args=["x", "y", "k1", "k2"],
            initial_value=1.0,
        )
    )
    dxdt, dydt = model(0.0, [2.0, 1.0])
    assert dxdt == -1.0
    assert dydt == 0.5 * 2.0 - 0.3 * 1.0


def test_diff_eq_depending_on_a_dynamic_derived_quantity() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_derived("rate", fn=lambda x, k: k * x, args=["x", "k"])
        .add_diff_eq("x", fn=lambda rate: -rate, args=["rate"], initial_value=2.0)
    )
    assert model(0.0, [2.0]) == (-1.0,)


def test_full_simulation_matches_analytic_exponential_decay() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0)
    )
    result = Simulator(model).simulate(10.0).get_result().unwrap_or_err()
    df = result.get_variables(
        include_derived_variables=False,
        include_readouts=False,
        include_surrogate_variables=False,
    )
    t_final = df.index[-1]
    analytic = 2.0 * np.exp(-0.5 * t_final)
    np.testing.assert_allclose(df["x"].iloc[-1], analytic, rtol=1e-4)


def test_get_variable_names_matches_get_diff_eq_names() -> None:
    model = OdeModelBuilder().add_diff_eq(
        "x", fn=lambda x: -x, args=["x"], initial_value=1.0
    )
    assert model.get_variable_names() == model.get_diff_eq_names() == ["x"]


def test_add_surrogate_targeted_output_sums_onto_dxdt() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0)
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[],
                outputs=["corr_out"],
                targets={"corr_out": ["x"]},
                correction=0.1,
            ),
        )
    )
    (dxdt,) = model(0.0, [2.0])
    # mechanistic term (-0.5 * 2.0 = -1.0) + surrogate output (0.1), always
    # additive, coefficient 1 — no mechanism/composition choice at all
    # (mxlpy.surrogates.abstract_ode's module docstring).
    assert dxdt == pytest.approx(-1.0 + 0.1)


def test_add_surrogate_untargeted_output_is_a_plain_derived_value() -> None:
    """An output absent from `targets` never touches dx/dt — it's just a named value, usable as an arg elsewhere."""
    model = (
        OdeModelBuilder()
        .add_diff_eq(
            "x",
            fn=lambda x, corr_extra: -x + corr_extra,
            args=["x", "corr_extra"],
            initial_value=2.0,
        )
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[], outputs=["corr_extra"], targets={}, correction=0.4
            ),
        )
    )
    (dxdt,) = model(0.0, [2.0])
    assert dxdt == pytest.approx(-2.0 + 0.4)
    assert model.get_args()["corr_extra"] == pytest.approx(0.4)


def test_derived_can_reference_an_earlier_surrogates_untargeted_output() -> None:
    """Surrogates fold into the same topological sort as `_derived` — a derived quantity may depend on a surrogate's output, and vice versa."""
    model = (
        OdeModelBuilder()
        .add_diff_eq("x", fn=lambda x: -x, args=["x"], initial_value=1.0)
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[], outputs=["corr_out"], targets={}, correction=3.0
            ),
        )
        .add_derived("doubled", fn=lambda corr_out: corr_out * 2, args=["corr_out"])
    )
    assert model.get_args()["doubled"] == pytest.approx(6.0)


def test_multiple_surrogates_targeting_the_same_variable_sum_order_independently() -> (
    None
):
    model = (
        OdeModelBuilder()
        .add_diff_eq("x", fn=lambda _x: 0.0, args=["x"], initial_value=1.0)
        .add_surrogate(
            "a",
            _MockOdeSurrogate(
                args=[], outputs=["a_out"], targets={"a_out": ["x"]}, correction=1.0
            ),
        )
        .add_surrogate(
            "b",
            _MockOdeSurrogate(
                args=[], outputs=["b_out"], targets={"b_out": ["x"]}, correction=1.0
            ),
        )
    )
    (dxdt,) = model(0.0, [1.0])
    assert dxdt == pytest.approx(2.0)


def test_add_surrogate_with_unknown_target_raises_and_does_not_leak_the_name() -> None:
    """A failed `add_surrogate` (bad `targets`) must not permanently reserve `name` in the id namespace.

    Regression test: `add_surrogate` used to call `_insert_id` before
    validating `surrogate.targets`, so a validation failure left `name`
    registered in `self._ids` with no matching entry in
    `self._surrogates` — an unrecoverable half-added state (the name
    became permanently unusable, and `remove_surrogate` couldn't clean it
    up either, since it pops from `self._surrogates`, which never
    contained it).
    """
    model = OdeModelBuilder().add_diff_eq(
        "x", fn=lambda x: -x, args=["x"], initial_value=1.0
    )
    bad_surrogate = _MockOdeSurrogate(
        args=[], outputs=["bad_out"], targets={"bad_out": ["y"]}, correction=0.0
    )
    with pytest.raises(KeyError, match="y"):
        model.add_surrogate("corr", bad_surrogate)

    assert "corr" not in model.ids
    assert "bad_out" not in model.ids
    assert model.get_raw_surrogates() == {}

    # The name must be fully reusable afterwards.
    good_surrogate = _MockOdeSurrogate(
        args=[], outputs=["good_out"], targets={"good_out": ["x"]}, correction=0.1
    )
    model.add_surrogate("corr", good_surrogate)
    assert "corr" in model.get_raw_surrogates()


def test_remove_surrogate_removes_its_contribution_and_its_output_ids() -> None:
    model = (
        OdeModelBuilder()
        .add_diff_eq("x", fn=lambda x: -x, args=["x"], initial_value=1.0)
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[],
                outputs=["corr_out"],
                targets={"corr_out": ["x"]},
                correction=0.1,
            ),
        )
    )
    assert "corr_out" in model.ids
    with_correction = model(0.0, [1.0])
    model.remove_surrogate("corr")
    without_correction = model(0.0, [1.0])

    assert with_correction != without_correction
    assert without_correction == (-1.0,)
    assert model.get_raw_surrogates() == {}
    assert "corr" not in model.ids
    assert "corr_out" not in model.ids


def test_remove_diff_eq_strips_it_from_surrogate_targets() -> None:
    """A removed diff_eq must not linger in a surrogate's `targets`, or a later evaluation raises `KeyError`.

    Regression test (found by an independent review): `remove_diff_eq` had
    no analog of `KineticModelBuilder.remove_variable`'s
    `remove_stoichiometries` cleanup, so a surrogate could keep targeting
    a diff_eq name that no longer existed.
    """
    model = (
        OdeModelBuilder()
        .add_diff_eq("x", fn=lambda x: -x, args=["x"], initial_value=1.0)
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[],
                outputs=["corr_out"],
                targets={"corr_out": ["x"]},
                correction=0.1,
            ),
        )
    )
    model.remove_diff_eq("x")
    model.add_diff_eq("y", fn=lambda y: -y, args=["y"], initial_value=2.0)

    assert model.get_raw_surrogates()["corr"].targets == {"corr_out": []}
    # Must not raise KeyError: "x" is gone from every surrogate's targets.
    (dydt,) = model(0.0, [2.0])
    assert dydt == pytest.approx(-2.0)


def test_rename_updates_surrogate_name_output_name_and_target_references() -> None:
    """`rename` must handle a surrogate's own name, a surrogate output name, and a targeted diff_eq's name.

    Regression test (found by an independent review): none of the three
    were handled, silently breaking the id-uniqueness invariant
    `_insert_id` exists to enforce for a renamed surrogate output.
    """
    model = (
        OdeModelBuilder()
        .add_diff_eq(
            "x",
            fn=lambda x, corr_out: -x + corr_out,
            args=["x", "corr_out"],
            initial_value=1.0,
        )
        .add_surrogate(
            "corr",
            _MockOdeSurrogate(
                args=[],
                outputs=["corr_out"],
                targets={"corr_out": ["x"]},
                correction=1.0,
            ),
        )
    )

    model.rename("corr_out", "renamed_out")
    surrogate = model.get_raw_surrogates()["corr"]
    assert surrogate.outputs == ["renamed_out"]
    assert surrogate.targets == {"renamed_out": ["x"]}
    assert model.get_raw_diff_eqs()["x"].args == ["x", "renamed_out"]
    with pytest.raises(NameError):
        model.add_derived("renamed_out", fn=lambda: 1.0, args=[])

    model.rename("corr", "corr2")
    assert "corr2" in model.get_raw_surrogates()
    assert "corr" not in model.get_raw_surrogates()

    model.rename("x", "x2")
    assert model.get_raw_surrogates()["corr2"].targets == {"renamed_out": ["x2"]}

    # Mechanistic fn reads the renamed "renamed_out" arg directly (-1.0 +
    # 1.0 = 0.0), *and* the surrogate's targets-summation pass separately
    # adds its output again (+1.0) — both capabilities are independent and
    # compose, this isn't double-counting a single mechanism.
    (dxdt,) = model(0.0, [1.0])
    assert dxdt == pytest.approx((-1.0 + 1.0) + 1.0)


def production_decay_model() -> OdeModelBuilder:
    """dx/dt = k_in - k * x, with steady state x = k_in / k."""
    return (
        OdeModelBuilder()
        .add_parameter("k_in", 1.0)
        .add_parameter("k", 0.5)
        .add_diff_eq(
            "x",
            fn=lambda k_in, x, k: k_in - k * x,
            args=["k_in", "x", "k"],
            initial_value=0.0,
        )
        .add_derived("d", fn=lambda x: 2 * x, args=["x"])
    )


def test_update_variable() -> None:
    model = production_decay_model().update_variable("x", 3.0)
    assert model.get_initial_conditions() == {"x": 3.0}


def test_update_variables() -> None:
    model = production_decay_model().update_variables({"x": 4.0})
    assert model.get_initial_conditions() == {"x": 4.0}


def test_update_variable_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        production_decay_model().update_variable("y", 1.0)


def test_simulation_variables_are_concentrations() -> None:
    model = production_decay_model()
    result = Simulator(model).simulate(10.0).get_result().unwrap_or_err()
    assert isinstance(result, OdeSimulation)

    variables = result.get_combined()
    assert list(variables.columns) == ["x", "d"]
    np.testing.assert_allclose(variables["d"], 2 * variables["x"])

    rhs = result.get_right_hand_side()
    np.testing.assert_allclose(rhs["x"], 1.0 - 0.5 * variables["x"])


def test_simulate_to_steady_state() -> None:
    result = (
        Simulator(production_decay_model())
        .simulate_to_steady_state()
        .get_result()
        .unwrap_or_err()
    )
    assert result.get_new_y0()["x"] == pytest.approx(2.0, rel=1e-4)


def test_simulator_rejects_steady_state_model() -> None:
    with pytest.raises(TypeError, match="evaluate"):
        Simulator(SteadyStateModelBuilder().add_parameter("a", 1.0))  # type: ignore[arg-type]


def test_use_jacobian_warns_for_ode_model(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        Simulator(production_decay_model(), use_jacobian=True)
    assert "Jacobians are only supported" in caplog.text


def test_simulation_args_robust_to_builder_returning_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OdeModelBuilder.get_args_time_course

    def with_variables(
        self: OdeModelBuilder, variables: pd.DataFrame, **kwargs: bool
    ) -> pd.DataFrame:
        return pd.concat((variables, original(self, variables, **kwargs)), axis=1)

    monkeypatch.setattr(OdeModelBuilder, "get_args_time_course", with_variables)
    model = production_decay_model()
    result = Simulator(model).simulate(1.0).get_result().unwrap_or_err()
    assert list(result.get_combined().columns) == ["x", "d"]
