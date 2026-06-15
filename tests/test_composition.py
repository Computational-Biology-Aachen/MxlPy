from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, Simulator, compose, fns


def _producer() -> Model:
    """Model that produces x."""
    return (
        Model()
        .add_parameter("k_in", 1.0)
        .add_variable("x", 1.0)
        .add_reaction("v_in", fns.constant, args=["k_in"], stoichiometry={"x": 1})
    )


def _consumer() -> Model:
    """Model that consumes the shared variable x."""
    return (
        Model()
        .add_parameter("k_out", 1.0)
        .add_variable("x", 1.0)
        .add_reaction(
            "v_out", fns.proportional, args=["k_out", "x"], stoichiometry={"x": -1}
        )
    )


def _independent() -> Model:
    """Model with a disjoint namespace from _producer."""
    return (
        Model()
        .add_parameter("k2", 2.0)
        .add_variable("y", 3.0)
        .add_reaction("v2", fns.constant, args=["k2"], stoichiometry={"y": 1})
    )


def test_compose_disjoint_merges_all_components() -> None:
    merged = compose(_producer(), _independent())

    assert set(merged.get_variable_names()) == {"x", "y"}
    assert set(merged.get_parameter_names()) == {"k_in", "k2"}
    assert set(merged.get_reaction_names()) == {"v_in", "v2"}


def test_compose_single_model_returns_independent_copy() -> None:
    original = _producer()
    merged = compose(original)

    assert merged is not original
    assert merged.ids == original.ids
    # Mutating the copy must not touch the original
    merged.update_parameter("k_in", 99.0)
    assert original.get_raw_parameters()["k_in"].value == 1.0


def test_compose_no_models_raises() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        compose()


def test_compose_conflict_raises_by_default() -> None:
    with pytest.raises(ValueError, match="duplicate names"):
        compose(_producer(), _consumer())


def test_compose_override_warns_and_later_model_wins() -> None:
    m1 = Model().add_parameter("k", 1.0).add_variable("x", 1.0)
    m2 = Model().add_parameter("k", 2.0).add_variable("x", 5.0)

    with pytest.warns(UserWarning, match="overridden"):
        merged = compose(m1, m2, raise_on_conflict=False)

    assert merged.get_raw_parameters()["k"].value == 2.0
    assert merged.get_raw_variables()["x"].initial_value == 5.0


def test_compose_does_not_mutate_inputs() -> None:
    m1 = _producer()
    m2 = _independent()
    ids_before_1 = m1.ids
    ids_before_2 = m2.ids

    compose(m1, m2)

    assert m1.ids == ids_before_1
    assert m2.ids == ids_before_2


def test_compose_three_models() -> None:
    m1 = Model().add_variable("x", 1.0)
    m2 = Model().add_variable("y", 2.0)
    m3 = Model().add_variable("z", 3.0)

    merged = compose(m1, m2, m3)

    assert set(merged.get_variable_names()) == {"x", "y", "z"}


def test_compose_merges_derived_readouts_and_data() -> None:
    base = Model().add_parameter("k_in", 1.0).add_variable("x", 1.0)
    extra = (
        Model()
        .add_parameter("k2", 2.0)
        .add_derived("d1", fns.proportional, args=["k2", "k2"])
        .add_readout("r1", fns.constant, args=["k2"])
        .add_data("dataset", pd.Series({"a": 1.0}))
    )

    merged = compose(base, extra)

    assert "d1" in merged.get_raw_derived()
    assert "r1" in merged.get_raw_readouts()
    assert "dataset" in merged.ids


def test_compose_shared_variable_simulates() -> None:
    # Producer and consumer share variable x; override resolves the collision.
    with pytest.warns(UserWarning, match="overridden"):
        merged = compose(_producer(), _consumer(), raise_on_conflict=False)

    result = Simulator(merged).simulate(10).get_result().unwrap_or_err()
    variables = result.variables

    assert "x" in variables.columns
    # x relaxes towards the steady state k_in / k_out = 1.0
    assert np.isclose(variables["x"].iloc[-1], 1.0, atol=1e-3)
