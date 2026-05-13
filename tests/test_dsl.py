"""Tests for the reaction-network DSL."""

from __future__ import annotations

import pytest

from mxlpy import Model
from mxlpy.dsl import from_dsl
from mxlpy.fns import hill_1s, mass_action_1s, michaelis_menten_1s


def test_simple_irreversible() -> None:
    model = from_dsl(
        Model(),
        "k1, A --> B",
        variables={"A": 1.0, "B": 0.0},
        parameters={"k1": 0.1},
    )
    assert set(model.get_reaction_names()) == {"r1"}
    assert set(model.get_variable_names()) == {"A", "B"}


def test_reversible_reaction() -> None:
    model = from_dsl(
        Model(),
        "(k2, k3), C <--> D",
        variables={"C": 1.0, "D": 0.0},
        parameters={"k2": 0.05, "k3": 0.02},
    )
    assert set(model.get_reaction_names()) == {"r1_fwd", "r1_rev"}


def test_sequential_naming() -> None:
    model = from_dsl(
        Model(),
        """
        k1, A --> B
        k2, B --> C
        """,
        variables={"A": 1.0, "B": 0.0, "C": 0.0},
        parameters={"k1": 0.1, "k2": 0.2},
    )
    assert set(model.get_reaction_names()) == {"r1", "r2"}


def test_mm_shorthand() -> None:
    model = from_dsl(
        Model(),
        "mm(kcat, Km), S --> P",
        variables={"S": 2.0, "P": 0.0},
        parameters={"kcat": 1.0, "Km": 0.5},
    )
    assert "r1" in model.get_reaction_names()
    rxn = model._reactions["r1"]
    assert rxn.fn is michaelis_menten_1s
    assert rxn.args == ["S", "kcat", "Km"]


def test_hill_shorthand() -> None:
    model = from_dsl(
        Model(),
        "hill(v, K, n), A --> B",
        variables={"A": 1.0, "B": 0.0},
        parameters={"v": 1.0, "K": 0.3, "n": 2.0},
    )
    rxn = model._reactions["r1"]
    assert rxn.fn is hill_1s
    assert rxn.args == ["A", "v", "K", "n"]


def test_ma_shorthand() -> None:
    model = from_dsl(
        Model(),
        "ma(k1), A + B --> C",
        variables={"A": 1.0, "B": 0.5, "C": 0.0},
        parameters={"k1": 0.1},
    )
    rxn = model._reactions["r1"]
    result = rxn.fn(*[model.get_parameter_values()["k1"], 1.0, 0.5])
    assert abs(result - 0.05) < 1e-9


def test_arbitrary_expression() -> None:
    model = from_dsl(
        Model(),
        "k_deg * X, X -->",
        variables={"X": 1.0},
        parameters={"k_deg": 0.1},
    )
    rxn = model._reactions["r1"]
    assert set(rxn.args) == {"X", "k_deg"}


def test_named_fns_lookup() -> None:
    model = from_dsl(
        Model(),
        "mass_action_1s(A, k1), A --> B",
        variables={"A": 1.0, "B": 0.0},
        parameters={"k1": 0.1},
    )
    rxn = model._reactions["r1"]
    assert rxn.fn is mass_action_1s
    assert rxn.args == ["A", "k1"]


def test_degradation_empty_product() -> None:
    model = from_dsl(
        Model(),
        "k_deg * X, X -->",
        variables={"X": 2.0},
        parameters={"k_deg": 0.5},
    )
    stoich = model._reactions["r1"].stoichiometry
    assert "X" in stoich


def test_production_empty_reactant() -> None:
    model = from_dsl(
        Model(),
        "ma(k_syn), --> Y",
        variables={"Y": 0.0},
        parameters={"k_syn": 0.2},
    )
    stoich = model._reactions["r1"].stoichiometry
    assert "Y" in stoich


def test_stoichiometric_coefficient() -> None:
    model = from_dsl(
        Model(),
        "k1, 2A + B --> C",
        variables={"A": 2.0, "B": 1.0, "C": 0.0},
        parameters={"k1": 0.1},
    )
    stoich = model._reactions["r1"].stoichiometry
    assert stoich["A"] == -2.0
    assert stoich["B"] == -1.0
    assert stoich["C"] == 1.0


def test_comment_lines_ignored() -> None:
    model = from_dsl(
        Model(),
        """
        # this is a comment
        k1, A --> B
        # another comment
        """,
        variables={"A": 1.0, "B": 0.0},
        parameters={"k1": 0.1},
    )
    assert model.get_reaction_names() == ["r1"]


def test_missing_species_raises() -> None:
    with pytest.raises(ValueError, match="Species"):
        from_dsl(
            Model(),
            "k1, A --> B",
            variables={"A": 1.0},  # B missing
            parameters={"k1": 0.1},
        )


def test_missing_parameter_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        from_dsl(
            Model(),
            "k1 * A, A --> B",
            variables={"A": 1.0, "B": 0.0},
            parameters={},  # k1 missing
        )


def test_mm_ambiguous_substrate_raises() -> None:
    with pytest.raises(ValueError, match="mm\\(\\) requires exactly one"):
        from_dsl(
            Model(),
            "mm(kcat, Km), A + B --> C",
            variables={"A": 1.0, "B": 1.0, "C": 0.0},
            parameters={"kcat": 1.0, "Km": 0.5},
        )


def test_hill_ambiguous_substrate_raises() -> None:
    with pytest.raises(ValueError, match="hill\\(\\) requires exactly one"):
        from_dsl(
            Model(),
            "hill(v, K, n), A + B --> C",
            variables={"A": 1.0, "B": 1.0, "C": 0.0},
            parameters={"v": 1.0, "K": 0.3, "n": 2.0},
        )


def test_reversible_without_tuple_raises() -> None:
    with pytest.raises(ValueError, match="tuple"):
        from_dsl(
            Model(),
            "k1, C <--> D",
            variables={"C": 1.0, "D": 0.0},
            parameters={"k1": 0.05},
        )


def test_full_example() -> None:
    """Smoke test: the canonical example from the issue must build without error."""
    model = from_dsl(
        Model(),
        """
        k1,              A + B --> C
        (k2, k3),        C <--> D
        mm(kcat, Km),    E + S --> E + P
        hill(v, K, n),   A --> B
        k_deg * X,       X -->
        ma(k_syn),       --> Y
        """,
        variables={
            "A": 1.0,
            "B": 0.5,
            "C": 0.0,
            "D": 0.0,
            "E": 0.1,
            "S": 2.0,
            "P": 0.0,
            "X": 1.0,
            "Y": 0.0,
        },
        parameters={
            "k1": 0.1,
            "k2": 0.05,
            "k3": 0.02,
            "kcat": 1.0,
            "Km": 0.5,
            "v": 1.0,
            "K": 0.3,
            "n": 2.0,
            "k_deg": 0.1,
            "k_syn": 0.2,
        },
    )
    expected_rxns = {"r1", "r2_fwd", "r2_rev", "r3", "r4", "r5", "r6"}
    assert set(model.get_reaction_names()) == expected_rxns


def test_model_from_reactions_classmethod() -> None:
    model = Model.add_reactions_from_dsl(
        Model(),
        "k1, A --> B",
        variables={"A": 1.0, "B": 0.0},
        parameters={"k1": 0.1},
    )
    assert "r1" in model.get_reaction_names()
