"""Tests for the label_map module."""

from __future__ import annotations

import numpy as np
import pytest

from mxlpy import Model
from mxlpy.fns import mass_action_1s
from mxlpy.label_map import (
    LabelMapper,
    _assign_compound_labels,
    _create_isotopomer_reactions,
    _generate_binary_labels,
    _get_external_labels,
    _get_labels_per_variable,
    _map_substrates_to_products,
    _repack_stoichiometries,
    _split_label_string,
    _total_concentration,
    _unpack_stoichiometries,
)


def test_total_concentration() -> None:
    result = _total_concentration(0.1, 0.2, 0.3)
    assert np.isclose(result, 0.6)

    result = _total_concentration(1.0, 2.0, 3.0, 4.0)
    assert np.isclose(result, 10.0)

    result = _total_concentration(0.0)
    assert np.isclose(result, 0.0)


def test_generate_binary_labels() -> None:
    result = _generate_binary_labels("cpd", 0)
    assert result == ["cpd"]

    result = _generate_binary_labels("cpd", 1)
    assert result == ["cpd__0", "cpd__1"]

    result = _generate_binary_labels("cpd", 2)
    assert set(result) == {"cpd__00", "cpd__01", "cpd__10", "cpd__11"}
    assert len(result) == 4

    result = _generate_binary_labels("cpd", 3)
    assert len(result) == 8


def test_split_label_string() -> None:
    result = _split_label_string("01", [2])
    assert result == ["01"]

    result = _split_label_string("01", [1, 1])
    assert result == ["0", "1"]

    result = _split_label_string("0011", [4])
    assert result == ["0011"]

    result = _split_label_string("0011", [3, 1])
    assert result == ["001", "1"]

    result = _split_label_string("0011", [2, 2])
    assert result == ["00", "11"]

    result = _split_label_string("0011", [1, 3])
    assert result == ["0", "011"]


def test_map_substrates_to_products() -> None:
    result = _map_substrates_to_products("01", [1, 0])
    assert result == "10"

    result = _map_substrates_to_products("01", [0, 1])
    assert result == "01"

    result = _map_substrates_to_products("01", [1, 1])
    assert result == "11"

    result = _map_substrates_to_products("0110", [3, 2, 1, 0])
    assert result == "0110"

    result = _map_substrates_to_products("0110", [0, 1, 2, 3])
    assert result == "0110"

    result = _map_substrates_to_products("0110", [3, 1, 2, 0])
    assert result == "0110"

    result = _map_substrates_to_products("1100", [2, 3, 0, 1])
    assert result == "0011"

    result = _map_substrates_to_products("1010", [1, 0, 3, 2])
    assert result == "0101"


def test_unpack_stoichiometries() -> None:
    result = _unpack_stoichiometries({"A": -1, "B": -2, "C": 1})
    assert result == (["A", "B", "B"], ["C"])

    result = _unpack_stoichiometries({"A": -1, "B": 2})
    assert result == (["A"], ["B", "B"])

    result = _unpack_stoichiometries({"A": -1, "B": -1, "C": 2})
    assert result == (["A", "B"], ["C", "C"])

    result = _unpack_stoichiometries({"A": 0, "B": -1, "C": 1})
    assert result == (["B"], ["C"])


def test_get_labels_per_variable() -> None:
    result = _get_labels_per_variable({"A": 1, "B": 2}, ["A", "B", "C"])
    assert result == [1, 2, 0]

    result = _get_labels_per_variable({"A": 1}, ["A", "B", "C"])
    assert result == [1, 0, 0]

    result = _get_labels_per_variable({}, ["A", "B", "C"])
    assert result == [0, 0, 0]

    result = _get_labels_per_variable({"A": 3, "C": 2}, ["A", "B", "C"])
    assert result == [3, 0, 2]


def test_repack_stoichiometries() -> None:
    result = _repack_stoichiometries(["A", "B"], ["C"])
    assert result == {"A": -1, "B": -1, "C": 1}

    result = _repack_stoichiometries(["A"], ["B", "B"])
    assert result == {"A": -1, "B": 2}

    result = _repack_stoichiometries(["A", "A"], ["B"])
    assert result == {"A": -2, "B": 1}

    result = _repack_stoichiometries([], ["A"])
    assert result == {"A": 1}

    result = _repack_stoichiometries(["A"], [])
    assert result == {"A": -1}


def test_assign_compound_labels() -> None:
    result = _assign_compound_labels(["A", "B"], ["", "01"])
    assert result == ["A", "B__01"]

    result = _assign_compound_labels(["A", "B", "C"], ["01", "", "10"])
    assert result == ["A__01", "B", "C__10"]

    result = _assign_compound_labels(["A", "B"], ["", ""])
    assert result == ["A", "B"]

    result = _assign_compound_labels(["A", "B"], ["01", "10"])
    assert result == ["A__01", "B__10"]


def test_get_external_labels() -> None:
    result = _get_external_labels(total_product_labels=2, total_substrate_labels=1)
    assert result == "1"

    result = _get_external_labels(total_product_labels=1, total_substrate_labels=1)
    assert result == ""

    result = _get_external_labels(total_product_labels=3, total_substrate_labels=1)
    assert result == "11"

    result = _get_external_labels(total_product_labels=0, total_substrate_labels=0)
    assert result == ""

    result = _get_external_labels(total_product_labels=0, total_substrate_labels=1)
    assert result == ""


@pytest.fixture
def simple_model() -> Model:
    model = Model()
    model.add_parameters({"k1": 1.0, "k2": 2.0})
    model.add_variables({"A": 1.0, "B": 0.0, "C": 0.0})

    model.add_reaction(
        "v1",
        fn=mass_action_1s,
        args=["A", "k1"],
        stoichiometry={"A": -1, "B": 1},
    )

    model.add_reaction(
        "v2",
        fn=mass_action_1s,
        args=["B", "k2"],
        stoichiometry={"B": -1, "C": 1},
    )

    return model


def test_create_isotopomer_reactions() -> None:
    test_model = Model()
    test_model.add_parameters({"k1": 1.0})
    test_model.add_variables({"A": 1.0, "B": 0.0})

    _create_isotopomer_reactions(
        model=test_model,
        label_variables={"A": 1, "B": 1},
        rate_name="v1",
        function=mass_action_1s,
        stoichiometry={"A": -1, "B": 1},
        labelmap=[0],
        args=["A", "k1"],
    )

    assert "v1__0" in test_model.get_raw_reactions()
    assert "v1__1" in test_model.get_raw_reactions()

    # Check that stoichiometries are correctly assigned
    assert test_model.get_raw_reactions()["v1__0"].stoichiometry == {
        "A__0": -1,
        "B__0": 1,
    }
    assert test_model.get_raw_reactions()["v1__1"].stoichiometry == {
        "A__1": -1,
        "B__1": 1,
    }

    # Test error case when labelmap is too short
    with pytest.raises(ValueError, match="Labelmap 'missing'"):
        _create_isotopomer_reactions(
            model=test_model,
            label_variables={"A": 2, "B": 1},
            rate_name="v2",
            function=mass_action_1s,
            stoichiometry={"A": -1, "B": 1},
            labelmap=[0],  # Too short for 2 labels
            args=["A", "k1"],
        )


def test_labelmapper_init(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 1},
        label_maps={"v1": [0]},
    )

    assert mapper.model == simple_model
    assert mapper.label_variables == {"A": 1, "B": 1}
    assert mapper.label_maps == {"v1": [0]}


def test_get_isotopomers(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 2},
    )

    isotopomers = mapper.get_isotopomers()
    assert set(isotopomers.keys()) == {"A", "B"}
    assert isotopomers["A"] == ["A__0", "A__1"]
    assert isotopomers["B"] == ["B__00", "B__01", "B__10", "B__11"]


def test_get_isotopomer_of(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 2, "C": 3},
    )

    assert mapper.get_isotopomer_of("A") == ["A__0", "A__1"]
    assert mapper.get_isotopomer_of("B") == ["B__00", "B__01", "B__10", "B__11"]
    assert len(mapper.get_isotopomer_of("C")) == 8  # 2^3 = 8 isotopomers


def test_get_isotopomers_by_regex(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 2, "C": 3},
    )

    result = mapper.get_isotopomers_by_regex("B", "B__1[01]")
    assert set(result) == {"B__10", "B__11"}

    result = mapper.get_isotopomers_by_regex("B", "B__0[01]")
    assert set(result) == {"B__00", "B__01"}

    result = mapper.get_isotopomers_by_regex("C", "C__1.*")
    assert len(result) == 4  # All isotopomers starting with 1


def test_get_isotopomers_of_at_position(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 2, "C": 3},
    )

    # Test with single position
    result = mapper.get_isotopomers_of_at_position("B", 0)
    assert set(result) == {"B__10", "B__11"}

    result = mapper.get_isotopomers_of_at_position("B", 1)
    assert set(result) == {"B__01", "B__11"}

    # Test with multiple positions for compound C with 3 labels
    result = mapper.get_isotopomers_of_at_position("C", [0, 2])
    expected = {"C__101", "C__111"}
    assert set(result) == expected


def test_get_isotopomers_of_with_n_labels(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 2, "C": 3},
    )

    # Test with A (1 label position)
    result = mapper.get_isotopomers_of_with_n_labels("A", 1)
    assert result == ["A__1"]

    result = mapper.get_isotopomers_of_with_n_labels("A", 0)
    assert result == ["A__0"]

    # Test with B (2 label positions)
    result = mapper.get_isotopomers_of_with_n_labels("B", 1)
    assert set(result) == {"B__10", "B__01"}

    result = mapper.get_isotopomers_of_with_n_labels("B", 2)
    assert result == ["B__11"]

    # Test with C (3 label positions)
    result = mapper.get_isotopomers_of_with_n_labels("C", 2)
    assert len(result) == 3  # 3C2 = 3 combinations with 2 labels
    assert set(result) == {"C__110", "C__101", "C__011"}


def test_build_model(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 1, "C": 1},
        label_maps={"v1": [0], "v2": [0]},
    )

    # Build model without initial labels
    labeled_model = mapper.build_model()

    # Check variables
    variables = labeled_model.get_raw_variables()
    assert "A__0" in variables
    assert "A__1" in variables
    assert "B__0" in variables
    assert "B__1" in variables
    assert "C__0" in variables
    assert "C__1" in variables

    # Check derived variables for totals
    derived = labeled_model.get_raw_derived()
    assert "A__total" in derived
    assert "B__total" in derived
    assert "C__total" in derived

    # Check reactions
    reactions = labeled_model.get_raw_reactions()
    assert "v1__0" in reactions
    assert "v1__1" in reactions
    assert "v2__0" in reactions
    assert "v2__1" in reactions

    # Build model with initial labels - position 0 is labeled (corresponding to A__1)
    labeled_model = mapper.build_model(initial_labels={"A": [0]})
    variables = labeled_model.get_initial_conditions()
    assert variables["A__1"] == 1.0  # Labeled at position 0 means A__1
    assert variables["A__0"] == 0.0  # Unlabeled isotopomer is 0

    # Test with index-based initial labels (providing the position of the label)
    labeled_model = mapper.build_model(initial_labels={"A": 0})
    variables = labeled_model.get_initial_conditions()
    assert variables["A__1"] == 1.0  # Same as above, different syntax
    assert variables["A__0"] == 0.0


def test_model_with_derived_variables(simple_model: Model) -> None:
    # Add derived variables
    simple_model.add_derived("A_plus_B", lambda a, b: a + b, args=["A", "B"])

    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 1, "C": 1},
        label_maps={"v1": [0], "v2": [0]},
    )

    labeled_model = mapper.build_model()

    # Check that derived variables are properly mapped
    derived = labeled_model.get_raw_derived()
    assert "A_plus_B" in derived
    assert derived["A_plus_B"].args == [
        "A__total",
        "B__total",
    ]


def test_build_model_with_list_initial_labels(simple_model: Model) -> None:
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 2, "B": 2, "C": 1},
        label_maps={"v1": [0, 1], "v2": [0, 1]},
    )

    # Test with list of initial labels
    labeled_model = mapper.build_model(initial_labels={"A": [0, 1]})

    # The isotopomer A__11 should have the full concentration
    variables = labeled_model.get_initial_conditions()
    assert variables["A__11"] == 1.0
    assert variables["A__00"] == 0.0
    assert variables["A__10"] == 0.0
    assert variables["A__01"] == 0.0


def test_build_model_without_labels(simple_model: Model) -> None:
    # Test the case where a metabolite is in the model but not in label_variables
    mapper = LabelMapper(
        model=simple_model,
        label_variables={"A": 1, "B": 1},  # C not included in label variables
        label_maps={"v1": [0], "v2": [0]},
    )

    labeled_model = mapper.build_model()

    # Verify that C exists in the model but is not labeled
    variables = labeled_model.get_raw_variables()
    assert "C" in variables
    assert "C__0" not in variables
    assert "C__1" not in variables


def test_create_isotopomer_reactions_with_error() -> None:
    test_model = Model()
    test_model.add_parameters({"k1": 1.0})
    test_model.add_variables({"A": 1.0, "B": 0.0})

    # Test error case when labelmap is too short
    with pytest.raises(ValueError, match="Labelmap 'missing'"):
        _create_isotopomer_reactions(
            model=test_model,
            label_variables={"A": 2, "B": 1},
            rate_name="v2",
            function=mass_action_1s,
            stoichiometry={"A": -1, "B": 1},
            labelmap=[0],  # Too short for 2 labels
            args=["A", "k1"],
        )
