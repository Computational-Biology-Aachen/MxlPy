"""Tests for the linear label map module."""

import pandas as pd
import pytest

from mxlpy.linear_label_map import (
    LinearLabelMapper,
    _add_label_influx_or_efflux,
    _generate_isotope_labels,
    _map_substrates_to_labelmap,
    _neg_one_div,
    _one_div,
    _relative_label_flux,
    _stoichiometry_to_duplicate_list,
    _unpack_stoichiometries,
)
from mxlpy.model import Derived, Model


def test_generate_isotope_labels() -> None:
    """Test generation of isotope labels."""
    labels = _generate_isotope_labels("A", 3)
    assert labels == ["A__0", "A__1", "A__2"]

    with pytest.raises(ValueError):
        _generate_isotope_labels("A", 0)


def test_unpack_stoichiometries() -> None:
    """Test unpacking of stoichiometries."""
    stoich = {"A": -2, "B": 1, "C": 3}
    subs, prods = _unpack_stoichiometries(stoich)

    assert subs == {"A": 2}
    assert prods == {"B": 1, "C": 3}

    with pytest.raises(NotImplementedError):
        _unpack_stoichiometries({"A": Derived(fn=lambda x: x, args=["A"])})


def test_stoichiometry_to_duplicate_list() -> None:
    """Test conversion of stoichiometry to duplicate list."""
    stoich = {"A": 2, "B": 1, "C": 3}
    result = _stoichiometry_to_duplicate_list(stoich)

    assert len(result) == 6
    assert result.count("A") == 2
    assert result.count("B") == 1
    assert result.count("C") == 3


def test_map_substrates_to_labelmap() -> None:
    """Test mapping substrates to label map."""
    substrates = ["A", "B", "C"]
    labelmap = [2, 0, 1]

    result = _map_substrates_to_labelmap(substrates, labelmap)

    assert result == ["B", "C", "A"]


def test_add_label_influx_or_efflux() -> None:
    """Test adding label influx or efflux."""
    # Test adding efflux
    subs = ["A", "B"]
    prods = ["C"]
    labelmap = [0, 1]

    subs_out, prods_out = _add_label_influx_or_efflux(subs, prods, labelmap)

    assert subs_out == subs
    assert prods_out == ["C", "EXT"]

    # Test adding influx
    subs = ["A"]
    prods = ["B", "C"]
    labelmap = [0, 1]

    subs_out, prods_out = _add_label_influx_or_efflux(subs, prods, labelmap)

    assert subs_out == ["A", "EXT"]
    assert prods_out == prods

    # Test error case
    subs = ["A", "B"]
    prods = ["C"]
    labelmap = [0]

    with pytest.raises(ValueError):
        _add_label_influx_or_efflux(subs, prods, labelmap)


def test_relative_label_flux() -> None:
    """Test calculation of relative label flux."""
    result = _relative_label_flux(0.5, 10.0)
    assert result == 5.0


def test_one_div() -> None:
    """Test one_div function."""
    assert _one_div(2.0) == 0.5
    assert _one_div(4.0) == 0.25


def test_neg_one_div() -> None:
    """Test neg_one_div function."""
    assert _neg_one_div(2.0) == -0.5
    assert _neg_one_div(4.0) == -0.25


def test_linear_label_mapper_get_isotopomers() -> None:
    """Test LinearLabelMapper get_isotopomers method."""
    model = Model()
    mapper = LinearLabelMapper(model=model, label_variables={"A": 2, "B": 3, "C": 1})

    isotopomers = mapper.get_isotopomers(["A", "C"])

    assert "A" in isotopomers
    assert "C" in isotopomers
    assert "B" not in isotopomers
    assert isotopomers["A"] == ["A__0", "A__1"]
    assert isotopomers["C"] == ["C__0"]


def test_linear_label_mapper_build_model() -> None:
    """Test LinearLabelMapper build_model method."""
    # Create a simple model
    model = Model()
    model.add_variables({"A": 1.0, "B": 0.0})
    model.add_parameters({"k1": 0.1})  # Changed from v1 to k1
    model.add_reaction(
        name="v1",
        fn=lambda k1: k1,
        stoichiometry={"A": -1, "B": 1},
        args=["k1"],  # Changed parameter name to k1
    )

    # Create the mapper
    mapper = LinearLabelMapper(
        model=model, label_variables={"A": 2, "B": 2}, label_maps={"v1": [0, 1]}
    )

    # Create conc and flux series
    concs = pd.Series({"A": 1.0, "B": 0.0})
    fluxes = pd.Series({"v1": 0.1})

    # Build the label model
    label_model = mapper.build_model(concs, fluxes, external_label=1.0)

    # Check that the model has the expected variables
    variables = label_model.get_raw_variables()
    assert "A__0" in variables
    assert "A__1" in variables
    assert "B__0" in variables
    assert "B__1" in variables

    # Check that the model has the expected reactions
    reactions = label_model.get_raw_reactions()
    assert "v1__0" in reactions
    assert "v1__1" in reactions


def test_linear_label_mapper_build_model_with_initial_labels() -> None:
    """Test LinearLabelMapper build_model with initial labels."""
    # Create a simple model
    model = Model()
    model.add_variables({"A": 1.0, "B": 1.0})
    model.add_parameters({"k1": 0.1})  # Changed from v1 to k1
    model.add_reaction(
        name="v1",
        fn=lambda k1: k1,
        stoichiometry={"A": -1, "B": 1},
        args=["k1"],  # Changed parameter name to k1
    )

    # Create the mapper
    mapper = LinearLabelMapper(
        model=model, label_variables={"A": 2, "B": 2}, label_maps={"v1": [0, 1]}
    )

    # Build the label model with initial labels
    label_model = mapper.build_model(
        concs=pd.Series({"A": 1.0, "B": 1.0}),
        fluxes=pd.Series({"v1": 0.1}),
        external_label=1.0,
        initial_labels={"A": 0},
    )

    # Check that the initial label is set
    y0 = label_model.get_initial_conditions()
    assert y0["A__0"] == 1.0
    assert y0["A__1"] == 0.0


def test_linear_label_mapper_build_model_with_multiple_initial_labels() -> None:
    # Test with multiple initial labels
    model = Model()
    model.add_variables({"A": 1.0, "B": 1.0})
    model.add_parameters({"k1": 0.1})  # Changed from v1 to k1
    model.add_reaction(
        name="v1",
        fn=lambda k1: k1,
        stoichiometry={"A": -1, "B": 1},
        args=["k1"],  # Changed parameter name to k1
    )

    # Create the mapper
    mapper = LinearLabelMapper(
        model=model, label_variables={"A": 2, "B": 2}, label_maps={"v1": [0, 1]}
    )

    label_model = mapper.build_model(
        concs=pd.Series({"A": 1.0, "B": 1.0}),
        fluxes=pd.Series({"v1": 0.1}),
        external_label=1.0,
        initial_labels={"A": [0, 1]},
    )

    # Check that the initial labels are set (should be 0.5 for each label)
    y0 = label_model.get_initial_conditions()
    assert y0["A__0"] == 0.5
    assert y0["A__1"] == 0.5
