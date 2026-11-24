# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Serialisation of terms and hidden-state blocks (spec_version 0.5, #229)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

import mxlpy
from mxlpy import KineticModelBuilder, OdeModelBuilder, fns
from mxlpy.serialize import (
    model_from_dict,
    model_to_dict,
    mxl_json_weights_files,
    ode_model_from_dict,
    ode_model_to_dict,
)
from mxlpy.types import InitialAssignment, SerializationError

ASSETS = Path(__file__).parent / "assets"
SCHEMAS = Path(__file__).resolve().parents[3] / "misc" / "mxl-schemas" / "v0"


def exp_mechanism(base: float, term: float) -> float:
    return base * math.exp(term)


def _decay(s: float, k: float) -> float:
    return -k * s


def _kinetic() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 1.0, "k2": 0.5, "kc": 0.1})
        .add_variables({"S": 1.0, "P": 0.5})
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1, "P": 1},
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["P", "k2"], stoichiometry={"P": -1}
        )
        .add_term("t_rxn", fns.constant, args=["kc"], target="v1", mechanism="relative")
        .add_term(
            "t_exp", fns.constant, args=["kc"], target="P", mechanism=exp_mechanism
        )
        .add_hidden_states(
            "aug",
            n_hidden=2,
            layers=(3,),
            activation="softplus",
            targets=["v1", "v2"],
            mechanism="relative",
            input_scales={"S": 2.0},
            init_scale=0.5,
        )
    )


def _ode() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameters({"k": 1.0, "kc": 0.1})
        .add_diff_eq("S", 1.0, _decay, args=["S", "k"])
        .add_term("t", fns.constant, args=["kc"], target="S", mechanism="additive")
        .add_hidden_states(
            "aug", n_hidden=1, layers=(2,), targets=["S"], mechanism="additive"
        )
    )


def _validate(doc: dict, kind: str) -> None:
    if not SCHEMAS.exists():
        pytest.skip("mxl-schemas checkout not found")
    schema = json.loads((SCHEMAS / f"{kind}-model.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(doc)


@pytest.mark.parametrize("model_fn", [_kinetic, _ode])
def test_round_trip(model_fn: object, tmp_path: Path) -> None:
    model = model_fn()  # type: ignore[operator]
    path = tmp_path / "m.mxl.json"
    mxlpy.save(model, path)
    doc = json.loads(path.read_text())
    assert doc["spec_version"] == "0.5"
    kind = "ode" if isinstance(model, OdeModelBuilder) else "kinetic"
    _validate(doc, kind)
    if SCHEMAS.exists():
        weights_schema = json.loads((SCHEMAS / "nn-weights.schema.json").read_text())
        jsonschema.Draft202012Validator(weights_schema).validate(
            json.loads((tmp_path / "aug.weights.json").read_text())
        )

    loaded = mxlpy.load(path)
    assert isinstance(loaded, KineticModelBuilder | OdeModelBuilder)
    assert type(loaded) is type(model)
    assert loaded.get_term_names() == model.get_term_names()
    assert loaded.get_weight_values() == model.get_weight_values()
    assert loaded.get_variable_names() == model.get_variable_names()
    state = {k: 0.7 + 0.1 * i for i, k in enumerate(model.get_variable_names())}
    pd.testing.assert_series_equal(
        loaded.get_right_hand_side(state), model.get_right_hand_side(state)
    )
    # saving the loaded model gives the same files
    again = tmp_path / "again" / "m.mxl.json"
    again.parent.mkdir()
    mxlpy.save(loaded, again)
    assert json.loads(again.read_text()) == doc


def test_mechanisms_and_input_scales_in_file() -> None:
    doc = model_to_dict(_kinetic(), model_id="m")
    terms = doc["model"]["terms"]
    assert terms["t_rxn"]["target_kind"] == "reaction"
    assert terms["t_exp"]["target_kind"] == "variable"
    block = doc["model"]["nn_blocks"]["aug"]
    assert block["hidden_states"] == 2
    assert block["input_scales"] == [2.0, 1.0]
    assert block["target_kind"] == "reaction"
    assert "scale" not in block
    assert block["layers"][0]["activation"]["name"] == "softplus"
    assert "activation" not in block["layers"][1]
    weights = mxl_json_weights_files(_kinetic())["aug.weights.json"]
    assert set(weights) == {"w1", "b1", "w2", "b2", "W", "h0"}
    assert np.shape(weights["w1"]) == (3, 4)
    assert np.shape(weights["W"]) == (2, 2)


def test_without_terms_the_version_is_unchanged() -> None:
    model = KineticModelBuilder().add_parameter("k", 1.0)
    assert model_to_dict(model, model_id="m")["spec_version"] == "0.2"


def test_unsaveable_blocks_raise() -> None:
    model = _kinetic().add_parameter("h_start", 0.2)
    model.update_variable(
        "aug_h0", InitialAssignment(fn=fns.constant, args=["h_start"])
    )
    with pytest.raises(SerializationError, match="initial assignments"):
        model_to_dict(model, model_id="m")
    mixed = KineticModelBuilder().add_variables({"S": 1.0}).add_parameter("k", 1.0)
    mixed.add_reaction(
        "v", fns.mass_action_1s, args=["S", "k"], stoichiometry={"S": -1}
    )
    mixed.add_hidden_states(
        "aug", n_hidden=1, layers=(), targets=["S", "v"], mechanism="additive"
    )
    with pytest.raises(SerializationError, match="mixes variable and reaction"):
        model_to_dict(mixed, model_id="m")


def test_custom_mechanism_from_file_is_symbolic() -> None:
    doc = model_to_dict(_kinetic(), model_id="m")
    loaded = model_from_dict(doc, weights_by_ref=mxl_json_weights_files(_kinetic()))
    mech = loaded.get_raw_terms()["t_exp"].mechanism
    assert mech.name is None
    assert mech.fn(2.0, 0.0) == pytest.approx(2.0)
    # it converts back to the same expression
    assert (
        model_to_dict(loaded, model_id="m")["model"]["terms"] == doc["model"]["terms"]
    )


def test_mxlweb_showcase_ude_matches_mxlweb_core() -> None:
    """mxlweb's showcase/ude (relative mechanism, softplus) as mxlweb-core computes it."""
    fixture = json.loads((ASSETS / "mxlweb_showcase_ude.fixture.json").read_text())
    model = model_from_dict(
        fixture["doc"],
        weights_by_ref={"ude_correction.weights.json": fixture["weights"]},
    )
    block = model.get_raw_hidden_states()["ude_correction"]
    assert block.n_hidden == 0
    assert block.scale == "ude_correction_scale"
    assert block.mechanism.name == "relative"
    # mxlweb-core's parameter vector: parameters, then weights
    assert model.get_addressable_names() == fixture["names"]
    for (prey, predator), expected in zip(
        fixture["states"], fixture["derivatives"], strict=True
    ):
        rhs = model.get_right_hand_side({"Prey": prey, "Predator": predator})
        np.testing.assert_allclose(rhs.to_numpy(), expected, rtol=1e-12, atol=1e-14)


def test_untrained_block_is_initialised_from_its_seed() -> None:
    doc = ode_model_to_dict(_ode(), model_id="m")
    block = doc["model"]["nn_blocks"]["aug"]
    block["trained"] = False
    del block["weights_ref"]
    loaded = ode_model_from_dict(doc)
    assert loaded.get_weight_names() == _ode().get_weight_names()
    assert ode_model_to_dict(loaded, model_id="m")["model"]["nn_blocks"]["aug"]


def test_review_fixes(tmp_path: Path) -> None:
    # blocks from files are validated like add_hidden_states
    fixture = json.loads((ASSETS / "mxlweb_showcase_ude.fixture.json").read_text())
    doc = json.loads(json.dumps(fixture["doc"]))
    doc["model"]["nn_blocks"]["ude_correction"]["targets"] = ["Prey", "Predatr"]
    with pytest.raises(SerializationError, match="Predatr"):
        model_from_dict(
            doc, weights_by_ref={"ude_correction.weights.json": fixture["weights"]}
        )
    weights = dict(fixture["weights"])
    del weights["b2"]
    with pytest.raises(SerializationError, match="weights file needs"):
        model_from_dict(
            fixture["doc"], weights_by_ref={"ude_correction.weights.json": weights}
        )
    # output scales round trip
    model = _ode()
    model.update_hidden_states("aug", output_scales={"S": 3.0})
    mxlpy.save(model, tmp_path / "m.mxl.json")
    loaded = mxlpy.load(tmp_path / "m.mxl.json")
    assert isinstance(loaded, OdeModelBuilder)
    assert loaded.get_raw_hidden_states()["aug"].output_scales[0] == 3.0
    pd.testing.assert_series_equal(
        loaded.get_right_hand_side(), model.get_right_hand_side()
    )
