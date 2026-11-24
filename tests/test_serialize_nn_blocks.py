# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Tests for nn_blocks export from a torch surrogate and import as a symbolic block.

Closes the gap `test_serialize.py::test_surrogate_raises_serialization_error`
documents: a surrogate whose architecture *is* representable in the shared
schema (dense layers, a recognized per-layer activation, unit-coefficient
stoichiometry) should round-trip through `save`/`load`, not be rejected the
way an opaque one (`_poly.Surrogate`) still correctly is.

The surrogate's own name ("corr_block") is deliberately distinct from its
output name ("corr"): `add_surrogate` inserts both as ids, and an
output name matching the surrogate's own name would collide with itself.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

import mxlpy
from mxlpy import KineticModelBuilder
from mxlpy.serialize import model_from_dict, model_to_dict, mxl_json_weights_files
from mxlpy.surrogates._torch import Surrogate
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from pathlib import Path


def _dense_softplus_model() -> nn.Sequential:
    """A tiny, deterministic (fixed-weight) dense+softplus network — 2 in, hidden 3, 1 out."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.Softplus(),
        nn.Linear(3, 1),
    )


def _make_model_with_surrogate() -> KineticModelBuilder:
    surrogate = Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    return (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_variable("y", 2.0)
        .add_surrogate("corr_block", surrogate)
    )


def test_exportable_surrogate_produces_nn_blocks_section() -> None:
    model = _make_model_with_surrogate()
    data = model_to_dict(model, model_id="m")
    nn_blocks = data["model"]["nn_blocks"]
    assert "corr_block" in nn_blocks
    block = nn_blocks["corr_block"]
    assert block["inputs"] == ["x", "y"]
    assert block["targets"] == ["x"]
    assert [layer["width"] for layer in block["layers"]] == [3, 1]
    assert block["layers"][0]["activation"]["name"] == "softplus"
    assert "activation" not in block["layers"][1]
    assert "activation" not in block
    assert block["trained"] is True
    assert block["weights_ref"] == "corr_block.weights.json"
    # additive mechanism: Add(Name(ode), Name(nde))
    assert block["mechanism"]["type"] == "Add"


def test_mxl_json_weights_files_shapes_match_layers() -> None:
    model = _make_model_with_surrogate()
    files = mxl_json_weights_files(model)
    assert set(files.keys()) == {"corr_block.weights.json"}
    weights = files["corr_block.weights.json"]
    # layer 1: 2 in -> 3 out
    assert len(weights["w1"]) == 3
    assert len(weights["w1"][0]) == 2
    assert len(weights["b1"]) == 3
    # layer 2 (output): 3 in -> 1 out
    assert len(weights["w2"]) == 1
    assert len(weights["w2"][0]) == 3
    assert len(weights["b2"]) == 1


def test_non_unit_stoichiometry_is_not_exportable() -> None:
    surrogate = Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 2.0}},
    )
    model = (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_variable("y", 2.0)
        .add_surrogate("corr_block", surrogate)
    )
    with pytest.raises(SerializationError, match="surrogates"):
        model_to_dict(model, model_id="m")


def test_unrecognized_activation_is_not_exportable() -> None:
    model_net = nn.Sequential(nn.Linear(2, 3), nn.ELU(), nn.Linear(3, 1))
    surrogate = Surrogate(
        model=model_net,
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    model = (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_variable("y", 2.0)
        .add_surrogate("corr_block", surrogate)
    )
    with pytest.raises(SerializationError, match="surrogates"):
        model_to_dict(model, model_id="m")


def test_save_load_round_trips_predictions(tmp_path: Path) -> None:
    """A torch surrogate loads as a symbolic block with the same dynamics."""
    model = _make_model_with_surrogate()
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")

    assert (tmp_path / "corr_block.weights.json").exists()
    weights_on_disk = json.loads((tmp_path / "corr_block.weights.json").read_text())
    assert set(weights_on_disk.keys()) == {"w1", "b1", "w2", "b2"}

    reloaded = mxlpy.load(path)
    assert isinstance(reloaded, KineticModelBuilder)
    block = reloaded.get_raw_hidden_states()["corr_block"]
    assert block.inputs == ["x", "y"]
    assert block.targets == ["x"]
    assert block.n_hidden == 0
    for state in ([0.3, -0.7], [1.0, 2.0]):
        np.testing.assert_allclose(
            reloaded(0.0, state), model(0.0, state), rtol=1e-6, atol=1e-7
        )


def test_save_load_save_is_idempotent_with_nn_blocks(tmp_path: Path) -> None:
    model = _make_model_with_surrogate()
    first = tmp_path / "a.mxl.json"
    second = tmp_path / "b.mxl.json"
    third = tmp_path / "c.mxl.json"

    mxlpy.save(model, first, model_id="m")
    mxlpy.save(mxlpy.load(first), second, model_id="m")
    mxlpy.save(mxlpy.load(second), third, model_id="m")

    assert json.loads(second.read_text()) == json.loads(third.read_text())


def test_load_rejects_nn_block_with_missing_weights_file(tmp_path: Path) -> None:
    model = _make_model_with_surrogate()
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")
    (tmp_path / "corr_block.weights.json").unlink()

    with pytest.raises(SerializationError, match="weights"):
        mxlpy.load(path)


def test_model_from_dict_reads_any_mechanism() -> None:
    model = _make_model_with_surrogate()
    data = model_to_dict(model, model_id="m")
    data["model"]["nn_blocks"]["corr_block"]["mechanism"] = {
        "type": "Mul",
        "children": [
            {"type": "Name", "value": "ode"},
            {"type": "Name", "value": "nde"},
        ],
    }
    weights = mxl_json_weights_files(model)
    loaded = model_from_dict(data, weights_by_ref=weights)
    mechanism = loaded.get_raw_hidden_states()["corr_block"].mechanism
    assert mechanism.name is None
    assert mechanism.fn(2.0, 3.0) == 6.0


def test_model_from_dict_rejects_non_dense_layer_type() -> None:
    model = _make_model_with_surrogate()
    data = model_to_dict(model, model_id="m")
    data["model"]["nn_blocks"]["corr_block"]["layers"][0]["type"] = "conv"
    weights = mxl_json_weights_files(model)
    with pytest.raises(SerializationError, match="dense"):
        model_from_dict(data, weights_by_ref=weights)


def test_model_from_dict_initialises_untrained_block() -> None:
    model = _make_model_with_surrogate()
    data = model_to_dict(model, model_id="m")
    block = data["model"]["nn_blocks"]["corr_block"]
    del block["weights_ref"]
    block["trained"] = False
    loaded = model_from_dict(data, weights_by_ref={})
    assert len(loaded.get_weight_names()) == 2 * 3 + 3 + 3 + 1


def test_surrogate_with_non_additive_term_on_its_target_is_refused() -> None:
    """A loaded surrogate is applied after the terms, which would change the dynamics."""
    model = _make_model_with_surrogate().add_parameter("kc", 0.5)
    model.add_term(
        "t", mxlpy.fns.constant, args=["kc"], target="x", mechanism="relative"
    )
    with pytest.raises(SerializationError, match="non-additive term"):
        model_to_dict(model, model_id="m")
    additive = _make_model_with_surrogate().add_parameter("kc", 0.5)
    additive.add_term(
        "t", mxlpy.fns.constant, args=["kc"], target="x", mechanism="additive"
    )
    assert model_to_dict(additive, model_id="m")["model"]["terms"]
