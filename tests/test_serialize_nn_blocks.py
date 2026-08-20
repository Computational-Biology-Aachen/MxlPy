"""Tests for nn_blocks export/import (mxl-schemas nn_blocks v2) via a torch surrogate.

Closes the gap `test_serialize.py::test_surrogate_raises_serialization_error`
documents: a surrogate whose architecture *is* representable in the shared
schema (dense layers, softplus activation, unit-coefficient stoichiometry)
should round-trip through `save`/`load`, not be rejected the way an opaque
one (`_poly.Surrogate`) still correctly is.

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
from mxlpy.serialize import model_from_dict, model_to_dict, nn_block_weights_files
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
    assert block["layers"] == [
        {"type": "dense", "width": 3},
        {"type": "dense", "width": 1},
    ]
    assert block["trained"] is True
    assert block["weights_ref"] == "corr_block.weights.json"
    assert block["activation"]["name"] == "softplus"
    # additive mechanism: Add(Name(ode), Name(nde))
    assert block["mechanism"]["type"] == "Add"


def test_nn_block_weights_files_shapes_match_layers() -> None:
    model = _make_model_with_surrogate()
    files = nn_block_weights_files(model)
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


def test_relu_activation_is_not_exportable() -> None:
    model_net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
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
    model = _make_model_with_surrogate()
    original_surrogate = model._surrogates["corr_block"]
    probe = np.array([0.3, -0.7])
    original_prediction = original_surrogate.predict_raw(probe)  # type: ignore[attr-defined]

    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")

    assert (tmp_path / "corr_block.weights.json").exists()
    weights_on_disk = json.loads((tmp_path / "corr_block.weights.json").read_text())
    assert set(weights_on_disk.keys()) == {"w1", "b1", "w2", "b2"}

    reloaded = mxlpy.load(path)
    # `load` now returns KineticModelBuilder | OdeModelBuilder (it dispatches
    # on the document's `kind`) — this file was written from a
    # KineticModelBuilder, so narrow back to it to keep `.outputs`/
    # `.stoichiometries` (KineticModelBuilder-only, via SurrogateProtocol)
    # visible to the type checker below.
    assert isinstance(reloaded, KineticModelBuilder)
    reloaded_surrogate = reloaded._surrogates["corr_block"]
    reloaded_prediction = reloaded_surrogate.predict_raw(probe)  # type: ignore[attr-defined]

    np.testing.assert_allclose(reloaded_prediction, original_prediction, rtol=1e-6)
    assert reloaded_surrogate.args == ["x", "y"]
    # Reconstruction can't reuse the original output name "corr" (add_surrogate
    # requires it to be a fresh, model-wide-unique id, and "x" — the target —
    # already exists as a variable) — it derives a block-scoped one instead.
    assert reloaded_surrogate.outputs == ["corr_block_x"]
    assert reloaded_surrogate.stoichiometries == {"corr_block_x": {"x": 1.0}}


def test_save_load_save_is_idempotent_with_nn_blocks(tmp_path: Path) -> None:
    model = _make_model_with_surrogate()
    first = tmp_path / "a.mxl.json"
    second = tmp_path / "b.mxl.json"

    mxlpy.save(model, first, model_id="m")
    mxlpy.save(mxlpy.load(first), second, model_id="m")

    doc_a = json.loads(first.read_text())
    doc_b = json.loads(second.read_text())
    assert doc_a == doc_b


def test_load_rejects_nn_block_with_missing_weights_file(tmp_path: Path) -> None:
    model = _make_model_with_surrogate()
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")
    (tmp_path / "corr_block.weights.json").unlink()

    with pytest.raises(SerializationError, match="weights"):
        mxlpy.load(path)


def test_model_from_dict_rejects_non_additive_mechanism() -> None:
    model = _make_model_with_surrogate()
    data = model_to_dict(model, model_id="m")
    data["model"]["nn_blocks"]["corr_block"]["mechanism"] = {
        "type": "Mul",
        "children": [
            {"type": "Name", "value": "ode"},
            {"type": "Name", "value": "nde"},
        ],
    }
    weights = nn_block_weights_files(model)
    with pytest.raises(SerializationError, match="additive"):
        model_from_dict(data, weights_by_ref=weights)
