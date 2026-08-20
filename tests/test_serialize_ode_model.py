"""Tests for OdeModelBuilder's .mxl.json serialisation (ode-model.schema.json, kind="ode").

Mirrors `test_serialize.py`/`test_serialize_nn_blocks.py`'s coverage of
`KineticModelBuilder`, but for the direct-derivative variant: variables carry
their dx/dt directly (no reactions/stoichiometry section), and attached
surrogates are `DerivativeSurrogateProtocol` (any of the three `mechanism`
presets round-trips, not just `additive` — see
`mxlpy.surrogates.abstract_derivative`'s module docstring for why that's
possible here but not for `KineticModelBuilder`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch import nn

import mxlpy
from mxlpy import KineticModelBuilder, OdeModelBuilder
from mxlpy.serialize import (
    model_to_dict,
    ode_model_from_dict,
    ode_model_to_dict,
    ode_nn_block_weights_files,
)
from mxlpy.surrogates._torch import DerivativeSurrogate
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from pathlib import Path


def _dense_softplus_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.Softplus(),
        nn.Linear(3, 1),
    )


def _decay(x: float, k: float) -> float:
    # A named, module-level function (not a lambda): `fn_to_sympy_expr`
    # recovers a rate function's source via `inspect`, which needs the
    # function's own statement to be independently parseable — a lambda
    # split across a chained multi-line `.add_diff_eq(...)` call doesn't
    # satisfy that, a plain `def` always does.
    return -k * x


def _make_model() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=_decay, args=["x", "k"], initial_value=2.0)
    )


def _make_model_with_surrogate(mechanism: str = "relative_multiply") -> OdeModelBuilder:
    model = _make_model()
    surrogate = DerivativeSurrogate(
        model=_dense_softplus_model(),
        args=["x", "k"],
        targets=["x"],
        mechanism=mechanism,  # type: ignore[arg-type]
    )
    return model.add_surrogate("corr_block", surrogate)


def test_ode_model_to_dict_has_kind_and_no_reactions() -> None:
    data = ode_model_to_dict(_make_model(), model_id="m")
    assert data["kind"] == "ode"
    assert data["spec_version"] == "1.0"
    assert "reactions" not in data["model"]
    assert "fn" in data["model"]["variables"]["x"]
    assert "value" in data["model"]["variables"]["x"]


def test_ode_model_from_dict_round_trips_dxdt() -> None:
    model = _make_model()
    restored = ode_model_from_dict(ode_model_to_dict(model, model_id="m"))
    assert restored(0.0, [2.0]) == pytest.approx(model(0.0, [2.0]))


def test_exportable_surrogate_produces_nn_blocks_section_with_chosen_mechanism() -> None:
    model = _make_model_with_surrogate(mechanism="relative_multiply")
    data = ode_model_to_dict(model, model_id="m")
    block = data["model"]["nn_blocks"]["corr_block"]
    assert block["inputs"] == ["x", "k"]
    assert block["targets"] == ["x"]
    assert block["trained"] is True
    assert block["weights_ref"] == "corr_block.weights.json"
    # relative_multiply: Mul(ode, Add(1, nde)) -- distinguishes it from the
    # kinetic path, which can only ever export "additive".
    assert block["mechanism"]["type"] == "Mul"


def test_ode_nn_block_weights_files_shapes_match_layers() -> None:
    model = _make_model_with_surrogate()
    files = ode_nn_block_weights_files(model)
    assert set(files.keys()) == {"corr_block.weights.json"}
    weights = files["corr_block.weights.json"]
    assert len(weights["w1"]) == 3
    assert len(weights["w1"][0]) == 2
    assert len(weights["w2"]) == 1
    assert len(weights["w2"][0]) == 3


@pytest.mark.parametrize("mechanism", ["additive", "relative_multiply", "multiply"])
def test_save_load_round_trips_predictions_for_every_mechanism(
    tmp_path: Path, mechanism: str
) -> None:
    model = _make_model_with_surrogate(mechanism=mechanism)
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")

    assert (tmp_path / "corr_block.weights.json").exists()

    reloaded = mxlpy.load(path)
    assert isinstance(reloaded, OdeModelBuilder)
    reloaded_surrogate = reloaded._surrogates["corr_block"]
    assert reloaded_surrogate.mechanism == mechanism
    assert reloaded_surrogate.args == ["x", "k"]
    assert reloaded_surrogate.targets == ["x"]

    original_dxdt = model(0.0, [2.0])
    reloaded_dxdt = reloaded(0.0, [2.0])
    np.testing.assert_allclose(reloaded_dxdt, original_dxdt, rtol=1e-6)


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


def test_load_dispatches_kinetic_documents_without_kind_field(tmp_path: Path) -> None:
    """A document with no `kind` field (pre-existing files, written before this field existed) is treated as kinetic."""
    kinetic_model = KineticModelBuilder().add_variable("x", 1.0)
    data = model_to_dict(kinetic_model, model_id="m")
    del data["kind"]
    path = tmp_path / "model.mxl.json"
    path.write_text(json.dumps(data))

    reloaded = mxlpy.load(path)
    assert isinstance(reloaded, KineticModelBuilder)
