"""Tests for the equinox surrogate backend, including nn_blocks export/import.

Mirrors `tests/surrogates/test_torch.py`'s coverage, plus the nn_blocks
(mxl-schemas) round-trip coverage `tests/test_serialize_nn_blocks.py` has for
the torch backend. Unlike torch/keras, an arbitrary `eqx.Module`'s activation
function isn't introspectable (it's just Python code inside `__call__`), so
export only recognizes `mxlpy.nn._equinox.SoftplusMLP` specifically — see
that class's docstring and `Surrogate.to_mxl_json`'s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

pytest.importorskip("equinox", exc_type=ImportError)

import equinox as eqx
import jax

if TYPE_CHECKING:
    import pandas as pd

from mxlpy.nn._equinox import MLP, SoftplusMLP
from mxlpy.surrogates._equinox import Surrogate, surrogate_from_mxl_json


def _softplus_model() -> SoftplusMLP:
    return SoftplusMLP(n_inputs=2, neurons_per_layer=[3, 1], key=jax.random.PRNGKey(0))


def test_equinox_surrogate_predict_raw() -> None:
    model = SoftplusMLP(n_inputs=2, neurons_per_layer=[2], key=jax.random.PRNGKey(0))
    surrogate = Surrogate(model=model, args=["x1", "x2"], outputs=["y1", "y2"])

    result = surrogate.predict_raw(np.array([1.0, 0.1], dtype=np.float32))

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_equinox_surrogate_predict() -> None:
    linear = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0))
    linear = eqx.tree_at(
        lambda lin: (lin.weight, lin.bias),
        linear,
        (np.array([[1.0, 1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)),
    )
    model = SoftplusMLP(n_inputs=2, neurons_per_layer=[1], key=jax.random.PRNGKey(0))
    model = eqx.tree_at(lambda m: m.layers, model, [linear])

    surrogate = Surrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["r1"],
        stoichiometries={"r1": {"x1": -1.0, "x2": 1.0}},
    )

    result = surrogate.predict({"x1": 1.0, "x2": 2.0})
    # A single-layer network has no hidden layer, so softplus (applied
    # "after every layer except the last") never triggers here — this is
    # a bare linear combination: 1*1.0 + 1*2.0 + 0.0.
    assert result["r1"] == pytest.approx(3.0, rel=1e-5)


def test_exportable_surrogate_produces_dense_softplus_additive_spec() -> None:
    surrogate = Surrogate(
        model=_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_mxl_json()
    assert export is not None
    spec = cast(dict[str, Any], export.spec)
    weights = cast(dict[str, Any], export.weights)
    assert spec["inputs"] == ["x", "y"]
    assert spec["targets"] == ["x"]
    assert spec["layers"] == [
        {"type": "dense", "width": 3},
        {"type": "dense", "width": 1},
    ]
    assert spec["activation"]["name"] == "softplus"
    assert spec["mechanism"]["type"] == "Add"
    assert set(weights.keys()) == {"w1", "b1", "w2", "b2"}
    assert len(weights["w1"]) == 3
    assert len(weights["w1"][0]) == 2


def test_non_unit_stoichiometry_is_not_exportable() -> None:
    surrogate = Surrogate(
        model=_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 2.0}},
    )
    assert surrogate.to_mxl_json() is None


def test_default_mlp_activation_is_not_exportable() -> None:
    # MLP hardcodes ReLU, which has no nn_blocks counterpart, and — unlike
    # torch's/keras's alternatives — there is no way to introspect an
    # arbitrary eqx.Module's activation, so this always declines.
    model = MLP(n_inputs=2, neurons_per_layer=[3, 1], key=jax.random.PRNGKey(0))
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    assert surrogate.to_mxl_json() is None


def test_surrogate_from_mxl_json_round_trips_predictions() -> None:
    surrogate = Surrogate(
        model=_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_mxl_json()
    assert export is not None

    reconstructed = surrogate_from_mxl_json("corr_block", export.spec, export.weights)

    probe: dict[str, float | pd.Series | pd.DataFrame] = {"x": 0.3, "y": -0.7}
    original = surrogate.predict(probe)["corr"]
    # Reconstruction can't reuse the original output name "corr" (add_surrogate
    # requires it to be a fresh, model-wide-unique id) — it derives a
    # block-scoped one instead, matching the torch backend's convention.
    reconstructed_value = reconstructed.predict(probe)["corr_block_x"]
    assert reconstructed_value == pytest.approx(original, rel=1e-5)
    assert reconstructed.args == ["x", "y"]
    assert reconstructed.stoichiometries == {"corr_block_x": {"x": 1.0}}
