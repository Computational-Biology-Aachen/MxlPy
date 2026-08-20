"""Tests for the keras surrogate backend, including nn_blocks export/import.

Mirrors `tests/surrogates/test_torch.py`'s coverage, plus the nn_blocks
(mxl-schemas) round-trip coverage `tests/test_serialize_nn_blocks.py` has for
the torch backend — `Surrogate.to_nn_block_export`/`surrogate_from_nn_block`
here mirror that module's implementation exactly, just against
`keras.Sequential`/`Dense` instead of `torch.nn.Sequential`/`Linear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

pytest.importorskip("keras", exc_type=ImportError)

import keras

from mxlpy.surrogates._keras import Surrogate, surrogate_from_nn_block

if TYPE_CHECKING:
    import pandas as pd


def _dense_softplus_model() -> keras.Sequential:
    keras.utils.set_random_seed(0)
    return keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(3, activation="softplus"),
            keras.layers.Dense(1),
        ]
    )


def test_keras_surrogate_predict_raw() -> None:
    model = keras.Sequential(
        [keras.Input(shape=(2,)), keras.layers.Dense(2, activation=None)]
    )
    surrogate = Surrogate(model=model, args=["x1", "x2"], outputs=["y1", "y2"])

    result = surrogate.predict_raw(np.array([1.0, 0.1]))

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_keras_surrogate_predict() -> None:
    model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1)])
    model.layers[0].set_weights(
        [np.array([[1.0], [1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    surrogate = Surrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["r1"],
        stoichiometries={"r1": {"x1": -1.0, "x2": 1.0}},
    )

    result = surrogate.predict({"x1": 1.0, "x2": 2.0})
    assert result["r1"] == pytest.approx(3.0)


def test_exportable_surrogate_produces_dense_softplus_additive_spec() -> None:
    surrogate = Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_nn_block_export()
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
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 2.0}},
    )
    assert surrogate.to_nn_block_export() is None


def test_default_mlp_activation_is_not_exportable() -> None:
    # keras.layers.Dense's own default (activation=None, "linear") has no
    # nn_blocks counterpart, matching torch's/equinox's MLP (ReLU) both
    # similarly declining by default.
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(3),
            keras.layers.Dense(1),
        ]
    )
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    assert surrogate.to_nn_block_export() is None


def test_functional_api_model_is_not_exportable() -> None:
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(1)(keras.layers.Dense(3, activation="softplus")(inputs))
    model = keras.Model(inputs=inputs, outputs=outputs)
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    assert surrogate.to_nn_block_export() is None


def test_bias_free_layer_is_not_exportable() -> None:
    """A `use_bias=False` layer must decline (return `None`), not raise.

    Regression test: `to_nn_block_export` used to unpack
    `layer.get_weights()` into `(kernel, bias)` unconditionally — a
    bias-free layer's `get_weights()` returns a 1-element list, so this
    raised `ValueError` instead of returning `None`, breaking the
    documented "`None` is a normal, expected outcome, not a bug" contract
    every caller relies on (`mxlpy.surrogates.abstract.AbstractSurrogate.
    to_nn_block_export`'s docstring).
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(3, activation="softplus", use_bias=False),
            keras.layers.Dense(1, use_bias=False),
        ]
    )
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    assert surrogate.to_nn_block_export() is None


def test_surrogate_from_nn_block_round_trips_predictions() -> None:
    surrogate = Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_nn_block_export()
    assert export is not None

    reconstructed = surrogate_from_nn_block("corr_block", export.spec, export.weights)

    probe: dict[str, float | pd.Series | pd.DataFrame] = {"x": 0.3, "y": -0.7}
    original = surrogate.predict(probe)["corr"]
    # Reconstruction can't reuse the original output name "corr" (add_surrogate
    # requires it to be a fresh, model-wide-unique id) — it derives a
    # block-scoped one instead, matching the torch backend's convention.
    reconstructed_value = reconstructed.predict(probe)["corr_block_x"]
    assert reconstructed_value == pytest.approx(original, rel=1e-5)
    assert reconstructed.args == ["x", "y"]
    assert reconstructed.stoichiometries == {"corr_block_x": {"x": 1.0}}
