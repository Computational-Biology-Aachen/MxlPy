"""Tests for the keras surrogate backend, including nn_blocks export/import.

Mirrors `tests/surrogates/test_torch.py`'s coverage, plus the nn_blocks
(mxl-schemas) round-trip coverage `tests/test_serialize_nn_blocks.py` has for
the torch backend — `Surrogate.to_mxl_json`/`surrogate_from_mxl_json` here
mirror that module's implementation exactly, just against
`keras.Sequential`/`Dense` instead of `torch.nn.Sequential`/`Linear`.

`to_mxl_json` raises `SerializationError` (rather than returning `None`) once
it's committed to trying — see `mxlpy.surrogates.abstract.AbstractSurrogate.
to_mxl_json`'s docstring for the raise-vs-`None` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

pytest.importorskip("keras", exc_type=ImportError)

import keras

from mxlpy.surrogates._keras import (
    OdeSurrogate,
    Surrogate,
    ode_surrogate_from_mxl_json,
    surrogate_from_mxl_json,
)
from mxlpy.types import SerializationError

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


def _dense_relu_sigmoid_model() -> keras.Sequential:
    keras.utils.set_random_seed(0)
    return keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(3, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
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
    export = surrogate.to_mxl_json()
    spec = cast(dict[str, Any], export.spec)
    weights = cast(dict[str, Any], export.weights)
    assert spec["inputs"] == ["x", "y"]
    assert spec["targets"] == ["x"]
    layers = cast(list[dict[str, Any]], spec["layers"])
    assert [layer["width"] for layer in layers] == [3, 1]
    assert layers[0]["activation"]["name"] == "softplus"
    assert "activation" not in layers[1]
    assert "activation" not in spec
    assert spec["mechanism"]["type"] == "Add"
    assert set(weights.keys()) == {"w1", "b1", "w2", "b2"}
    assert len(weights["w1"]) == 3
    assert len(weights["w1"][0]) == 2


def test_final_layer_may_carry_a_non_identity_activation() -> None:
    """Per-layer activation (mxl-schemas, per-layer move) allows a non-identity final layer."""
    surrogate = Surrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_mxl_json()
    layers = cast(list[dict[str, Any]], export.spec["layers"])
    assert layers[0]["activation"]["name"] == "relu"
    assert layers[1]["activation"]["name"] == "sigmoid"


def test_non_unit_stoichiometry_raises() -> None:
    surrogate = Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 2.0}},
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_unrecognized_activation_raises() -> None:
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(3, activation="elu"),
            keras.layers.Dense(1),
        ]
    )
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_functional_api_model_raises() -> None:
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(1)(keras.layers.Dense(3, activation="softplus")(inputs))
    model = keras.Model(inputs=inputs, outputs=outputs)
    surrogate = Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_bias_free_layer_raises() -> None:
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
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_surrogate_from_mxl_json_round_trips_predictions() -> None:
    surrogate = Surrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_mxl_json()

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


def test_ode_surrogate_predict() -> None:
    model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1)])
    model.layers[0].set_weights(
        [np.array([[1.0], [1.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    surrogate = OdeSurrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["o1"],
        targets={"o1": ["x"]},
    )

    result = surrogate.predict({"x1": 1.0, "x2": 2.0})
    assert result["o1"] == pytest.approx(3.0)


def test_ode_surrogate_untargeted_output_raises_on_export() -> None:
    """An output absent from `targets` has no nn_blocks shape — export requires one target per output."""
    surrogate = OdeSurrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        targets={},
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_ode_surrogate_to_mxl_json_round_trips_predictions() -> None:
    surrogate = OdeSurrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        targets={"corr": ["x"]},
    )
    export = surrogate.to_mxl_json()
    layers = cast(list[dict[str, Any]], export.spec["layers"])
    assert layers[0]["activation"]["name"] == "relu"
    assert layers[1]["activation"]["name"] == "sigmoid"

    reconstructed = ode_surrogate_from_mxl_json("corr_block", export.spec, export.weights)

    probe: dict[str, float | pd.Series | pd.DataFrame] = {"x": 0.3, "y": -0.7}
    original = surrogate.predict(probe)["corr"]
    reconstructed_value = reconstructed.predict(probe)["corr_block_x"]
    assert reconstructed_value == pytest.approx(original, rel=1e-5)
    assert reconstructed.targets == {"corr_block_x": ["x"]}
