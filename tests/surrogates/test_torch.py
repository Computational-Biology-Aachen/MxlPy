from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", exc_type=ImportError)

import torch
from torch import nn

from mxlpy.surrogates import torch as ts
from mxlpy.types import SerializationError


class SimpleModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _dense_softplus_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.Softplus(),
        nn.Linear(3, 1),
    )


def _dense_relu_sigmoid_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
        nn.Sigmoid(),
    )


@pytest.fixture
def features_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    targets = pd.DataFrame(
        {
            "y1": [2.0, 4.0, 6.0, 8.0, 10.0],
            "y2": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    return features, targets


def test_torch_surrogate_predict_raw() -> None:
    model = SimpleModel(n_inputs=2, n_outputs=2)
    surrogate = ts.Surrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["y1", "y2"],
        stoichiometries={},
    )

    input_data = np.array([[1.0, 0.1], [2.0, 0.2]])

    result = surrogate.predict_raw(input_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)  # 2 samples, 2 outputs


def test_train_torch_surrogate_with_default_model(
    features_targets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = features_targets

    surrogate, losses = ts.train(
        features=features,
        targets=targets,
        epochs=3,
        batch_size=None,  # Use full batch
    )

    assert isinstance(surrogate, ts.Surrogate)
    assert isinstance(surrogate.model, nn.Module)
    assert isinstance(losses, pd.Series)
    assert len(losses) == 3  # 3 epochs


def test_train_torch_surrogate_with_custom_model(
    features_targets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = features_targets
    model = SimpleModel(n_inputs=2, n_outputs=2)

    surrogate, losses = ts.train(
        features=features,
        targets=targets,
        epochs=3,
        model=model,
    )

    assert isinstance(surrogate, ts.Surrogate)
    assert surrogate.model is model
    assert isinstance(losses, pd.Series)
    assert len(losses) == 3


def test_train_torch_surrogate_with_batch(
    features_targets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = features_targets

    surrogate, losses = ts.train(
        features=features,
        targets=targets,
        epochs=3,
        batch_size=2,
    )

    assert isinstance(surrogate, ts.Surrogate)
    assert isinstance(surrogate.model, nn.Module)
    assert isinstance(losses, pd.Series)
    assert len(losses) == 3


def test_train_torch_surrogate_with_args_and_stoichiometries(
    features_targets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = features_targets
    surrogate_args = ["x1", "x2"]
    surrogate_stoichiometries = {"r1": {"x1": -1.0, "x2": 1.0}}

    surrogate, losses = ts.train(
        features=features,
        targets=targets,
        epochs=3,
        surrogate_args=surrogate_args,
        surrogate_stoichiometries=surrogate_stoichiometries,  # type: ignore
    )

    assert isinstance(surrogate, ts.Surrogate)
    assert surrogate.args == surrogate_args
    assert surrogate.stoichiometries == surrogate_stoichiometries
    assert isinstance(losses, pd.Series)
    assert len(losses) == 3


def test_torch_surrogate_predict() -> None:
    model = SimpleModel(n_inputs=2, n_outputs=1)
    model.linear.weight.data = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    model.linear.bias.data = torch.tensor([0.0], dtype=torch.float32)

    surrogate = ts.Surrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["r1"],
        stoichiometries={"r1": {"x1": -1.0, "x2": 1.0}},
    )

    # When passed as numpy array
    inputs_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    outputs_np = surrogate.predict_raw(inputs_np)
    assert outputs_np.shape == (2, 1)
    assert np.isclose(outputs_np[0, 0], 3.0)  # 1.0 + 2.0
    assert np.isclose(outputs_np[1, 0], 7.0)  # 3.0 + 4.0


def test_exportable_surrogate_produces_dense_softplus_additive_spec() -> None:
    surrogate = ts.Surrogate(
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
    surrogate = ts.Surrogate(
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
    surrogate = ts.Surrogate(
        model=_dense_softplus_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 2.0}},
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_unrecognized_activation_raises() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.ELU(), nn.Linear(3, 1))
    surrogate = ts.Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_lstm_backed_model_raises() -> None:
    class LstmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return out

    surrogate = ts.Surrogate(
        model=LstmModel(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_bias_free_layer_raises() -> None:
    model = nn.Sequential(
        nn.Linear(2, 3, bias=False),
        nn.Softplus(),
        nn.Linear(3, 1, bias=False),
    )
    surrogate = ts.Surrogate(
        model=model, args=["x", "y"], outputs=["corr"], stoichiometries={"corr": {"x": 1.0}}
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_surrogate_from_mxl_json_round_trips_predictions() -> None:
    surrogate = ts.Surrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        stoichiometries={"corr": {"x": 1.0}},
    )
    export = surrogate.to_mxl_json()

    reconstructed = ts.surrogate_from_mxl_json("corr_block", export.spec, export.weights)

    probe: dict[str, float | pd.Series | pd.DataFrame] = {"x": 0.3, "y": -0.7}
    original = surrogate.predict(probe)["corr"]
    # Reconstruction can't reuse the original output name "corr" (add_surrogate
    # requires it to be a fresh, model-wide-unique id) — it derives a
    # block-scoped one instead.
    reconstructed_value = reconstructed.predict(probe)["corr_block_x"]
    assert reconstructed_value == pytest.approx(original, rel=1e-5)
    assert reconstructed.args == ["x", "y"]
    assert reconstructed.stoichiometries == {"corr_block_x": {"x": 1.0}}


def test_ode_surrogate_predict() -> None:
    model = SimpleModel(n_inputs=2, n_outputs=1)
    model.linear.weight.data = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    model.linear.bias.data = torch.tensor([0.0], dtype=torch.float32)

    surrogate = ts.OdeSurrogate(
        model=model,
        args=["x1", "x2"],
        outputs=["o1"],
        targets={"o1": ["x"]},
    )

    result = surrogate.predict({"x1": 1.0, "x2": 2.0})
    assert result["o1"] == pytest.approx(3.0)


def test_ode_surrogate_untargeted_output_raises_on_export() -> None:
    """An output absent from `targets` has no nn_blocks shape — export requires one target per output."""
    surrogate = ts.OdeSurrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        targets={},
    )
    with pytest.raises(SerializationError):
        surrogate.to_mxl_json()


def test_ode_surrogate_to_mxl_json_round_trips_predictions() -> None:
    surrogate = ts.OdeSurrogate(
        model=_dense_relu_sigmoid_model(),
        args=["x", "y"],
        outputs=["corr"],
        targets={"corr": ["x"]},
    )
    export = surrogate.to_mxl_json()
    layers = cast(list[dict[str, Any]], export.spec["layers"])
    assert layers[0]["activation"]["name"] == "relu"
    assert layers[1]["activation"]["name"] == "sigmoid"

    reconstructed = ts.ode_surrogate_from_mxl_json("corr_block", export.spec, export.weights)

    probe: dict[str, float | pd.Series | pd.DataFrame] = {"x": 0.3, "y": -0.7}
    original = surrogate.predict(probe)["corr"]
    reconstructed_value = reconstructed.predict(probe)["corr_block_x"]
    assert reconstructed_value == pytest.approx(original, rel=1e-5)
    assert reconstructed.targets == {"corr_block_x": ["x"]}
