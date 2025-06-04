from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from mxlpy.surrogates import torch as ts


class SimpleModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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
