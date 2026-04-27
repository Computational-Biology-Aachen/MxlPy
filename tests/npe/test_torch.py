from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch", exc_type=ImportError)

import torch
from torch import nn

from mxlpy.npe._torch import (
    SteadyState,
    TimeCourse,
    train_steady_state,
)


class SimpleModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleTimeModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, features)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Reshape to combine batch and seq dimensions
        flattened = x.reshape(-1, x.shape[-1])
        # Apply linear layer
        out = self.linear(flattened)
        # Reshape back to separate batch and seq dimensions
        reshaped = out.reshape(batch_size, seq_len, -1)
        # Average over sequence dimension
        return torch.mean(reshaped, dim=1)


@pytest.fixture
def ss_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    targets = pd.DataFrame(
        {
            "p1": [2.0, 4.0, 6.0, 8.0, 10.0],
            "p2": [0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    return features, targets


@pytest.fixture
def time_course_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create a consistent time course dataset
    time_points = [0.0, 1.0, 2.0]
    experiments = ["exp1", "exp2", "exp3"]  # Add a third experiment

    # Create a MultiIndex DataFrame for features
    index = pd.MultiIndex.from_product(
        [experiments, time_points], names=["experiment", "time"]
    )

    features = pd.DataFrame(
        {
            "f1": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        index=index,
    )

    # Create a target DataFrame
    targets = pd.DataFrame(
        {
            "p1": [2.0, 4.0, 6.0],
            "p2": [0.2, 0.4, 0.6],
        },
        index=pd.Index(experiments, name="experiment"),
    )

    return features, targets


def test_torch_ss_estimator(ss_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    features, targets = ss_data
    model = SimpleModel(n_inputs=2, n_outputs=2)

    estimator = SteadyState(model=model, parameter_names=["p1", "p2"])

    predictions = estimator.predict(features)
    assert predictions.shape == (5, 2)
    assert list(predictions.columns) == ["p1", "p2"]


def test_torch_time_course_estimator(
    time_course_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = time_course_data
    model = SimpleTimeModel(n_inputs=2, n_outputs=2)

    estimator = TimeCourse(model=model, parameter_names=["p1", "p2"])

    predictions = estimator.predict(features)
    assert predictions.shape == (3, 2)  # 3 experiments, 2 parameters
    assert list(predictions.columns) == ["p1", "p2"]


def test_train_torch_ss_estimator(ss_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    features, targets = ss_data

    estimator, losses = train_steady_state(
        features=features, targets=targets, epochs=5, batch_size=2
    )

    assert isinstance(estimator, SteadyState)
    assert estimator.parameter_names == ["p1", "p2"]
    assert isinstance(losses, pd.Series)
    assert len(losses) == 5

    predictions = estimator.predict(features)
    assert predictions.shape == (5, 2)
    assert list(predictions.columns) == ["p1", "p2"]


def test_train_torch_ss_estimator_with_custom_model(
    ss_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, targets = ss_data
    model = SimpleModel(n_inputs=2, n_outputs=2)

    estimator, losses = train_steady_state(
        features=features, targets=targets, epochs=5, model=model
    )

    assert isinstance(estimator, SteadyState)
    assert estimator.model is model
    assert estimator.parameter_names == ["p1", "p2"]
    assert isinstance(losses, pd.Series)
    assert len(losses) == 5
