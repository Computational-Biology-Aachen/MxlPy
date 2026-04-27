from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch", exc_type=ImportError)

import torch
from torch import nn

from mxlpy.nn._torch import train


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


def test_train_full(features_targets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    features, targets = features_targets
    model = SimpleModel(n_inputs=2, n_outputs=2)
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    losses = train(
        model=model,
        features=features.to_numpy(),
        targets=targets.to_numpy(),
        epochs=3,
        batch_size=None,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_fn=loss,
    )

    assert isinstance(losses, pd.Series)
    assert len(losses) == 3  # 3 epochs
    assert losses.dtype == float

    # Check if loss is decreasing or stable
    if len(losses) > 1:
        assert losses.iloc[-1] <= losses.iloc[0] * 1.1  # Allow small fluctuations
