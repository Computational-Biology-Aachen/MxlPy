"""Neural network architectures.

This module provides implementations of neural network architectures used for mechanistic learning.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

type LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.adam import Adam

    from mxlpy.types import Array

__all__ = [
    "DefaultDevice",
    "LSTM",
    "LossFn",
    "MLP",
    "train",
]

DefaultDevice = torch.device("cpu")


def train(
    model: nn.Module,
    features: Array,
    targets: Array,
    epochs: int,
    optimizer: Adam,
    device: torch.device,
    batch_size: int | None,
    loss_fn: LossFn,
) -> pd.Series:
    """Train the neural network using mini-batch gradient descent.

    Args:
        model: Neural network model to train.
        features: Input features as a tensor.
        targets: Target values as a tensor.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        device: torch device
        batch_size: Size of mini-batches for training.
        loss_fn: Loss function

    Returns:
        pd.Series: Series containing the training loss history.

    """
    losses = {}

    data = TensorDataset(
        torch.tensor(features.astype(np.float32), dtype=torch.float32, device=device),
        torch.tensor(targets.astype(np.float32), dtype=torch.float32, device=device),
    )
    data_loader = DataLoader(
        data,
        batch_size=len(features) if batch_size is None else batch_size,
        shuffle=True,
    )

    for i in tqdm.trange(epochs):
        epoch_loss = 0
        for xb, yb in data_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        losses[i] = epoch_loss / len(data_loader.dataset)  # type: ignore
    return pd.Series(losses, dtype=float)


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) for surrogate modeling and neural posterior estimation.

    Attributes:
        net: Sequential neural network model.

    Methods:
        forward: Forward pass through the neural network.

    """

    def __init__(
        self,
        n_inputs: int,
        neurons_per_layer: list[int],
        activation: Callable | None = None,
        output_activation: Callable | None = None,
    ) -> None:
        """Initializes the MLP with the given number of inputs and list of (hidden) layers.

        Args:
            n_inputs: The number of input features.
            neurons_per_layer: Number of neurons per layer
            n_outputs: A list containing the number of neurons in hidden and output layer.
            activation: The activation function to be applied after each hidden layer (default nn.ReLU)
            output_activation: The activation function to be applied after the final (output) layer

        For instance, MLP(10, layers = [50, 50, 10]) initializes a neural network with the following architecture:
        - Linear layer with `n_inputs` inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 10 outputs

        The weights of the linear layers are initialized with a normal distribution
        (mean=0, std=0.1) and the biases are initialized to 0.

        """
        super().__init__()
        self.layers = neurons_per_layer
        self.activation = nn.ReLU() if activation is None else activation
        self.output_activation = output_activation

        levels = []
        previous_neurons = n_inputs

        for neurons in self.layers[:-1]:
            levels.append(nn.Linear(previous_neurons, neurons))
            if self.activation:
                levels.append(self.activation)
            previous_neurons = neurons

        # Output layer
        levels.append(nn.Linear(previous_neurons, self.layers[-1]))
        if self.output_activation:
            levels.append(self.output_activation)

        self.net = nn.Sequential(*levels)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.net(x)


class LSTM(nn.Module):
    """Default LSTM neural network model for time-series approximation."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
    ) -> None:
        """Initializes the neural network model.

        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output features.
            n_hidden (int): Number of hidden units in the LSTM layer.

        """
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(n_inputs, n_hidden)
        self.to_out = nn.Linear(n_hidden, n_outputs)

        nn.init.normal_(self.to_out.weight, mean=0, std=0.1)
        nn.init.constant_(self.to_out.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        # lstm_out, (hidden_state, cell_state)
        _, (hn, _) = self.lstm(x)
        return cast(torch.Tensor, self.to_out(hn[-1]))  # Use last hidden state
