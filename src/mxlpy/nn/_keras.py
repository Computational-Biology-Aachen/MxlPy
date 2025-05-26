from __future__ import annotations

from typing import TYPE_CHECKING, cast

import keras
import pandas as pd
from tqdm.keras import TqdmCallback

if TYPE_CHECKING:
    from mxlpy.types import Array

__all__ = [
    "LSTM",
    "MLP",
    "train",
]


def train(
    model: keras.Model,
    features: pd.DataFrame | Array,
    targets: pd.DataFrame | Array,
    epochs: int,
    batch_size: int | None,
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
    history = model.fit(
        features,
        targets,
        batch_size=batch_size,
        epochs=epochs,
        verbose=cast(str, 0),
        callbacks=[TqdmCallback()],
    )
    return pd.Series(history.history["loss"])


def MLP(  # noqa: N802
    n_inputs: int,
    neurons_per_layer: list[int],
    activation: None = None,
    output_activation: None = None,
) -> keras.Sequential:
    """Multilayer Perceptron (MLP) for surrogate modeling and neural posterior estimation.

    Methods:
        forward: Forward pass through the neural network.

    """
    model = keras.Sequential([keras.Input(shape=(n_inputs,))])
    for neurons in neurons_per_layer[:-1]:
        model.add(keras.layers.Dense(neurons, activation=activation))
    model.add(keras.layers.Dense(neurons_per_layer[-1], activation=output_activation))
    return model


def LSTM(  # noqa: N802
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
) -> keras.Sequential:
    """Long Short-Term Memory (LSTM) network for time series modeling.

    Methods:
        forward: Forward pass through the neural network.

    """
    model = keras.Sequential(
        [
            keras.Input(
                shape=(n_inputs),
            )
        ]
    )
    model.add(keras.layers.LSTM(n_hidden))
    model.add(keras.layers.Dense(n_outputs))
    return model
