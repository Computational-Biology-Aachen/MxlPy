"""Neural network architectures.

This module provides implementations of neural network architectures used for mechanistic learning.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
import tqdm
from jaxtyping import Array, PyTree
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from collections.abc import Callable

    import optax


__all__ = [
    "LSTM",
    "LossFn",
    "MLP",
    "cosine_similarity",
    "mean_abs_error",
    "mean_absolute_percentage",
    "mean_error",
    "mean_squared_error",
    "mean_squared_logarithmic",
    "rms_error",
    "train",
]


###############################################################################
# Loss functions
###############################################################################

type LossFn = Callable[[eqx.Module, Array, Array], Array]


@eqx.filter_jit
def mean_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean error."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(pred - true)


@eqx.filter_jit
def mean_squared_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean squared error."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.square(pred - true))


@eqx.filter_jit
def rms_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate root mean square error."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.sqrt(jnp.mean(jnp.square(pred - true)))


@eqx.filter_jit
def mean_abs_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean absolute error."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.abs(pred - true))


@eqx.filter_jit
def mean_absolute_percentage(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean absolute percentag error."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return 100 * jnp.mean(jnp.abs((true - pred) / pred))


@eqx.filter_jit
def mean_squared_logarithmic(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate root mean square error between model and data."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.square(jnp.log(pred + 1) - jnp.log(true + 1)))


@eqx.filter_jit
def cosine_similarity(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate root mean square error between model and data."""
    pred = jax.vmap(model)(inp)  # type: ignore
    return -jnp.sum(jnp.linalg.norm(pred, 2) * jnp.linalg.norm(true, 2))


###############################################################################
# Training routines
###############################################################################


def train(
    model: eqx.Module,
    features: Array,
    targets: Array,
    epochs: int,
    optimizer: optax.GradientTransformation,
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
        torch.tensor(features.astype(np.float32), dtype=torch.float32),
        torch.tensor(targets.astype(np.float32), dtype=torch.float32),
    )
    data_loader = DataLoader(
        data,
        batch_size=len(features) if batch_size is None else batch_size,
        shuffle=True,
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: eqx.Module,
        opt_state: PyTree,
        x: Array,
        y: Array,
    ) -> tuple[eqx.Module, Array, Array]:
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for i in tqdm.trange(epochs):
        epoch_loss = 0
        for xb, yb in data_loader:
            model, opt_state, train_loss = make_step(
                model,
                opt_state,
                xb.numpy(),
                yb.numpy(),
            )
            epoch_loss += train_loss * xb.size(0)
        losses[i] = epoch_loss / len(data_loader.dataset)  # type: ignore
    return pd.Series(losses, dtype=float)


###############################################################################
# Actual models
###############################################################################


class MLP(eqx.Module):
    """Multilayer Perceptron (MLP) for surrogate modeling and neural posterior estimation.

    Attributes:
        net: Sequential neural network model.

    Methods:
        forward: Forward pass through the neural network.

    """

    layers: list

    def __init__(
        self,
        n_inputs: int,
        neurons_per_layer: list[int],
        key: Array,
    ) -> None:
        """Initializes the MLP with the given number of inputs and list of (hidden) layers.

        Args:
            n_inputs: The number of input features.
            neurons_per_layer: Number of neurons per layer
            n_outputs: A list containing the number of neurons in hidden and output layer.
            key: jax.random.PRNGKey(SEED) for initial parameters

        For instance, MLP(10, layers = [50, 50, 10]) initializes a neural network with the following architecture:
        - Linear layer with `n_inputs` inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 50 outputs
        - ReLU activation
        - Linear layer with 50 inputs and 10 outputs

        The weights of the linear layers are initialized with a normal distribution
        (mean=0, std=0.1) and the biases are initialized to 0.

        """
        keys = iter(jax.random.split(key, len(neurons_per_layer)))
        previous_neurons = n_inputs
        layers = []
        for neurons in neurons_per_layer:
            layers.append(eqx.nn.Linear(previous_neurons, neurons, key=next(keys)))
            previous_neurons = neurons
        self.layers = layers

    def __call__(self, x: Array) -> Array:
        """Forward pass through the neural network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class LSTM(eqx.Module):
    """Default LSTM neural network model for time-series approximation."""

    lstm_cell: eqx.nn.LSTMCell
    n_hidden: int
    linear: eqx.nn.Linear

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        key: Array,
    ) -> None:
        """Initializes the LSTM neural network model.

        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output features.
            n_hidden (int): Number of hidden units in the LSTM layer.
            key (Array): JAX random key for initialization.

        """
        k1, k2 = jax.random.split(key, 2)
        self.lstm_cell = eqx.nn.LSTMCell(n_inputs, n_hidden, key=k1)
        self.n_hidden = n_hidden
        self.linear = eqx.nn.Linear(n_hidden, n_outputs, key=k2)

    def __call__(
        self,
        x: Array,
        *,
        h: Array | None = None,
        c: Array | None = None,
    ) -> Array:
        """Forward pass through the LSTM network.

        Args:
            x: Input tensor of shape (seq_len, batch_size, n_inputs).
            h: Optional initial hidden state (batch_size, n_hidden).
            c: Optional initial cell state (batch_size, n_hidden).

        Returns:
            Output tensor of shape (seq_len, batch_size, n_outputs).

        """
        seq_len, batch_size, _ = x.shape
        if h is None:
            h = jnp.zeros((batch_size, self.n_hidden))
        if c is None:
            c = jnp.zeros((batch_size, self.n_hidden))

        outputs = []
        for t in range(seq_len):
            h, c = self.lstm_cell(x[t], (h, c))
            outputs.append(h)
        outputs = jnp.stack(outputs, axis=0)
        return jax.vmap(self.linear)(outputs)
