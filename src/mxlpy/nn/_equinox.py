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
    """Calculate mean error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Mean error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(pred - true)


@eqx.filter_jit
def mean_squared_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean squared error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Mean squared error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.square(pred - true))


@eqx.filter_jit
def rms_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate root mean square error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Root mean square error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.sqrt(jnp.mean(jnp.square(pred - true)))


@eqx.filter_jit
def mean_abs_error(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean absolute error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Mean absolute error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.abs(pred - true))


@eqx.filter_jit
def mean_absolute_percentage(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean absolute percentage error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Mean absolute percentage error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return 100 * jnp.mean(jnp.abs((true - pred) / pred))


@eqx.filter_jit
def mean_squared_logarithmic(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate mean squared logarithmic error.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Mean squared logarithmic error scalar.

    """
    pred = jax.vmap(model)(inp)  # type: ignore
    return jnp.mean(jnp.square(jnp.log(pred + 1) - jnp.log(true + 1)))


@eqx.filter_jit
def cosine_similarity(model: eqx.Module, inp: Array, true: Array) -> Array:
    """Calculate negative cosine similarity.

    Parameters
    ----------
    model
        Neural network model.
    inp
        Input features.
    true
        Ground truth target values.

    Returns
    -------
    Array
        Negative cosine similarity scalar.

    """
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

    Parameters
    ----------
    model
        Neural network model to train.
    features
        Input features as a tensor.
    targets
        Target values as a tensor.
    epochs
        Number of training epochs.
    optimizer
        Optimizer for training.
    device
        torch device
    batch_size
        Size of mini-batches for training.
    loss_fn
        Loss function

    Returns
    -------
    pd.Series
        Series containing the training loss history.

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
        """Perform a single optimisation step.

        Parameters
        ----------
        model
            Neural network model.
        opt_state
            Current optimizer state.
        x
            Input batch.
        y
            Target batch.

        Returns
        -------
        tuple[eqx.Module, Array, Array]
            Updated model, optimizer state, and loss value.

        """
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

        Parameters
        ----------
        n_inputs : int
            Number of input features.
        n_outputs : int
            Number of output features.
        n_hidden : int
            Number of hidden units in the LSTM layer.
        key : Array
            JAX random key for initialization.

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

        Parameters
        ----------
        x
            Input tensor of shape (seq_len, batch_size, n_inputs).
        h
            Optional initial hidden state (batch_size, n_hidden).
        c
            Optional initial cell state (batch_size, n_hidden).

        Returns
        -------
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
