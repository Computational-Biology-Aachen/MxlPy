"""Neural Network Parameter Estimation (NPE) Module.

This module provides classes and functions for training neural network models to estimate
parameters in metabolic models. It includes functionality for both steady-state and
time-series data.

Functions:
    train_torch_surrogate: Train a PyTorch surrogate model
    train_torch_time_course_estimator: Train a PyTorch time course estimator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from mxlpy.nn._equinox import LSTM, MLP, LossFn, mean_abs_error
from mxlpy.nn._equinox import train as _train
from mxlpy.types import AbstractEstimator

if TYPE_CHECKING:
    import equinox as eqx

__all__ = [
    "SteadyState",
    "SteadyStateTrainer",
    "TimeCourse",
    "TimeCourseTrainer",
    "train_steady_state",
    "train_time_course",
]


@dataclass(kw_only=True)
class SteadyState(AbstractEstimator):
    """Estimator for steady state data using PyTorch models."""

    model: eqx.Module

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        # One has to implement __call__ on eqx.Module, so this should
        # always exist. Should really be abstract on eqx.Module
        pred = jax.vmap(self.model)(jnp.array(features))  # type: ignore
        return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass(kw_only=True)
class TimeCourse(AbstractEstimator):
    """Estimator for time course data using PyTorch models."""

    model: eqx.Module

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        idx = cast(pd.MultiIndex, features.index)
        features_ = jnp.array(
            np.swapaxes(
                features.to_numpy().reshape(
                    (
                        len(idx.levels[0]),
                        len(idx.levels[1]),
                        len(features.columns),
                    )
                ),
                axis1=0,
                axis2=1,
            ),
        )
        # One has to implement __call__ on eqx.Module, so this should
        # always exist. Should really be abstract on eqx.Module
        pred = jax.vmap(self.model)(features_)  # type: ignore
        return pd.DataFrame(pred, columns=self.parameter_names)


@dataclass
class SteadyStateTrainer:
    """Trainer for steady state data using PyTorch models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    model: eqx.Module
    optimizer: optax.GradientTransformation
    losses: list[pd.Series]
    loss_fn: LossFn
    seed: int

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model: eqx.Module | None = None,
        optimizer: optax.GradientTransformation | None = None,
        loss_fn: LossFn = mean_abs_error,
        seed: int = 0,
    ) -> None:
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            model: Predefined neural network model (None to use default MLP)
            optimizer: Optimizer class to use for training (default: Adam)
            device: Device to run the training on (default: DefaultDevice)
            loss_fn: Loss function
            seed: seed of random initialisation

        """
        self.features = features
        self.targets = targets

        if model is None:
            n_hidden = max(2 * len(features.columns) * len(targets.columns), 10)
            n_outputs = len(targets.columns)
            model = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[n_hidden, n_hidden, n_outputs],
                key=jax.random.PRNGKey(seed),
            )
        self.model = model
        self.optimizer = (
            optax.adamw(learning_rate=0.001) if optimizer is None else optimizer
        )
        self.loss_fn = loss_fn
        self.losses = []
        self.seed = seed

    def train(
        self,
        epochs: int,
        batch_size: int | None = None,
    ) -> Self:
        """Train the model using the provided features and targets.

        Args:
            epochs: Number of training epochs
            batch_size: Size of mini-batches for training (None for full-batch)

        """
        losses = _train(
            model=self.model,
            features=jnp.array(self.features),
            targets=jnp.array(self.targets),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            loss_fn=self.loss_fn,
        )

        if len(self.losses) > 0:
            losses.index += self.losses[-1].index[-1]
        self.losses.append(losses)
        return self

    def get_loss(self) -> pd.Series:
        """Get the loss history of the training process."""
        return pd.concat(self.losses)

    def get_estimator(self) -> SteadyState:
        """Get the trained estimator."""
        return SteadyState(
            model=self.model,
            parameter_names=list(self.targets.columns),
        )


@dataclass
class TimeCourseTrainer:
    """Trainer for time course data using PyTorch models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    model: eqx.Module
    optimizer: optax.GradientTransformation
    losses: list[pd.Series]
    loss_fn: LossFn

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model: eqx.Module | None = None,
        optimizer: optax.GradientTransformation | None = None,
        loss_fn: LossFn = mean_abs_error,
    ) -> None:
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            model: Predefined neural network model (None to use default LSTM)
            optimizer: Optimizer class to use for training (default: Adam)
            device: Device to run the training on (default: DefaultDevice)
            loss_fn: Loss function

        """
        self.features = features
        self.targets = targets

        if model is None:
            model = LSTM(
                n_inputs=len(features.columns),
                n_outputs=len(targets.columns),
                n_hidden=1,
                key=jnp.array([]),
            )
        self.model = model
        self.optimizer = (
            optax.adamw(learning_rate=0.001) if optimizer is None else optimizer
        )
        self.loss_fn = loss_fn
        self.losses = []

    def train(
        self,
        epochs: int,
        batch_size: int | None = None,
    ) -> Self:
        """Train the model using the provided features and targets.

        Args:
            epochs: Number of training epochs
            batch_size: Size of mini-batches for training (None for full-batch)

        """
        losses = _train(
            model=self.model,
            features=jnp.array(
                np.swapaxes(
                    self.features.to_numpy().reshape(
                        (len(self.targets), -1, len(self.features.columns))
                    ),
                    axis1=0,
                    axis2=1,
                )
            ),
            targets=jnp.array(self.targets.to_numpy()),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            loss_fn=self.loss_fn,
        )

        if len(self.losses) > 0:
            losses.index += self.losses[-1].index[-1]
        self.losses.append(losses)
        return self

    def get_loss(self) -> pd.Series:
        """Get the loss history of the training process."""
        return pd.concat(self.losses)

    def get_estimator(self) -> TimeCourse:
        """Get the trained estimator."""
        return TimeCourse(
            model=self.model,
            parameter_names=list(self.targets.columns),
        )


def train_steady_state(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    model: eqx.Module | None = None,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[SteadyState, pd.Series]:
    """Train a PyTorch steady state estimator.

    This function trains a neural network model to estimate steady state data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_torch_ss_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        model: Predefined neural network model (None to use default MLP)
        optimizer: Optimizer class to use for training (default: Adam)
        device: Device to run the training on (default: DefaultDevice)

    Returns:
        tuple[TorchTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = SteadyStateTrainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()


def train_time_course(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    model: eqx.Module | None = None,
    optimizer: optax.GradientTransformation | None = None,
) -> tuple[TimeCourse, pd.Series]:
    """Train a PyTorch time course estimator.

    This function trains a neural network model to estimate time course data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_torch_time_course_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        model: Predefined neural network model (None to use default LSTM)
        optimizer: Optimizer class to use for training (default: Adam)
        device: Device to run the training on (default: DefaultDevice)

    Returns:
        tuple[TorchTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = TimeCourseTrainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()
