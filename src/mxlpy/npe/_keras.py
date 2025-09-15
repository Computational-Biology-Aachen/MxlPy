from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import keras
import numpy as np
import pandas as pd

from mxlpy.nn._keras import LSTM, MLP, train
from mxlpy.npe.abstract import AbstractEstimator

__all__ = [
    "DefaultLoss",
    "DefaultOptimizer",
    "LossFn",
    "Optimizer",
    "SteadyState",
    "SteadyStateTrainer",
    "TimeCourse",
    "TimeCourseTrainer",
    "train_steady_state",
    "train_time_course",
]

type Optimizer = keras.optimizers.Optimizer | str
type LossFn = keras.losses.Loss | str

DefaultOptimizer = keras.optimizers.Adam()
DefaultLoss = keras.losses.MeanAbsoluteError()


@dataclass(kw_only=True)
class SteadyState(AbstractEstimator):
    """Estimator for steady state data using Keras models."""

    model: keras.Model

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        return pd.DataFrame(
            self.model.predict(features),
            columns=self.parameter_names,
            dtype=float,
        )


@dataclass(kw_only=True)
class TimeCourse(AbstractEstimator):
    """Estimator for time course data using Keras models."""

    model: keras.Model

    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
        idx = cast(pd.MultiIndex, features.index)
        features_ = (
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
        return pd.DataFrame(
            self.model.predict(features_),
            columns=self.parameter_names,
            dtype=float,
        )


@dataclass
class SteadyStateTrainer:
    """Trainer for steady state data using Keras models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    model: keras.Model
    optimizer: Optimizer
    losses: list[pd.Series]
    loss_fn: LossFn

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model: keras.Model | None = None,
        optimizer: Optimizer = DefaultOptimizer,
        loss: LossFn = DefaultLoss,
    ) -> None:
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            model: Predefined neural network model (None to use default MLP)
            optimizer: Optimizer class to use for training (default: Adam)
            loss: Loss function

        """
        self.features = features
        self.targets = targets

        if model is None:
            n_hidden = max(2 * len(features.columns) * len(targets.columns), 10)
            n_outputs = len(targets.columns)
            model = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[n_hidden, n_hidden, n_outputs],
            )
        self.model = model
        model.compile(optimizer=cast(str, optimizer), loss=loss)

        self.loss_fn = loss
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
        losses = train(
            model=self.model,
            features=self.features,
            targets=self.targets,
            epochs=epochs,
            batch_size=batch_size,
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
    """Trainer for time course data using Keras models."""

    features: pd.DataFrame
    targets: pd.DataFrame
    model: keras.Model
    optimizer: Optimizer

    losses: list[pd.Series]
    loss_fn: LossFn

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model: keras.Model | None = None,
        optimizer: Optimizer = DefaultOptimizer,
        loss: LossFn = DefaultLoss,
    ) -> None:
        """Initialize the trainer with features, targets, and model.

        Args:
            features: DataFrame containing the input features for training
            targets: DataFrame containing the target values for training
            model: Predefined neural network model (None to use default LSTM)
            optimizer: Optimizer class to use for training
            loss: Loss function

        """
        self.features = features
        self.targets = targets

        if model is None:
            model = LSTM(
                n_inputs=len(features.columns),
                n_outputs=len(targets.columns),
                n_hidden=1,
            )
        self.model = model
        model.compile(optimizer=cast(str, optimizer), loss=loss)
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
        losses = train(
            model=self.model,
            features=np.swapaxes(
                self.features.to_numpy().reshape(
                    (len(self.targets), -1, len(self.features.columns))
                ),
                axis1=0,
                axis2=1,
            ),
            targets=self.targets,
            epochs=epochs,
            batch_size=batch_size,
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
    model: keras.Model | None = None,
    optimizer: Optimizer = DefaultOptimizer,
    loss: LossFn = DefaultLoss,
) -> tuple[SteadyState, pd.Series]:
    """Train a Keras steady state estimator.

    This function trains a neural network model to estimate steady state data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_keras_ss_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        model: Predefined neural network model (None to use default MLP)
        optimizer: Optimizer class to use for training (default: Adam)
        loss: Loss function for the training

    Returns:
        tuple[KerasTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = SteadyStateTrainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
        loss=loss,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()


def train_time_course(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    batch_size: int | None = None,
    model: keras.Model | None = None,
    optimizer: Optimizer = DefaultOptimizer,
    loss: LossFn = DefaultLoss,
) -> tuple[TimeCourse, pd.Series]:
    """Train a Keras time course estimator.

    This function trains a neural network model to estimate time course data
    using the provided features and targets. It supports both full-batch and
    mini-batch training.

    Examples:
        >>> train_keras_time_course_estimator(features, targets, epochs=100)

    Args:
        features: DataFrame containing the input features for training
        targets: DataFrame containing the target values for training
        epochs: Number of training epochs
        batch_size: Size of mini-batches for training (None for full-batch)
        model: Predefined neural network model (None to use default LSTM)
        optimizer: Optimizer class to use for training (default: Adam)
        loss: Loss function for the training

    Returns:
        tuple[KerasTimeSeriesEstimator, pd.Series]: Trained estimator and loss history

    """
    trainer = TimeCourseTrainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
        loss=loss,
    ).train(epochs=epochs, batch_size=batch_size)

    return trainer.get_estimator(), trainer.get_loss()
