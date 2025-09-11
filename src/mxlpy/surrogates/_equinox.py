from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from mxlpy.nn._equinox import MLP, LossFn, mean_abs_error
from mxlpy.nn._equinox import train as _train
from mxlpy.types import AbstractSurrogate, Derived

if TYPE_CHECKING:
    import equinox as eqx

__all__ = ["Surrogate", "Trainer", "train"]


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    """Surrogate model using PyTorch.

    Attributes:
        model: PyTorch neural network model.

    Methods:
        predict: Predict outputs based on input data using the PyTorch model.

    """

    model: eqx.Module

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data using the PyTorch model.

        Args:
            y: Input data as a numpy array.

        Returns:
            dict[str, float]: Dictionary mapping output variable names to predicted values.

        """
        # One has to implement __call__ on eqx.Module, so this should
        # always exist. Should really be abstract on eqx.Module
        return self.model(y).numpy()  # type: ignore

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> dict[str, float]:
        """Predict outputs based on input data."""
        return dict(
            zip(
                self.outputs,
                self.predict_raw(np.array([args[arg] for arg in self.args])),
                strict=True,
            )
        )


@dataclass(init=False)
class Trainer:
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
        self.features = features
        self.targets = targets

        if model is None:
            model = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[50, 50, len(targets.columns)],
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
        return pd.concat(self.losses)

    def get_surrogate(
        self,
        surrogate_args: list[str] | None = None,
        surrogate_outputs: list[str] | None = None,
        surrogate_stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
    ) -> Surrogate:
        return Surrogate(
            model=self.model,
            args=surrogate_args if surrogate_args is not None else [],
            outputs=surrogate_outputs if surrogate_outputs is not None else [],
            stoichiometries=surrogate_stoichiometries
            if surrogate_stoichiometries is not None
            else {},
        )


def train(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    epochs: int,
    surrogate_args: list[str] | None = None,
    surrogate_outputs: list[str] | None = None,
    surrogate_stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
    batch_size: int | None = None,
    model: eqx.Module | None = None,
    optimizer: optax.GradientTransformation | None = None,
    loss_fn: LossFn = mean_abs_error,
) -> tuple[Surrogate, pd.Series]:
    """Train a PyTorch surrogate model.

    Examples:
        >>> train_torch_surrogate(
        ...     features,
        ...     targets,
        ...     epochs=100,
        ...     surrogate_inputs=["x1", "x2"],
        ...     surrogate_stoichiometries={
        ...         "v1": {"x1": -1, "x2": 1, "ATP": -1},
        ...     },
        ...)surrogate_stoichiometries

    Args:
        features: DataFrame containing the input features for training.
        targets: DataFrame containing the target values for training.
        epochs: Number of training epochs.
        surrogate_args: Names of inputs arguments for the surrogate model.
        surrogate_outputs: Names of output arguments from the surrogate.
        surrogate_stoichiometries: Mapping of variables to their stoichiometries
        batch_size: Size of mini-batches for training (None for full-batch).
        model: Predefined neural network model (None to use default MLP features-50-50-output).
        optimizer: Optimizer class to use for training (default: optax.GradientTransformation).
        device: Device to run the training on (default: DefaultDevice).
        loss_fn: Custom loss function or instance of torch loss object

    Returns:
        tuple[TorchSurrogate, pd.Series]: Trained surrogate model and loss history.

    """
    trainer = Trainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    ).train(
        epochs=epochs,
        batch_size=batch_size,
    )
    return trainer.get_surrogate(
        surrogate_args=surrogate_args,
        surrogate_outputs=surrogate_outputs,
        surrogate_stoichiometries=surrogate_stoichiometries,
    ), trainer.get_loss()
