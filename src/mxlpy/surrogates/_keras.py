from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import keras
import numpy as np
import pandas as pd

from mxlpy.nn._keras import MLP
from mxlpy.nn._keras import train as _train
from mxlpy.types import AbstractSurrogate, Array, Derived

__all__ = [
    "DefaultLoss",
    "DefaultOptimizer",
    "LossFn",
    "Optimizer",
    "Surrogate",
    "Trainer",
    "train",
]

type Optimizer = keras.optimizers.Optimizer | str
type LossFn = keras.losses.Loss | str

DefaultOptimizer = keras.optimizers.Adam()
DefaultLoss = keras.losses.MeanAbsoluteError()


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    model: keras.Model

    def predict_raw(self, y: Array) -> Array:
        return np.atleast_1d(np.squeeze(self.model.predict(y)))

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
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
    model: keras.Model
    optimizer: Optimizer | str
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
        self.features = features
        self.targets = targets
        if model is None:
            model = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[50, 50, len(targets.columns)],
            )
        self.model = model
        model.compile(optimizer=cast(str, optimizer), loss=loss)

        self.losses = []

    def train(self, epochs: int, batch_size: int | None = None) -> Self:
        losses = _train(
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
    model: keras.Model | None = None,
    optimizer: Optimizer = DefaultOptimizer,
    loss: LossFn = DefaultLoss,
) -> tuple[Surrogate, pd.Series]:
    trainer = Trainer(
        features=features,
        targets=targets,
        model=model,
        optimizer=optimizer,
        loss=loss,
    ).train(
        epochs=epochs,
        batch_size=batch_size,
    )
    return trainer.get_surrogate(
        surrogate_args=surrogate_args,
        surrogate_outputs=surrogate_outputs,
        surrogate_stoichiometries=surrogate_stoichiometries,
    ), trainer.get_loss()
