from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from mxlpy.nn._equinox import MLP, LossFn, SoftplusMLP, mean_abs_error
from mxlpy.nn._equinox import train as _train
from mxlpy.surrogates.abstract import (
    AbstractSurrogate,
    NNBlockExport,
    nn_block_activation_softplus,
    nn_block_mechanism_additive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mxlpy.types import Derived

__all__ = ["Surrogate", "Trainer", "surrogate_from_nn_block", "train"]


def surrogate_from_nn_block(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> Surrogate:
    """Reconstruct an equinox :class:`Surrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`Surrogate.to_nn_block_export`, mirroring
    `mxlpy.surrogates._torch.surrogate_from_nn_block`'s reconstruction
    (same "fresh output name" `f"{name}_{target}"` derivation and unit
    stoichiometry, since this is the same kinetic `AbstractSurrogate`
    shape, just with a `SoftplusMLP` model instead of a torch one).
    """
    layers_spec = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    outputs = [f"{name}_{target}" for target in targets]

    dummy = SoftplusMLP(
        n_inputs=len(inputs),
        neurons_per_layer=[cast("int", layer["width"]) for layer in layers_spec],
        key=jax.random.PRNGKey(0),
    )
    linears: list[eqx.nn.Linear] = []
    for i, linear in enumerate(dummy.layers, start=1):
        w = jnp.asarray(weights[f"w{i}"], dtype=jnp.float32)
        b = jnp.asarray(weights[f"b{i}"], dtype=jnp.float32)
        linears.append(
            eqx.tree_at(lambda lin: (lin.weight, lin.bias), linear, (w, b))
        )
    model = eqx.tree_at(lambda m: m.layers, dummy, linears)

    return Surrogate(
        model=model,
        args=inputs,
        outputs=outputs,
        stoichiometries={
            output: {target: 1.0} for output, target in zip(outputs, targets, strict=True)
        },
    )


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    """Surrogate model using PyTorch.

    Attributes
    ----------
    model
        PyTorch neural network model.

    Methods
    -------
        predict: Predict outputs based on input data using the PyTorch model.

    """

    model: eqx.Module

    def to_nn_block_export(self) -> NNBlockExport | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, if it's representable.

        Requires, structurally:

        - ``self.model`` is a :class:`mxlpy.nn._equinox.SoftplusMLP` — the
          one equinox architecture whose activation is known by
          construction rather than introspected (see that class's
          docstring); any other `eqx.Module` (including the default
          `MLP`, which hardcodes ReLU) declines.
        - every `eqx.nn.Linear` layer has a bias (`use_bias=True`).
        - every output has stoichiometry ``{compound: 1.0}`` against a
          reaction named after that same output — see
          :meth:`mxlpy.surrogates._torch.Surrogate.to_nn_block_export`'s
          identical requirement and rationale.

        Returns ``None`` when any of the above doesn't hold.
        """
        if not isinstance(self.model, SoftplusMLP):
            return None
        linears = self.model.layers
        if len(linears) == 0:
            return None
        if any(not linear.use_bias for linear in linears):
            return None

        targets: list[str] = []
        for output in self.outputs:
            stoich = self.stoichiometries.get(output)
            if stoich is None or len(stoich) != 1:
                return None
            ((compound, factor),) = stoich.items()
            if factor != 1:
                return None
            targets.append(compound)

        layers = [{"type": "dense", "width": lin.out_features} for lin in linears]
        weights: dict[str, list[object]] = {}
        for i, lin in enumerate(linears, start=1):
            weights[f"w{i}"] = np.asarray(lin.weight).tolist()
            weights[f"b{i}"] = np.asarray(lin.bias).tolist()

        spec: dict[str, object] = {
            "inputs": list(self.args),
            "layers": layers,
            "seed": 0,
            "targets": targets,
            "trained": True,
            "scale": 1.0,
            "mechanism": nn_block_mechanism_additive(),
            "activation": {
                "name": "softplus",
                "expression": nn_block_activation_softplus(),
            },
        }
        return NNBlockExport(spec=spec, weights=weights)

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data using the PyTorch model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        dict[str, float]
            Dictionary mapping output variable names to predicted values.

        """
        # One has to implement __call__ on eqx.Module, so this should
        # always exist. Should really be abstract on eqx.Module
        #
        # A jax array has no .numpy() method (unlike torch's) — np.asarray
        # is the actual jax-array -> numpy-array conversion.
        return np.asarray(self.model(y))  # type: ignore[operator]

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> dict[str, float]:
        """Predict outputs based on input data.

        Parameters
        ----------
        args
            Mapping of input names to their values.

        Returns
        -------
        dict[str, float]
            Mapping of output names to predicted values.

        """
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
        """Train the surrogate model.

        Parameters
        ----------
        epochs
            Number of training epochs.
        batch_size
            Size of mini-batches for training. None for full-batch.

        Returns
        -------
        Self
            The trainer instance for method chaining.

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
        return pd.concat(self.losses)

    def get_surrogate(
        self,
        surrogate_args: list[str] | None = None,
        surrogate_outputs: list[str] | None = None,
        surrogate_stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
    ) -> Surrogate:
        """Create a surrogate from the trained model.

        Parameters
        ----------
        surrogate_args
            Names of input arguments for the surrogate.
        surrogate_outputs
            Names of output arguments from the surrogate.
        surrogate_stoichiometries
            Mapping of reaction names to stoichiometry dicts.

        Returns
        -------
        Surrogate
            Configured surrogate model.

        """
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

    Examples
    --------
        >>> train_torch_surrogate(
        ...     features,
        ...     targets,
        ...     epochs=100,
        ...     surrogate_inputs=["x1", "x2"],
        ...     surrogate_stoichiometries={
        ...         "v1": {"x1": -1, "x2": 1, "ATP": -1},
        ...     },
        ...)surrogate_stoichiometries

    Parameters
    ----------
    features
        DataFrame containing the input features for training.
    targets
        DataFrame containing the target values for training.
    epochs
        Number of training epochs.
    surrogate_args
        Names of inputs arguments for the surrogate model.
    surrogate_outputs
        Names of output arguments from the surrogate.
    surrogate_stoichiometries
        Mapping of variables to their stoichiometries
    batch_size
        Size of mini-batches for training (None for full-batch).
    model
        Predefined neural network model (None to use default MLP features-50-50-output).
    optimizer
        Optimizer class to use for training (default: optax.GradientTransformation).
    device
        Device to run the training on (default: DefaultDevice).
    loss_fn
        Custom loss function or instance of torch loss object

    Returns
    -------
    tuple[TorchSurrogate, pd.Series]
        Trained surrogate model and loss history.

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
