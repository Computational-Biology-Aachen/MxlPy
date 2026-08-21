from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import keras
import numpy as np
import pandas as pd

from mxlpy.nn._keras import MLP
from mxlpy.nn._keras import train as _train
from mxlpy.surrogates.abstract import (
    ACTIVATION_BUILDERS,
    AbstractSurrogate,
    SurrogateJson,
    mxl_json_mechanism_additive,
)
from mxlpy.surrogates.abstract_ode import AbstractOdeSurrogate, OdeSurrogateJson
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mxlpy.types import Array, Derived

__all__ = [
    "DefaultLoss",
    "DefaultOptimizer",
    "LossFn",
    "OdeSurrogate",
    "Optimizer",
    "Surrogate",
    "Trainer",
    "ode_surrogate_from_mxl_json",
    "surrogate_from_mxl_json",
    "train",
]

type Optimizer = keras.optimizers.Optimizer | str
type LossFn = keras.losses.Loss | str

DefaultOptimizer = keras.optimizers.Adam()
DefaultLoss = keras.losses.MeanAbsoluteError()


def _dense_layers(model: keras.Model) -> list[keras.layers.Dense] | None:
    """The plain `keras.Sequential` stack of `Dense` layers this model wraps, if any.

    Mirrors `mxlpy.surrogates._torch._dense_sequential`'s role: any other
    model shape (functional-API model, a non-`Dense` layer type) isn't
    representable as a mxl-schemas `nn_blocks` entry.
    """
    if not isinstance(model, keras.Sequential):
        return None
    layers = list(model.layers)
    if any(not isinstance(layer, keras.layers.Dense) for layer in layers):
        return None
    return cast("list[keras.layers.Dense]", layers)


def _export_dense_layers(
    model: keras.Model,
) -> tuple[list[dict[str, object]], dict[str, list[object]]]:
    """Export `model`'s architecture/weights as mxl-schemas `layers`/weights-sidecar content, or raise.

    Shared by :meth:`Surrogate.to_mxl_json`/:meth:`OdeSurrogate.to_mxl_json`
    — see `mxlpy.surrogates._torch._export_dense_layers`'s identical role.
    Keras's `Dense.kernel` is shaped `[in_features, out_features]` — the
    transpose of the schema's `[out_features, in_features]`
    (PyTorch/equinox-native) convention — so every weight matrix is
    transposed here, and un-transposed again in `_revive_dense_layers`.
    """
    layers_ = _dense_layers(model)
    if layers_ is None or len(layers_) == 0:
        msg = (
            f"{type(model).__name__} isn't representable as a mxl-schemas "
            "nn_blocks entry: not a plain keras.Sequential of Dense "
            "layers — a functional-API model or any other layer type "
            "can't be expressed this way."
        )
        raise SerializationError(msg)
    if any(not layer.use_bias for layer in layers_):
        msg = (
            f"{type(model).__name__} isn't representable as a mxl-schemas "
            "nn_blocks entry: every layer must have a bias "
            "(use_bias=True) — the schema's w{i}/b{i} weights sidecar "
            "always has both."
        )
        raise SerializationError(msg)

    layers_spec: list[dict[str, object]] = []
    weights: dict[str, list[object]] = {}
    for i, layer in enumerate(layers_, start=1):
        layer_spec: dict[str, object] = {"type": "dense", "width": layer.units}
        name = keras.activations.serialize(layer.activation)
        if name != "linear":
            builder = ACTIVATION_BUILDERS.get(cast("str", name))
            if builder is None:
                msg = (
                    f"Layer {i} uses unsupported activation {name!r} — "
                    f"recognized: {sorted(ACTIVATION_BUILDERS)}."
                )
                raise SerializationError(msg)
            layer_spec["activation"] = {"name": name, "expression": builder()}
        layers_spec.append(layer_spec)
        kernel, bias = layer.get_weights()
        weights[f"w{i}"] = np.asarray(kernel).T.tolist()
        weights[f"b{i}"] = np.asarray(bias).tolist()

    return layers_spec, weights


def _revive_dense_layers(
    layers: list[dict[str, object]],
    in_features: int,
    weights: Mapping[str, list[object]],
) -> keras.Sequential:
    """Build a keras `Sequential` from a mxl-schemas `layers` array and its weights sidecar.

    Inverse of :func:`_export_dense_layers`. Each layer's optional
    `activation` is verified structurally against
    `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS` (matching the exact
    `expression` a known preset produces, not just trusting `name`)
    before being passed to `Dense` as its native string identifier; an
    unrecognized or non-matching activation raises.
    """
    model = keras.Sequential([keras.Input(shape=(in_features,))])
    for i, layer in enumerate(layers, start=1):
        out_features = cast("int", layer["width"])
        activation_spec = cast("dict[str, object] | None", layer.get("activation"))
        activation_name: str | None = None
        if activation_spec is not None:
            name = cast("str", activation_spec["name"])
            builder = ACTIVATION_BUILDERS.get(name)
            if builder is None or builder() != activation_spec.get("expression"):
                msg = f"Layer {i}: unrecognized or non-matching activation {activation_spec!r}."
                raise SerializationError(msg)
            activation_name = name
        model.add(keras.layers.Dense(out_features, activation=activation_name))

    for i, dense in enumerate(model.layers, start=1):
        kernel = np.asarray(weights[f"w{i}"], dtype=np.float32).T
        bias = np.asarray(weights[f"b{i}"], dtype=np.float32)
        dense.set_weights([kernel, bias])
    return model


def surrogate_from_mxl_json(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> Surrogate:
    """Reconstruct a keras :class:`Surrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`Surrogate.to_mxl_json`, mirroring
    `mxlpy.surrogates._torch.surrogate_from_mxl_json`'s reconstruction
    (same "fresh output name" `f"{name}_{target}"` derivation and unit
    stoichiometry).
    """
    layers_spec = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    outputs = [f"{name}_{target}" for target in targets]

    model = _revive_dense_layers(layers_spec, len(inputs), weights)

    return Surrogate(
        model=model,
        args=inputs,
        outputs=outputs,
        stoichiometries={
            output: {target: 1.0} for output, target in zip(outputs, targets, strict=True)
        },
    )


def ode_surrogate_from_mxl_json(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> OdeSurrogate:
    """Reconstruct a keras :class:`OdeSurrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`OdeSurrogate.to_mxl_json`. Mirrors
    `mxlpy.surrogates._torch.ode_surrogate_from_mxl_json`'s `name`-derived
    output renaming exactly.
    """
    layers_spec = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    outputs = [f"{name}_{target}" for target in targets]

    model = _revive_dense_layers(layers_spec, len(inputs), weights)

    return OdeSurrogate(
        model=model,
        args=inputs,
        outputs=outputs,
        targets={
            output: [target] for output, target in zip(outputs, targets, strict=True)
        },
    )


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    model: keras.Model

    def to_mxl_json(self) -> SurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Requires, structurally:

        - ``self.model`` is a plain `keras.Sequential` of `Dense` layers
          (see `_dense_layers`) — a functional-API model or any other
          layer type isn't representable at all.
        - every layer has a bias (`use_bias=True`, `Dense`'s own default).
        - every layer's activation is either `"linear"` (`Dense`'s own
          default — no `activation` field) or one of
          `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS`.
        - every output has stoichiometry ``{compound: 1.0}`` against a
          reaction named after that same output — see
          :meth:`mxlpy.surrogates._torch.Surrogate.to_mxl_json`'s
          identical requirement and rationale.

        Raises
        ------
        SerializationError
            If any of the above doesn't hold — see
            :meth:`mxlpy.surrogates.abstract.AbstractSurrogate.to_mxl_json`
            for why this raises rather than returning ``None``.

        """
        layers_spec, weights = _export_dense_layers(self.model)

        targets: list[str] = []
        for output in self.outputs:
            stoich = self.stoichiometries.get(output)
            if stoich is None or len(stoich) != 1:
                msg = (
                    f"Output {output!r} isn't representable as a "
                    "nn_blocks entry: needs exactly one stoichiometric "
                    f"target with coefficient 1 (got {stoich!r})."
                )
                raise SerializationError(msg)
            ((compound, factor),) = stoich.items()
            if factor != 1:
                msg = (
                    f"Output {output!r} has non-unit stoichiometry "
                    f"({factor!r}) against {compound!r} — a nn_blocks "
                    "entry always applies its whole output with an "
                    "implicit coefficient of 1."
                )
                raise SerializationError(msg)
            targets.append(compound)

        spec: dict[str, object] = {
            "inputs": list(self.args),
            "layers": layers_spec,
            "seed": 0,
            "targets": targets,
            "trained": True,
            "scale": 1.0,
            "mechanism": mxl_json_mechanism_additive(),
        }
        return SurrogateJson(spec=spec, weights=weights)

    def predict_raw(self, y: Array) -> Array:
        """Predict raw output from the Keras model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        Array
            Raw model prediction as a numpy array.

        """
        # keras.Model.predict expects a batched (2D) input, unlike this
        # method's callers (Surrogate.predict passes one flat 1D sample at
        # a time); np.squeeze then drops the resulting single-row batch
        # dimension back off.
        return np.atleast_1d(np.squeeze(self.model.predict(np.atleast_2d(y), verbose=0)))

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
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


@dataclass(kw_only=True)
class OdeSurrogate(AbstractOdeSurrogate):
    """Ode surrogate using Keras, for `OdeModelBuilder`."""

    model: keras.Model

    def to_mxl_json(self) -> OdeSurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Same structural requirements as :meth:`Surrogate.to_mxl_json`; see
        `mxlpy.surrogates._torch.OdeSurrogate.to_mxl_json`'s doc comment
        for why every output must still map to exactly one target
        diff_eq despite there being no stoichiometry-coefficient concept
        at the model level.

        Raises
        ------
        SerializationError
            If any of the above doesn't hold.

        """
        layers_spec, weights = _export_dense_layers(self.model)

        targets: list[str] = []
        for output in self.outputs:
            output_targets = self.targets.get(output)
            if output_targets is None or len(output_targets) != 1:
                msg = (
                    f"Output {output!r} isn't representable as a "
                    "nn_blocks entry: needs exactly one target diff_eq "
                    f"(got {output_targets!r}) — a nn_blocks entry's "
                    "`targets` has one entry per output."
                )
                raise SerializationError(msg)
            targets.append(output_targets[0])

        spec: dict[str, object] = {
            "inputs": list(self.args),
            "layers": layers_spec,
            "seed": 0,
            "targets": targets,
            "trained": True,
            "scale": 1.0,
            "mechanism": mxl_json_mechanism_additive(),
        }
        return OdeSurrogateJson(spec=spec, weights=weights)

    def predict_raw(self, y: Array) -> Array:
        """Predict raw output from the Keras model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        Array
            Raw model prediction as a numpy array, ordered like `self.outputs`.

        """
        return np.atleast_1d(np.squeeze(self.model.predict(np.atleast_2d(y), verbose=0)))

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
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
