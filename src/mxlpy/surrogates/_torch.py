from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.adam import Adam

from mxlpy.nn._torch import MLP, DefaultDevice, LossFn, mean_abs_error
from mxlpy.nn._torch import train as _train
from mxlpy.surrogates.abstract import (
    ACTIVATION_BUILDERS,
    AbstractSurrogate,
    SurrogateJson,
    mxl_json_mechanism_additive,
)
from mxlpy.surrogates.abstract_ode import AbstractOdeSurrogate, OdeSurrogateJson
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch.optim.optimizer import ParamsT

    from mxlpy.types import Derived


__all__ = [
    "OdeSurrogate",
    "Surrogate",
    "Trainer",
    "ode_surrogate_from_mxl_json",
    "surrogate_from_mxl_json",
    "train",
]

# mxl-schemas activation.name <-> the torch.nn.Module type it corresponds to.
# Any activation in `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS` is
# recognized here — a layer using anything else (or a model shape that
# isn't a plain Linear/activation-alternating nn.Sequential at all) makes
# `to_mxl_json` raise, not silently decline.
_ACTIVATION_MODULE_BY_NAME: dict[str, type[nn.Module]] = {
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
_ACTIVATION_NAME_BY_MODULE: dict[type[nn.Module], str] = {
    v: k for k, v in _ACTIVATION_MODULE_BY_NAME.items()
}


def _dense_sequential(model: nn.Module) -> nn.Sequential | None:
    """The plain `nn.Sequential` this model wraps, if any.

    Handles `mxlpy.nn._torch.MLP` (whose real layer stack lives on `.net`)
    and a bare hand-built `nn.Sequential` alike; any other module shape
    (LSTM-backed, custom `forward`, ...) returns `None`.
    """
    net = getattr(model, "net", model)
    return net if isinstance(net, nn.Sequential) else None


def _parse_dense_layers(
    net: nn.Sequential,
) -> list[tuple[nn.Linear, nn.Module | None]] | None:
    """Split `net` into `(Linear, activation-or-None)` pairs, one per layer.

    Each `Linear` is paired with whatever non-`Linear` module immediately
    follows it (that layer's activation) — absent if the next module is
    another `Linear` or there is no next module (identity/linear output),
    matching mxl-schemas' per-layer, independently-optional `activation`
    exactly. `None` (not an empty list) means `net` isn't representable at
    all — doesn't start with a `Linear`, or has two non-`Linear` modules in
    a row.
    """
    modules = list(net)
    layers: list[tuple[nn.Linear, nn.Module | None]] = []
    i = 0
    while i < len(modules):
        linear = modules[i]
        if not isinstance(linear, nn.Linear):
            return None
        i += 1
        activation: nn.Module | None = None
        if i < len(modules) and not isinstance(modules[i], nn.Linear):
            activation = modules[i]
            i += 1
        layers.append((linear, activation))
    return layers or None


def _export_dense_layers(
    model: nn.Module,
) -> tuple[list[dict[str, object]], dict[str, list[object]]]:
    """Export `model`'s architecture/weights as mxl-schemas `layers`/weights-sidecar content, or raise.

    Shared by :meth:`Surrogate.to_mxl_json`/:meth:`OdeSurrogate.to_mxl_json`
    — everything about exporting layers is identical between the two;
    they differ only in how `outputs` maps onto the schema's `targets`
    (kinetic: via unit stoichiometry; ode: via `self.targets` directly).
    """
    net = _dense_sequential(model)
    if net is None:
        msg = (
            f"{type(model).__name__} isn't representable as a mxl-schemas "
            "nn_blocks entry: not a plain nn.Sequential (or "
            "mxlpy.nn._torch.MLP) of Linear/activation layers — an "
            "LSTM-backed or custom-forward model can't be expressed this "
            "way."
        )
        raise SerializationError(msg)

    parsed = _parse_dense_layers(net)
    if parsed is None:
        msg = (
            f"{type(model).__name__} isn't representable as a mxl-schemas "
            "nn_blocks entry: its nn.Sequential doesn't consist of "
            "Linear layers, each optionally followed by a single "
            "activation module."
        )
        raise SerializationError(msg)

    layers_spec: list[dict[str, object]] = []
    weights: dict[str, list[object]] = {}
    for i, (linear, activation_module) in enumerate(parsed, start=1):
        if linear.bias is None:
            msg = f"Layer {i} has no bias (bias=False) — not representable."
            raise SerializationError(msg)
        layer_spec: dict[str, object] = {
            "type": "dense",
            "width": linear.out_features,
        }
        if activation_module is not None:
            name = _ACTIVATION_NAME_BY_MODULE.get(type(activation_module))
            if name is None:
                msg = (
                    f"Layer {i} uses unsupported activation "
                    f"{type(activation_module).__name__!r} — recognized: "
                    f"{sorted(ACTIVATION_BUILDERS)}."
                )
                raise SerializationError(msg)
            layer_spec["activation"] = {
                "name": name,
                "expression": ACTIVATION_BUILDERS[name](),
            }
        layers_spec.append(layer_spec)
        weights[f"w{i}"] = linear.weight.detach().cpu().numpy().tolist()
        weights[f"b{i}"] = linear.bias.detach().cpu().numpy().tolist()

    return layers_spec, weights


def _revive_dense_sequential(
    layers: list[dict[str, object]],
    in_features: int,
    weights: Mapping[str, list[object]],
) -> nn.Sequential:
    """Build a torch `nn.Sequential` from a mxl-schemas `layers` array and its weights sidecar.

    Inverse of :func:`_export_dense_layers`. Each layer's optional
    `activation` is verified structurally against
    `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS` (matching the exact
    `expression` a known preset produces, not just trusting `name`) before
    being revived as the matching `nn.Module`; an unrecognized or
    non-matching activation raises.
    """
    modules: list[nn.Module] = []
    for i, layer in enumerate(layers, start=1):
        out_features = cast("int", layer["width"])
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(weights[f"w{i}"], dtype=torch.float32))
            linear.bias.copy_(torch.tensor(weights[f"b{i}"], dtype=torch.float32))
        modules.append(linear)

        activation = cast("dict[str, object] | None", layer.get("activation"))
        if activation is not None:
            name = cast("str", activation["name"])
            module_type = _ACTIVATION_MODULE_BY_NAME.get(name)
            builder = ACTIVATION_BUILDERS.get(name)
            if module_type is None or builder is None or builder() != activation.get(
                "expression"
            ):
                msg = f"Layer {i}: unrecognized or non-matching activation {activation!r}."
                raise SerializationError(msg)
            modules.append(module_type())

        in_features = out_features
    return nn.Sequential(*modules)


def surrogate_from_mxl_json(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> Surrogate:
    """Reconstruct a torch :class:`Surrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`Surrogate.to_mxl_json` — only handles shapes that
    method can actually produce (dense layers, a recognized activation
    per layer, additive mechanism); :func:`mxlpy.serialize.model_from_dict`
    checks the mechanism before calling this, so it isn't re-validated
    here.

    `outputs` (the surrogate's own "reaction" names, one per target) are
    *not* set equal to `targets`: the schema has no field for an MxlPy
    reaction name distinct from its target compound, but `add_surrogate`
    still requires each output name to be a fresh, model-wide-unique id —
    reusing a target's own name would collide with that pre-existing
    variable the moment `add_surrogate` tries to register it. `name` (the
    block's own key, already guaranteed unique by the caller having
    registered it as an `nn_blocks` id) is prefixed onto each target to
    build a safe, deterministic output name instead — invisible to the
    schema either way, since `targets` is all `nn_blocks` records.
    """
    layers = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    outputs = [f"{name}_{target}" for target in targets]

    model = _revive_dense_sequential(layers, len(inputs), weights)

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
    """Reconstruct a torch :class:`OdeSurrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`OdeSurrogate.to_mxl_json`. Mirrors
    :func:`surrogate_from_mxl_json`'s `name`-derived output renaming
    exactly — an ode surrogate's outputs are fresh, model-wide-unique ids
    too (`mxlpy.surrogates.abstract_ode`'s module docstring), and a
    target's own name is already taken by the existing diff_eq.
    """
    layers = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    outputs = [f"{name}_{target}" for target in targets]

    model = _revive_dense_sequential(layers, len(inputs), weights)

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
    """Surrogate model using PyTorch.

    Attributes
    ----------
    model
        PyTorch neural network model.

    Methods
    -------
        predict: Predict outputs based on input data using the PyTorch model.

    """

    model: torch.nn.Module

    def to_mxl_json(self) -> SurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Requires, structurally:

        - ``self.model`` wraps a plain `nn.Sequential` of `Linear` layers,
          each optionally followed by a single recognized activation
          module (handles :class:`mxlpy.nn._torch.MLP` via its `.net`, or
          a hand-built `nn.Sequential` directly) — an LSTM-backed or
          custom-`forward` model isn't representable at all.
        - every activation used is one of
          `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS`.
        - every output has stoichiometry ``{compound: 1.0}`` against a
          reaction named after that same output — the only shape
          equivalent to an `nn_blocks` correction (a bare coefficient of
          1, one target per output); anything else (a non-unit or
          `Derived` coefficient, an output feeding more than one compound)
          raises rather than guess at a lossy mapping.

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
            # Only meaningful for a freshly Glorot-initialized (untrained)
            # block; an already-trained export always carries real weights
            # via `weights_ref` instead, so this is an inert placeholder.
            "seed": 0,
            "targets": targets,
            "trained": True,
            # No separate scale knob on the MxlPy side — a trained
            # network's own weights already encode the right magnitude, so
            # this is the neutral (no-op) multiplier, not an initial guess.
            "scale": 1.0,
            "mechanism": mxl_json_mechanism_additive(),
        }
        return SurrogateJson(spec=spec, weights=weights)

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
        with torch.no_grad():
            return self.model(
                torch.tensor(y, dtype=torch.float32),
            ).numpy()

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


@dataclass(kw_only=True)
class OdeSurrogate(AbstractOdeSurrogate):
    """Ode surrogate using PyTorch, for `OdeModelBuilder`.

    See :mod:`mxlpy.surrogates.abstract_ode`'s module docstring for why
    this is a separate, orthogonal type from :class:`Surrogate` rather
    than a shared base — same reasoning, applied one level down at the
    torch-backend layer.

    Attributes
    ----------
    model
        PyTorch neural network model.

    """

    model: torch.nn.Module

    def to_mxl_json(self) -> OdeSurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Same structural requirements as :meth:`Surrogate.to_mxl_json`
        (dense `Linear`/recognized-activation layers), and — despite
        having no stoichiometry-coefficient constraint at the model level
        — the *schema*'s `targets` is still one entry per output, so
        every output must map to exactly one target diff_eq to be
        representable: a plain-derived output (absent from `self.targets`)
        or one feeding more than one diff_eq has no `nn_blocks` shape to
        export as.

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

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data using the PyTorch model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        np.ndarray
            Array of predicted values, ordered like `self.outputs`.

        """
        with torch.no_grad():
            return self.model(
                torch.tensor(y, dtype=torch.float32),
            ).numpy()

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
    model: nn.Module
    optimizer: Adam
    device: torch.device
    losses: list[pd.Series]
    loss_fn: LossFn

    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model: nn.Module | None = None,
        optimizer_cls: Callable[[ParamsT], Adam] = Adam,
        device: torch.device = DefaultDevice,
        loss_fn: LossFn = mean_abs_error,
    ) -> None:
        self.features = features
        self.targets = targets

        if model is None:
            model = MLP(
                n_inputs=len(features.columns),
                neurons_per_layer=[50, 50, len(targets.columns)],
            )
        self.model = model.to(device)

        self.optimizer = optimizer_cls(model.parameters())
        self.device = device
        self.loss_fn = loss_fn
        self.losses = []

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
            features=self.features.to_numpy(),
            targets=self.targets.to_numpy(),
            epochs=epochs,
            optimizer=self.optimizer,
            batch_size=batch_size,
            device=self.device,
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
    model: nn.Module | None = None,
    optimizer_cls: Callable[[ParamsT], Adam] = Adam,
    device: torch.device = DefaultDevice,
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
    optimizer_cls
        Optimizer class to use for training (default: Adam).
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
        optimizer_cls=optimizer_cls,
        device=device,
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
