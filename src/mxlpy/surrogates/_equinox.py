from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from mxlpy.nn._equinox import LossFn, mean_abs_error
from mxlpy.nn._equinox import train as _train
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

    from mxlpy.types import Derived

__all__ = [
    "OdeSurrogate",
    "Surrogate",
    "Trainer",
    "ode_surrogate_from_mxl_json",
    "surrogate_from_mxl_json",
    "train",
]

# mxl-schemas activation.name <-> the jax activation function it corresponds
# to. Unlike torch (a discrete `nn.Module`) or keras (`Dense.activation`, a
# real introspectable attribute), an equinox model's activation is just a
# Python callable stashed on the module (`eqx.nn.Lambda.fn`, or
# `eqx.nn.MLP.activation`/`.final_activation`) — recognized here by identity
# against this registry. Anything else (or a model shape that isn't a plain
# `eqx.nn.Sequential`/`eqx.nn.MLP` of `Linear`/activation layers) makes
# `to_mxl_json` raise, not silently decline.
_ACTIVATION_FN_BY_NAME: dict[str, Callable] = {
    "softplus": jax.nn.softplus,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "sigmoid": jax.nn.sigmoid,
}
_ACTIVATION_NAME_BY_FN: dict[Callable, str] = {
    v: k for k, v in _ACTIVATION_FN_BY_NAME.items()
}

# `eqx.nn.MLP`'s own default `final_activation` (identity) — the same
# function object every time, since it's a module-level lambda inside
# equinox. Used to tell "this MLP has no final activation" apart from "this
# MLP's final activation happens to be one we don't recognize". Relies on
# `eqx.nn.MLP.__init__` passing that lambda through `filter_vmap` as an
# untouched static leaf (verified against equinox 0.13, not part of
# equinox's documented public contract) — an equinox upgrade that changes
# this would need this sentinel re-verified.
_MLP_DEFAULT_FINAL_ACTIVATION: Callable = eqx.nn.MLP(
    in_size=1, out_size=1, width_size=1, depth=1, key=jax.random.PRNGKey(0)
).final_activation


def _layers_from_sequential(
    seq: eqx.nn.Sequential,
) -> list[tuple[eqx.nn.Linear, Callable | None]] | None:
    """Split an `eqx.nn.Sequential` into `(Linear, activation-fn-or-None)` pairs.

    Each `Linear` is paired with the function wrapped by an immediately
    following `eqx.nn.Lambda` (that layer's activation) — absent if the next
    layer is another `Linear` or there is no next layer (identity/linear
    output). `None` (not an empty list) means `seq` isn't representable at
    all — doesn't start with a `Linear`, has two non-`Linear` layers in a
    row, or a non-`Linear`/`Lambda` layer.
    """
    modules = list(seq.layers)
    layers: list[tuple[eqx.nn.Linear, Callable | None]] = []
    i = 0
    while i < len(modules):
        linear = modules[i]
        if not isinstance(linear, eqx.nn.Linear):
            return None
        i += 1
        activation: Callable | None = None
        if i < len(modules):
            nxt = modules[i]
            if isinstance(nxt, eqx.nn.Lambda):
                activation = nxt.fn
                i += 1
            elif not isinstance(nxt, eqx.nn.Linear):
                return None
        layers.append((linear, activation))
    return layers or None


def _layers_from_mlp(
    mlp: eqx.nn.MLP,
) -> list[tuple[eqx.nn.Linear, Callable | None]]:
    """Split an `eqx.nn.MLP` into `(Linear, activation-fn-or-None)` pairs.

    `mlp.activation` applies to every hidden layer, `mlp.final_activation`
    only to the last — mapped onto mxl-schemas' per-layer, independently
    optional `activation` field the same way `_layers_from_sequential` does.
    """
    linears = list(mlp.layers)
    layers: list[tuple[eqx.nn.Linear, Callable | None]] = [
        (linear, mlp.activation) for linear in linears[:-1]
    ]
    final_activation = (
        None
        if mlp.final_activation is _MLP_DEFAULT_FINAL_ACTIVATION
        else mlp.final_activation
    )
    layers.append((linears[-1], final_activation))
    return layers


def _dense_layers(
    model: eqx.Module,
) -> list[tuple[eqx.nn.Linear, Callable | None]] | None:
    """The `(Linear, activation-fn-or-None)` layer stack `model` wraps, if any."""
    if isinstance(model, eqx.nn.MLP):
        return _layers_from_mlp(model)
    if isinstance(model, eqx.nn.Sequential):
        return _layers_from_sequential(model)
    return None


def _export_dense_layers(
    model: eqx.Module,
) -> tuple[list[dict[str, object]], dict[str, list[object]]]:
    """Export `model`'s architecture/weights as mxl-schemas `layers`/weights-sidecar content, or raise.

    Shared by :meth:`Surrogate.to_mxl_json`/:meth:`OdeSurrogate.to_mxl_json`
    — everything about exporting layers is identical between the two; they
    differ only in how `outputs` maps onto the schema's `targets` (kinetic:
    via unit stoichiometry; ode: via `self.targets` directly).
    """
    parsed = _dense_layers(model)
    if parsed is None:
        msg = (
            f"{type(model).__name__} isn't representable as a mxl-schemas "
            "nn_blocks entry: not a plain eqx.nn.Sequential of "
            "Linear/Lambda layers or an eqx.nn.MLP — an LSTM-backed or "
            "custom-__call__ model can't be expressed this way."
        )
        raise SerializationError(msg)

    layers_spec: list[dict[str, object]] = []
    weights: dict[str, list[object]] = {}
    for i, (linear, activation_fn) in enumerate(parsed, start=1):
        if not linear.use_bias:
            msg = f"Layer {i} has no bias (use_bias=False) — not representable."
            raise SerializationError(msg)
        layer_spec: dict[str, object] = {
            "type": "dense",
            "width": linear.out_features,
        }
        if activation_fn is not None:
            name = _ACTIVATION_NAME_BY_FN.get(activation_fn)
            if name is None:
                msg = (
                    f"Layer {i} uses unsupported activation {activation_fn!r} "
                    f"— recognized: {sorted(ACTIVATION_BUILDERS)}."
                )
                raise SerializationError(msg)
            layer_spec["activation"] = {
                "name": name,
                "expression": ACTIVATION_BUILDERS[name](),
            }
        layers_spec.append(layer_spec)
        weights[f"w{i}"] = np.asarray(linear.weight).tolist()
        weights[f"b{i}"] = np.asarray(linear.bias).tolist()

    return layers_spec, weights


def _revive_dense_sequential(
    layers: list[dict[str, object]],
    in_features: int,
    weights: Mapping[str, list[object]],
) -> eqx.nn.Sequential:
    """Build an equinox `eqx.nn.Sequential` from a mxl-schemas `layers` array and its weights sidecar.

    Inverse of :func:`_export_dense_layers`. Each layer's optional
    `activation` is verified structurally against
    `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS` (matching the exact
    `expression` a known preset produces, not just trusting `name`) before
    being revived as a matching `eqx.nn.Lambda`; an unrecognized or
    non-matching activation raises.
    """
    modules: list[Callable] = []
    key = jax.random.PRNGKey(0)
    for i, layer in enumerate(layers, start=1):
        out_features = cast("int", layer["width"])
        key, subkey = jax.random.split(key)
        linear = eqx.nn.Linear(in_features, out_features, key=subkey)
        w = jnp.asarray(weights[f"w{i}"], dtype=jnp.float32)
        b = jnp.asarray(weights[f"b{i}"], dtype=jnp.float32)
        linear = eqx.tree_at(lambda lin: (lin.weight, lin.bias), linear, (w, b))
        modules.append(linear)

        activation = cast("dict[str, object] | None", layer.get("activation"))
        if activation is not None:
            name = cast("str", activation["name"])
            fn = _ACTIVATION_FN_BY_NAME.get(name)
            builder = ACTIVATION_BUILDERS.get(name)
            if (
                fn is None
                or builder is None
                or builder() != activation.get("expression")
            ):
                msg = f"Layer {i}: unrecognized or non-matching activation {activation!r}."
                raise SerializationError(msg)
            modules.append(eqx.nn.Lambda(fn))

        in_features = out_features
    return eqx.nn.Sequential(modules)


def surrogate_from_mxl_json(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> Surrogate:
    """Reconstruct an equinox :class:`Surrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`Surrogate.to_mxl_json`, mirroring
    `mxlpy.surrogates._torch.surrogate_from_mxl_json`'s reconstruction (same
    "fresh output name" `f"{name}_{target}"` derivation and unit
    stoichiometry, since this is the same kinetic `AbstractSurrogate` shape,
    just with an equinox model instead of a torch one).
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
            output: {target: 1.0}
            for output, target in zip(outputs, targets, strict=True)
        },
    )


def ode_surrogate_from_mxl_json(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> OdeSurrogate:
    """Reconstruct an equinox :class:`OdeSurrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`OdeSurrogate.to_mxl_json`. Mirrors
    :func:`surrogate_from_mxl_json`'s `name`-derived output renaming exactly
    — an ode surrogate's outputs are fresh, model-wide-unique ids too
    (`mxlpy.surrogates.abstract_ode`'s module docstring), and a target's own
    name is already taken by the existing diff_eq.
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
    """Surrogate model using equinox.

    Attributes
    ----------
    model
        Equinox neural network model.

    Methods
    -------
        predict: Predict outputs based on input data using the equinox model.

    """

    model: eqx.Module

    def to_mxl_json(self) -> SurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Requires, structurally:

        - ``self.model`` is a plain `eqx.nn.Sequential` of `Linear` layers,
          each optionally followed by a single `eqx.nn.Lambda`-wrapped
          recognized activation function, or an `eqx.nn.MLP` whose
          `.activation`/`.final_activation` are each either the MLP
          default identity or a recognized activation — anything else
          (LSTM-backed, custom `__call__`, unrecognized activation
          function) isn't representable.
        - every output has stoichiometry ``{compound: 1.0}`` against a
          reaction named after that same output — the only shape
          equivalent to an `nn_blocks` correction (a bare coefficient of
          1, one target per output); anything else raises rather than
          guess at a lossy mapping.

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

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict outputs based on input data using the equinox model.

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


@dataclass(kw_only=True)
class OdeSurrogate(AbstractOdeSurrogate):
    """Ode surrogate using equinox, for `OdeModelBuilder`.

    See :mod:`mxlpy.surrogates.abstract_ode`'s module docstring for why this
    is a separate, orthogonal type from :class:`Surrogate` rather than a
    shared base — same reasoning, applied one level down at the
    equinox-backend layer.

    Attributes
    ----------
    model
        Equinox neural network model.

    """

    model: eqx.Module

    def to_mxl_json(self) -> OdeSurrogateJson:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Same structural requirements as :meth:`Surrogate.to_mxl_json`
        (dense `Linear`/recognized-activation layers), and — despite having
        no stoichiometry-coefficient constraint at the model level — the
        *schema*'s `targets` is still one entry per output, so every output
        must map to exactly one target diff_eq to be representable: a
        plain-derived output (absent from `self.targets`) or one feeding
        more than one diff_eq has no `nn_blocks` shape to export as.

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
        """Predict outputs based on input data using the equinox model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        np.ndarray
            Array of predicted values, ordered like `self.outputs`.

        """
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
            model = eqx.nn.MLP(
                in_size=len(features.columns),
                out_size=len(targets.columns),
                width_size=50,
                depth=2,
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
    """Train an equinox surrogate model.

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
    loss_fn
        Custom loss function or instance of torch loss object

    Returns
    -------
    tuple[Surrogate, pd.Series]
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
