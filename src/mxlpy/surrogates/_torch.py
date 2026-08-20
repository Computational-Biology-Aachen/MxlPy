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
    AbstractSurrogate,
    NNBlockExport,
    nn_block_activation_softplus,
    nn_block_mechanism_additive,
    nn_block_mechanism_multiply,
    nn_block_mechanism_relative_multiply,
)
from mxlpy.surrogates.abstract_derivative import (
    AbstractDerivativeSurrogate,
    DerivativeSurrogateExport,
    Mechanism,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch.optim.optimizer import ParamsT

    from mxlpy.types import Derived


__all__ = [
    "DerivativeSurrogate",
    "Surrogate",
    "Trainer",
    "derivative_surrogate_from_nn_block",
    "surrogate_from_nn_block",
    "train",
]

_MECHANISM_BUILDER: dict[Mechanism, Callable[[], dict[str, object]]] = {
    "additive": nn_block_mechanism_additive,
    "relative_multiply": nn_block_mechanism_relative_multiply,
    "multiply": nn_block_mechanism_multiply,
}


# mxl-schemas activation.name -> the torch.nn.Module type it corresponds to.
# Only softplus exists today (ADR 0005 §2.1.1's numerically-stable form,
# chosen for adjoint-fitting smoothness) — an MLP built with any other
# activation (the mxlpy.nn._torch.MLP default is ReLU) simply isn't
# representable in the schema yet, so to_nn_block_export declines it.
_ACTIVATION_BY_MODULE: dict[type[nn.Module], str] = {nn.Softplus: "softplus"}


def _dense_sequential(model: nn.Module) -> nn.Sequential | None:
    """The plain `nn.Sequential` of alternating Linear+activation this model wraps, if any.

    Handles `mxlpy.nn._torch.MLP` (whose real layer stack lives on `.net`)
    and a bare hand-built `nn.Sequential` alike; any other module shape
    (LSTM-backed, custom `forward`, ...) returns `None`.
    """
    net = getattr(model, "net", model)
    return net if isinstance(net, nn.Sequential) else None


def surrogate_from_nn_block(
    name: str,
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> Surrogate:
    """Reconstruct a torch :class:`Surrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`Surrogate.to_nn_block_export` — only handles the
    exact shape that method produces (dense layers, softplus hidden
    activation, additive mechanism); :func:`mxlpy.serialize.model_from_dict`
    checks that shape before calling this, so it isn't re-validated here.
    Reconstructing a block authored elsewhere (e.g. in mxlweb, with a
    different mechanism or a `layers` entry of a future non-dense type)
    is out of scope.

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

    modules: list[nn.Module] = []
    in_features = len(inputs)
    for i, layer in enumerate(layers):
        out_features = cast("int", layer["width"])
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(weights[f"w{i + 1}"], dtype=torch.float32))
            linear.bias.copy_(torch.tensor(weights[f"b{i + 1}"], dtype=torch.float32))
        modules.append(linear)
        if i < len(layers) - 1:
            modules.append(nn.Softplus())
        in_features = out_features

    return Surrogate(
        model=nn.Sequential(*modules),
        args=inputs,
        outputs=outputs,
        stoichiometries={
            output: {target: 1.0} for output, target in zip(outputs, targets, strict=True)
        },
    )


def derivative_surrogate_from_nn_block(
    spec: Mapping[str, object],
    weights: Mapping[str, list[object]],
) -> DerivativeSurrogate:
    """Reconstruct a torch :class:`DerivativeSurrogate` from a mxl-schemas ``nn_blocks`` entry and its weights sidecar.

    Inverse of :meth:`DerivativeSurrogate.to_nn_block_export`. Unlike
    :func:`surrogate_from_nn_block`, no `name`-derived output renaming is
    needed here: a direct-derivative surrogate has no `outputs`/
    `stoichiometries` indirection at all, it corrects `targets` directly.

    `spec["mechanism"]` is matched against the three known
    `nn_block_mechanism_*` presets (structural dict equality — the only
    shapes any exportable surrogate ever produces); a `mechanism` authored
    by hand or by mxlweb in some other equivalent-but-differently-built
    tree isn't recognized and raises `ValueError`.
    """
    layers = cast("list[dict[str, object]]", spec["layers"])
    inputs = cast("list[str]", spec["inputs"])
    targets = cast("list[str]", spec["targets"])
    mechanism_dict = spec["mechanism"]
    mechanism: Mechanism | None = next(
        (m for m, builder in _MECHANISM_BUILDER.items() if builder() == mechanism_dict),
        None,
    )
    if mechanism is None:
        msg = f"Unrecognized nn_blocks mechanism: {mechanism_dict!r}"
        raise ValueError(msg)

    modules: list[nn.Module] = []
    in_features = len(inputs)
    for i, layer in enumerate(layers):
        out_features = cast("int", layer["width"])
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(weights[f"w{i + 1}"], dtype=torch.float32))
            linear.bias.copy_(torch.tensor(weights[f"b{i + 1}"], dtype=torch.float32))
        modules.append(linear)
        if i < len(layers) - 1:
            modules.append(nn.Softplus())
        in_features = out_features

    return DerivativeSurrogate(
        model=nn.Sequential(*modules),
        args=inputs,
        targets=targets,
        mechanism=mechanism,
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

    def to_nn_block_export(self) -> NNBlockExport | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, if it's representable.

        Requires, structurally:

        - ``self.model`` wraps a plain `nn.Sequential` of strictly
          alternating `Linear`/activation modules (handles
          :class:`mxlpy.nn._torch.MLP` via its `.net`, or a hand-built
          `nn.Sequential` directly) ending in `Linear` — an LSTM-backed or
          custom-`forward` model isn't representable at all.
        - every hidden activation is the same, schema-known one (today,
          only `nn.Softplus` — `MLP`'s own default is `ReLU`, which has no
          `nn_blocks` counterpart yet).
        - every output has stoichiometry ``{compound: 1.0}`` against a
          reaction named after that same output — the only shape
          equivalent to an `nn_blocks` `additive` correction (a bare
          coefficient of 1, one target per output); anything else
          (a non-unit or `Derived` coefficient, an output feeding more
          than one compound) declines rather than guess at a lossy
          mapping.

        Returns ``None`` when any of the above doesn't hold.
        """
        net = _dense_sequential(self.model)
        if net is None:
            return None

        modules = list(net)
        linears = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            return None
        if len(modules) != 2 * len(linears) - 1:
            # Anything other than strict Linear/activation alternation
            # ending in Linear — including a nonzero `output_activation`,
            # which `nn_blocks` has no field for at all.
            return None

        activation_name: str | None = None
        for i, m in enumerate(modules):
            if i % 2 == 0:
                continue  # Linear, already collected above.
            mapped = _ACTIVATION_BY_MODULE.get(type(m))
            if mapped is None or (
                activation_name is not None and activation_name != mapped
            ):
                return None
            activation_name = mapped
        if len(linears) > 1 and activation_name is None:
            # Unreachable given the alternation check above (a >1-layer
            # network always has an odd-index activation slot), kept as an
            # explicit invariant rather than relying on that implicitly.
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
            weights[f"w{i}"] = lin.weight.detach().cpu().numpy().tolist()
            weights[f"b{i}"] = lin.bias.detach().cpu().numpy().tolist()

        spec: dict[str, object] = {
            "inputs": list(self.args),
            "layers": layers,
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
class DerivativeSurrogate(AbstractDerivativeSurrogate):
    """Direct-derivative surrogate using PyTorch, for `OdeModelBuilder`.

    See :mod:`mxlpy.surrogates.abstract_derivative`'s module docstring for
    why this is a separate, orthogonal type from :class:`Surrogate` rather
    than a shared base — same reasoning, applied one level down at the
    torch-backend layer.

    Attributes
    ----------
    model
        PyTorch neural network model.

    """

    model: torch.nn.Module

    def to_nn_block_export(self) -> DerivativeSurrogateExport | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, if it's representable.

        Same structural requirements as :meth:`Surrogate.to_nn_block_export`
        (dense `Linear`/softplus alternation) but, unlike that method, no
        stoichiometry-unit-coefficient constraint: `self.targets` already
        names exactly what this surrogate corrects, so every `self.mechanism`
        value (not just `additive`) exports faithfully via
        `_MECHANISM_BUILDER`.
        """
        net = _dense_sequential(self.model)
        if net is None:
            return None

        modules = list(net)
        linears = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            return None
        if len(modules) != 2 * len(linears) - 1:
            return None

        activation_name: str | None = None
        for i, m in enumerate(modules):
            if i % 2 == 0:
                continue  # Linear, already collected above.
            mapped = _ACTIVATION_BY_MODULE.get(type(m))
            if mapped is None or (
                activation_name is not None and activation_name != mapped
            ):
                return None
            activation_name = mapped
        if len(linears) > 1 and activation_name is None:
            return None

        layers = [{"type": "dense", "width": lin.out_features} for lin in linears]
        weights: dict[str, list[object]] = {}
        for i, lin in enumerate(linears, start=1):
            weights[f"w{i}"] = lin.weight.detach().cpu().numpy().tolist()
            weights[f"b{i}"] = lin.bias.detach().cpu().numpy().tolist()

        spec: dict[str, object] = {
            "inputs": list(self.args),
            "layers": layers,
            "seed": 0,
            "targets": list(self.targets),
            "trained": True,
            "scale": 1.0,
            "mechanism": _MECHANISM_BUILDER[self.mechanism](),
            "activation": {
                "name": "softplus",
                "expression": nn_block_activation_softplus(),
            },
        }
        return DerivativeSurrogateExport(spec=spec, weights=weights)

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict a correction per target based on input data using the PyTorch model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        np.ndarray
            Array of predicted correction values, ordered like `self.targets`.

        """
        with torch.no_grad():
            return self.model(
                torch.tensor(y, dtype=torch.float32),
            ).numpy()

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> dict[str, float]:
        """Predict a correction per target, from `args`.

        Parameters
        ----------
        args
            Mapping of input names to their values.

        Returns
        -------
        dict[str, float]
            Mapping of target names to raw predicted correction values.

        """
        return dict(
            zip(
                self.targets,
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
