"""Surrogate Interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from wadler_lindig import pformat

from mxlpy.meta import _mathml as mml

__all__ = [
    "ACTIVATION_BUILDERS",
    "AbstractSurrogate",
    "MockSurrogate",
    "SurrogateJson",
    "SurrogateProtocol",
    "mxl_json_activation_relu",
    "mxl_json_activation_sigmoid",
    "mxl_json_activation_softplus",
    "mxl_json_activation_tanh",
    "mxl_json_mechanism_additive",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd

    from mxlpy.types import Derived


def mxl_json_mechanism_additive() -> dict[str, object]:
    """`Add(ode, nde)` — mxl-schemas `nnBlock.mechanism`, the only shape any MxlPy surrogate ever composes onto its target(s) with.

    `mechanism` (a MathML expression over `ode`/`nde`, generalized past a
    closed enum) is a real mxl-schemas concept, but a UDE one:
    `mxlpy.jax.models.Ude`/`FluxUde` combine a *whole* mechanistic ODE with
    a *whole* neural-ODE via a selectable operator (`op`, e.g. `"rel"` for
    `ode * (1 + nde)`). A `Surrogate`/`OdeSurrogate` is not that — per
    `docs/llms-mxl.txt`, "a surrogate replaces part (or all) of a model"
    by being summed in, exactly like a reaction (kinetic: `AbstractSurrogate`
    outputs sum into dx/dt via stoichiometry) or a direct dx/dt term
    (ode: `AbstractOdeSurrogate` outputs sum into a targeted diff_eq) — in
    both cases plain addition, never a selectable composition algebra.
    `additive` is therefore the only mechanism any MxlPy surrogate ever
    exports as, or can reconstruct from on load (:mod:`mxlpy.serialize`
    rejects a loaded `.mxl.json` `nn_blocks` entry using any other
    mechanism — not representable back into a live MxlPy surrogate).
    """
    return mml.Add(children=[mml.Name(name="ode"), mml.Name(name="nde")]).to_dict()


def mxl_json_activation_softplus() -> dict[str, object]:
    """The canonical softplus `{name, expression}` pair (mxl-schemas `nnActivation`), ADR 0005 §2.1.1's numerically-stable form.

    Chosen as MxlPy's first supported activation for adjoint-fitting
    smoothness on the mxlweb side; see `ACTIVATION_BUILDERS` for the full
    set a `Surrogate`/`OdeSurrogate` may use.
    """
    x = mml.Name(name="x")
    expression = mml.Add(
        children=[
            mml.Max(children=[x, mml.Num(value=0)]),
            mml.Ln(
                child=mml.Add(
                    children=[
                        mml.Num(value=1),
                        mml.Exp(child=mml.Minus(children=[mml.Abs(child=x)])),
                    ]
                )
            ),
        ]
    )
    return expression.to_dict()


def mxl_json_activation_relu() -> dict[str, object]:
    """`Max(x, 0)` — mxl-schemas `nnActivation`, ReLU."""
    return mml.Max(children=[mml.Name(name="x"), mml.Num(value=0)]).to_dict()


def mxl_json_activation_tanh() -> dict[str, object]:
    """`Tanh(x)` — mxl-schemas `nnActivation`."""
    return mml.Tanh(child=mml.Name(name="x")).to_dict()


def mxl_json_activation_sigmoid() -> dict[str, object]:
    """`1 / (1 + Exp(-x))` — mxl-schemas `nnActivation`, the logistic sigmoid."""
    x = mml.Name(name="x")
    expression = mml.Divide(
        children=[
            mml.Num(value=1),
            mml.Add(
                children=[mml.Num(value=1), mml.Exp(child=mml.Minus(children=[x]))]
            ),
        ]
    )
    return expression.to_dict()


#: Every activation a `Surrogate`/`OdeSurrogate` may recognize on export and
#: reconstruct on import, keyed by the exact `nnActivation.name` mxl-schemas
#: uses. A layer whose activation isn't structurally one of these presets
#: (checked by comparing `expression` dicts, not just trusting a `name`
#: string — the same defense-in-depth `mxl_json_mechanism_additive`'s old
#: sibling comparison already relied on) is unrepresentable; see each
#: backend's `to_mxl_json`/`*_from_mxl_json` for how this is used both ways.
ACTIVATION_BUILDERS: dict[str, Callable[[], dict[str, object]]] = {
    "softplus": mxl_json_activation_softplus,
    "relu": mxl_json_activation_relu,
    "tanh": mxl_json_activation_tanh,
    "sigmoid": mxl_json_activation_sigmoid,
}


@dataclass
class SurrogateJson:
    """A surrogate exported as a mxl-schemas ``nn_blocks`` entry — the mxl.json representation of one surrogate.

    ``spec`` is the schema-shaped architecture/composition dict for one
    ``nn_blocks[id]`` entry (``inputs``/``layers``/``seed``/``targets``/
    ``trained``/``scale``/``mechanism`` — everything except ``weights_ref``,
    which the caller assigns once it knows the sidecar file's path; each
    ``layers[i]`` may carry its own optional ``activation`` — there is no
    top-level ``spec["activation"]`` field, mxl-schemas moved it per-layer).
    ``weights`` is the matching sidecar content
    (``nn-weights.schema.json``): ``w1``/``b1``/``w2``/``b2``/... keyed,
    1-indexed by layer, each weight matrix shaped
    ``[out_features, in_features]``.

    Kept structurally separate from the schema's own architecture dict
    (rather than merging weights into ``spec``) for the same reason the
    schema itself does: weight values carry no meaning the way the
    architecture fields do, and every consumer of ``spec`` alone (schema
    validation, diffing) shouldn't need to know about ``weights`` at all.
    """

    spec: dict[str, object]
    weights: dict[str, list[object]]


class SurrogateProtocol(Protocol):
    """FIXME: Something I will fill out."""

    args: list[str]
    outputs: list[str]
    stoichiometries: dict[str, dict[str, float | Derived]]

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
        ...

    def calculate_inpl(
        self,
        name: str,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> None:
        """Predict outputs and update args in-place.

        Parameters
        ----------
        name
            Name of the surrogate.
        args
            Mapping of input names to their values, updated in-place with predictions.

        """
        ...

    def to_mxl_json(self) -> SurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, or ``None`` if it isn't representable.

        See :meth:`AbstractSurrogate.to_mxl_json`.
        """
        ...


@dataclass(kw_only=True)
class AbstractSurrogate:
    """Abstract base class for surrogate models.

    Attributes
    ----------
    inputs
        List of input variable names.
    stoichiometries
        Dictionary mapping reaction names to stoichiometries.

    Methods
    -------
        predict: Abstract method to predict outputs based on input data.

    """

    args: list[str]
    outputs: list[str]
    stoichiometries: dict[str, dict[str, float | Derived]] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def to_mxl_json(self) -> SurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        This base implementation always returns ``None`` (not
        ``NotImplementedError``) — the correct behaviour for a surrogate
        that's not a neural net at all (``_qss.py``, ``_poly.py``) and so
        never attempts export in the first place; ``None`` here is a
        normal, expected outcome, not a bug.
        :func:`mxlpy.serialize.model_to_dict` treats "every attached
        surrogate exports" as the condition for allowing ``.mxl.json``
        export at all, narrowing the previous "reject if any surrogate is
        attached" rule down to "reject only the ones that actually can't
        round-trip."

        Every NN-backend override (``_torch.py``/``_keras.py``/
        ``_equinox.py``) behaves differently once it *has* committed to
        trying: it **raises** `mxlpy.types.SerializationError` — not
        `None` — when the actual model's architecture doesn't fit the
        schema (a non-dense layer type, an activation outside
        `ACTIVATION_BUILDERS`, or non-unit stoichiometry: a ``nn_blocks``
        entry always applies its whole (scaled) output to each target with
        an implicit coefficient of 1, so a surrogate wired with any other
        per-target coefficient can't be losslessly expressed this way) —
        since a real neural-net object failing to convert is always a
        specific, fixable problem worth surfacing, never a normal outcome
        the way "this isn't a neural net at all" is here.
        """
        return None

    @abstractmethod
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

    def calculate_inpl(
        self,
        name: str,  # noqa: ARG002, for API compatibility
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> None:
        """Predict outputs and update args in-place.

        Parameters
        ----------
        name
            Name of the surrogate.
        args
            Mapping of input names to their values, updated in-place with predictions.

        """
        args |= self.predict(args=args)


@dataclass(kw_only=True)
class MockSurrogate(AbstractSurrogate):
    """Mock surrogate model for testing purposes."""

    fn: Callable[..., Iterable[float]]

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
                self.fn(*(args[i] for i in self.args)),
                strict=True,
            )
        )  # type: ignore
