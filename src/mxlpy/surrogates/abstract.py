"""Surrogate Interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from wadler_lindig import pformat

from mxlpy.meta import _mathml as mml

__all__ = [
    "AbstractSurrogate",
    "MockSurrogate",
    "SurrogateJson",
    "SurrogateProtocol",
    "mxl_json_activation_softplus",
    "mxl_json_mechanism_additive",
    "mxl_json_mechanism_multiply",
    "mxl_json_mechanism_relative_multiply",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd

    from mxlpy.types import Derived


def mxl_json_mechanism_additive() -> dict[str, object]:
    """`Add(ode, nde)` — mxl-schemas `nnBlock.mechanism`, the only shape a stoichiometry-composed surrogate can correctly export as.

    MxlPy has no multiplicative-reaction concept at all: every reaction
    (surrogate-owned "reactions" included) contributes to a compound's
    dx/dt by being summed, weighted by its stoichiometric coefficient
    (`_kinetic_builder.py`'s `stoich_by_compounds`). Exporting any other
    mechanism (e.g. `relative_multiply`) would describe dynamics a
    schema-faithful consumer simulates differently from what MxlPy itself
    actually computes for the same model. Also the only mechanism shape
    :mod:`mxlpy.serialize` reconstructs a real surrogate from on load — a
    loaded `.mxl.json` whose `nn_blocks` entry uses a different mechanism
    (e.g. authored in mxlweb) isn't representable back into a live MxlPy
    surrogate.
    """
    return mml.Add(children=[mml.Name(name="ode"), mml.Name(name="nde")]).to_dict()


def mxl_json_mechanism_relative_multiply() -> dict[str, object]:
    """`Mul(ode, Add(1, nde))` — mxl-schemas `nnBlock.mechanism`.

    ``dx/dt = ode * (1 + nde)``: a near-zero/untrained network leaves
    ``ode`` unchanged. Only meaningful for a surrogate that composes
    directly onto an existing dx/dt (`mxlpy.surrogates.abstract_derivative`
    — `OdeModelBuilder`); the reaction/stoichiometry-composed kinetic
    surrogate (`AbstractSurrogate`) can never correctly export this shape,
    since it has no multiplicative-composition concept at all (see
    `mxl_json_mechanism_additive`'s doc comment).
    """
    return mml.Mul(
        children=[
            mml.Name(name="ode"),
            mml.Add(children=[mml.Num(value=1), mml.Name(name="nde")]),
        ]
    ).to_dict()


def mxl_json_mechanism_multiply() -> dict[str, object]:
    """`Mul(ode, nde)` — mxl-schemas `nnBlock.mechanism`.

    ``dx/dt = ode * nde``: a bare product, with none of
    `mxl_json_mechanism_relative_multiply`'s safeguard. Same
    direct-composition-only caveat as that function.
    """
    return mml.Mul(children=[mml.Name(name="ode"), mml.Name(name="nde")]).to_dict()


def mxl_json_activation_softplus() -> dict[str, object]:
    """The canonical softplus `{name, expression}` pair (mxl-schemas `nnActivation`), ADR 0005 §2.1.1's numerically-stable form.

    The only activation any exportable surrogate's hidden layers may use
    today — chosen for adjoint-fitting smoothness on the mxlweb side; a
    network built with any other activation (`mxlpy.nn._torch.MLP`'s own
    default is ReLU) isn't representable in the schema yet.
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


@dataclass
class SurrogateJson:
    """A surrogate exported as a mxl-schemas ``nn_blocks`` entry — the mxl.json representation of one surrogate.

    ``spec`` is the schema-shaped architecture/composition dict for one
    ``nn_blocks[id]`` entry (``inputs``/``layers``/``seed``/``targets``/
    ``trained``/``scale``/``mechanism``/``activation`` — everything except
    ``weights_ref``, which the caller assigns once it knows the sidecar
    file's path). ``weights`` is the matching sidecar content
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

        Returns ``None`` (not ``NotImplementedError``) when this surrogate
        can't be represented in the shared schema — an opaque/closed-form
        surrogate (``_qss.py``, ``_poly.py``), or a neural one whose
        architecture doesn't fit the schema's model (an unsupported
        activation, non-dense layers, or non-unit stoichiometry: a
        ``nn_blocks`` entry always applies its whole (scaled) output to
        each target with an implicit coefficient of 1, so a surrogate
        wired with any other per-target coefficient can't be losslessly
        expressed this way). ``None`` is a normal, expected outcome, not a
        bug: :func:`mxlpy.serialize.model_to_dict` treats "every attached
        surrogate exports" as the condition for allowing ``.mxl.json``
        export at all, narrowing the previous "reject if any surrogate is
        attached" rule down to "reject only the ones that actually can't
        round-trip."
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
