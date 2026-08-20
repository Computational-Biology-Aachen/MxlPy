"""Direct-derivative surrogate interface, for OdeModelBuilder.

Deliberately unrelated to :mod:`mxlpy.surrogates.abstract` — that module's
`AbstractSurrogate`/`SurrogateProtocol` are shaped around
`KineticModelBuilder`'s reaction/stoichiometry model (a surrogate's outputs
are new named "reaction rates", combined into a compound's dx/dt via a
stoichiometric coefficient, exactly like a real reaction). `OdeModelBuilder`
has no such indirection: a variable's dx/dt is a direct expression, so a
surrogate here corrects that dx/dt value directly — the same "ode"/"nde"
composition mxlweb's nn_blocks schema describes, not a reaction stand-in.

The two ability sets are genuinely different (`predict`+`correct_inpl`+
`targets`+`mechanism` here vs. `predict`+`calculate_inpl`+`outputs`+
`stoichiometries` there) and kept as separate Protocols/base classes on
purpose, so a kinetic-shaped surrogate can't accidentally be handed to
`OdeModelBuilder.add_surrogate` (or vice versa) and silently do the wrong
thing — `KineticModelBuilder.add_surrogate`/`OdeModelBuilder.add_surrogate`
type-hint against their own Protocol here, and the two share no base class.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from wadler_lindig import pformat

__all__ = [
    "AbstractDerivativeSurrogate",
    "DerivativeSurrogateJson",
    "DerivativeSurrogateProtocol",
    "Mechanism",
]

if TYPE_CHECKING:
    import pandas as pd

Mechanism = Literal["additive", "relative_multiply", "multiply"]


@dataclass
class DerivativeSurrogateJson:
    """A direct-derivative surrogate exported as a mxl-schemas ``nn_blocks`` entry.

    Same shape as :class:`mxlpy.surrogates.abstract.SurrogateJson` (a
    schema-shaped ``spec`` dict, plus the matching weights sidecar content)
    — kept as its own type rather than reusing that one, for the same
    "orthogonal typing" reason the two surrogate abstractions themselves
    are kept separate: nothing here should structurally satisfy the
    kinetic-shaped export path or vice versa.
    """

    spec: dict[str, object]
    weights: dict[str, list[object]]


def _compose(mechanism: Mechanism, ode: float, nde: float) -> float:
    """Runtime numeric composition of a mechanistic dx/dt value with a surrogate's (already-scaled) correction.

    Mirrors mxl-schemas' `nn_blocks[id].mechanism` presets exactly (ADR
    0005 §2.1's grill-me follow-up) — `additive`: `ode + nde`.
    `relative_multiply`: `ode * (1 + nde)` — a near-zero/untrained network
    leaves `ode` unchanged. `multiply`: `ode * nde` — a bare product with
    no such safeguard. Implemented as plain runtime arithmetic rather than
    a MathML expression tree substitution (`mxlweb-core`'s approach): a
    derivative surrogate's raw output is a real number computed by a real
    ML framework call, not a symbolic value sympy could substitute into an
    expression, so there's no tree to compose against here — just two
    floats.
    """
    if mechanism == "additive":
        return ode + nde
    if mechanism == "relative_multiply":
        return ode * (1 + nde)
    return ode * nde


class DerivativeSurrogateProtocol(Protocol):
    """Structural interface `OdeModelBuilder.add_surrogate` accepts.

    See the module docstring for why this is unrelated to
    :class:`mxlpy.surrogates.abstract.SurrogateProtocol`.
    """

    args: list[str]
    targets: list[str]
    mechanism: Mechanism

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        """Predict a correction per target, from `args` (state/parameter values, never a target's own dx/dt — see `correct_inpl`).

        Parameters
        ----------
        args
            Mapping of input names (`self.args`) to their values.

        Returns
        -------
        dict[str, float]
            Mapping of target names (`self.targets`) to raw (unscaled,
            un-composed) predicted correction values.

        """
        ...

    def correct_inpl(
        self,
        source: dict[str, float],
        target: dict[str, float],
    ) -> None:
        """Predict a correction from `source` and compose it onto each of `self.targets`' current value in `target`, in place.

        Parameters
        ----------
        source
            Concentrations/parameters/dynamic-derived values to predict
            from — must NOT be the same dict `target` mutates, since a
            variable's *concentration* (needed as input here) and its
            *dx/dt* (what `target` holds by the time this runs) share the
            same key.
        target
            The dx/dt-per-variable dict being built — mutated in place,
            composing this surrogate's prediction onto each existing
            `self.targets` entry via `self.mechanism`.

        """
        ...

    def to_mxl_json(self) -> DerivativeSurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, or ``None`` if it isn't representable."""
        ...


@dataclass(kw_only=True)
class AbstractDerivativeSurrogate:
    """Base class for a direct-derivative (OdeModelBuilder) surrogate.

    Attributes
    ----------
    args
        Names of existing variables/parameters/dynamic-derived quantities
        this surrogate reads.
    targets
        Which existing diff_eq(s) this surrogate corrects — one entry per
        predicted output, in order.
    mechanism
        How this surrogate's prediction composes onto each target's
        existing (mechanistic) dx/dt value. Default `"additive"`.

    """

    args: list[str]
    targets: list[str]
    mechanism: Mechanism = "additive"

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def to_mxl_json(self) -> DerivativeSurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Returns ``None`` (not ``NotImplementedError``) when this
        surrogate's architecture doesn't fit the schema's model — an
        unsupported activation or non-dense layers. Unlike the kinetic
        surrogate's export path, there is no stoichiometry constraint to
        satisfy here: this surrogate already composes directly onto its
        targets' dx/dt exactly the way `nn_blocks` describes, so any
        `mechanism` (not just `additive`) can be exported faithfully.
        """
        return None

    @abstractmethod
    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
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

    def correct_inpl(
        self,
        source: dict[str, float],
        target: dict[str, float],
    ) -> None:
        """Predict a correction from `source` and compose it onto each of `self.targets`' current value in `target`, in place.

        Parameters
        ----------
        source
            Concentrations/parameters/dynamic-derived values to predict
            from.
        target
            The dx/dt-per-variable dict being built — mutated in place.

        """
        prediction = self.predict(args={k: source[k] for k in self.args})
        for name in self.targets:
            target[name] = _compose(self.mechanism, target[name], prediction[name])
