"""Ode surrogate interface, for OdeModelBuilder.

Deliberately unrelated to :mod:`mxlpy.surrogates.abstract` — that module's
`AbstractSurrogate`/`SurrogateProtocol` are shaped around
`KineticModelBuilder`'s reaction/stoichiometry model (a surrogate's outputs
are new named "reaction rates", combined into a compound's dx/dt via a
stoichiometric coefficient, exactly like a real reaction). `OdeModelBuilder`
has no such indirection: a variable's dx/dt is a direct expression, so this
module's `targets` maps an output directly onto the diff_eq(s) it additively
feeds instead of via a stoichiometric coefficient.

Otherwise the two are as close to identical as the underlying model
difference allows — `AbstractOdeSurrogate`, like `AbstractSurrogate`, just
"replaces part (or all) of a model" (`docs/llms-mxl.txt`): an output present
in `targets` sums into that diff_eq's dx/dt (coefficient always 1 — unlike
kinetic stoichiometry, there's no real mass-balance ratio a direct dx/dt term
could carry, so this module doesn't offer one); an output absent from
`targets` is a plain derived value, usable elsewhere exactly like a real
`Derived`. There is no `mechanism`/composition-algebra concept at all here —
that belongs to `mxlpy.jax.models.Ude`/`FluxUde` (a UDE combines a *whole*
mechanistic ODE with a *whole* neural-ODE via a selectable operator), not to
a surrogate that only ever replaces part of one.

The two ability sets are still genuinely different enough (`targets: dict`
here vs. `stoichiometries: dict` there — no coefficient, and diff_eq names
instead of compound names) to keep as separate Protocols/base classes on
purpose, so a kinetic-shaped surrogate can't accidentally be handed to
`OdeModelBuilder.add_surrogate` (or vice versa) and silently do the wrong
thing — `KineticModelBuilder.add_surrogate`/`OdeModelBuilder.add_surrogate`
type-hint against their own Protocol here, and the two share no base class.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from wadler_lindig import pformat

__all__ = [
    "AbstractOdeSurrogate",
    "OdeSurrogateJson",
    "OdeSurrogateProtocol",
]

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class OdeSurrogateJson:
    """An ode surrogate exported as a mxl-schemas ``nn_blocks`` entry.

    Same shape as :class:`mxlpy.surrogates.abstract.SurrogateJson` (a
    schema-shaped ``spec`` dict, plus the matching weights sidecar content)
    — kept as its own type rather than reusing that one, for the same
    "orthogonal typing" reason the two surrogate abstractions themselves
    are kept separate: nothing here should structurally satisfy the
    kinetic-shaped export path or vice versa.
    """

    spec: dict[str, object]
    weights: dict[str, list[object]]


class OdeSurrogateProtocol(Protocol):
    """Structural interface `OdeModelBuilder.add_surrogate` accepts.

    See the module docstring for why this is unrelated to
    :class:`mxlpy.surrogates.abstract.SurrogateProtocol`.
    """

    args: list[str]
    outputs: list[str]
    targets: dict[str, list[str]]

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        """Predict outputs based on input data.

        Parameters
        ----------
        args
            Mapping of input names (`self.args`) to their values.

        Returns
        -------
        dict[str, float]
            Mapping of output names (`self.outputs`) to predicted values.

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

    def to_mxl_json(self) -> OdeSurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry, or ``None`` if it isn't representable."""
        ...


@dataclass(kw_only=True)
class AbstractOdeSurrogate:
    """Base class for an ode (`OdeModelBuilder`) surrogate.

    Attributes
    ----------
    args
        Names of existing variables/parameters/dynamic-derived quantities
        this surrogate reads.
    outputs
        Names of every value this surrogate predicts — every output is
        registered as a fresh, model-wide-unique id and merged into `args`
        by `calculate_inpl`, exactly like a real `Derived`.
    targets
        Which outputs additively feed which existing diff_eq(s)' dx/dt —
        `{output_name: [diff_eq_name, ...]}`. An output absent from this
        mapping is a plain derived value: present in `args` for anything
        downstream to reference by name, but contributing to no dx/dt on
        its own.

    """

    args: list[str]
    outputs: list[str]
    targets: dict[str, list[str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def to_mxl_json(self) -> OdeSurrogateJson | None:
        """Export this surrogate as a mxl-schemas ``nn_blocks`` entry.

        Returns ``None`` (not ``NotImplementedError``) — the correct
        behaviour for a base class with no real architecture to export;
        every NN-backend override (``_torch.py``/``_keras.py``/
        ``_equinox.py``) instead **raises**
        `mxlpy.types.SerializationError` once it has committed to trying —
        see :meth:`mxlpy.surrogates.abstract.AbstractSurrogate.to_mxl_json`
        for the full raise/``None`` contract, identical here.
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
