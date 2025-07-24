"""Diffing utilities for comparing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wadler_lindig import pformat

from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mxlpy.model import Model

__all__ = [
    "DerivedDiff",
    "ModelDiff",
    "ReactionDiff",
    "model_diff",
    "soft_eq",
]


@dataclass
class DerivedDiff:
    """Difference between two derived variables."""

    args1: list[str] = field(default_factory=list)
    args2: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class ReactionDiff:
    """Difference between two reactions."""

    args1: list[str] = field(default_factory=list)
    args2: list[str] = field(default_factory=list)
    stoichiometry1: dict[str, float | Derived] = field(default_factory=dict)
    stoichiometry2: dict[str, float | Derived] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class ModelDiff:
    """Difference between two models."""

    missing_parameters: set[str] = field(default_factory=set)
    missing_variables: set[str] = field(default_factory=set)
    missing_reactions: set[str] = field(default_factory=set)
    missing_surrogates: set[str] = field(default_factory=set)
    missing_readouts: set[str] = field(default_factory=set)
    missing_derived: set[str] = field(default_factory=set)
    different_parameters: dict[str, tuple[float, float]] = field(default_factory=dict)
    different_variables: dict[str, tuple[float, float]] = field(default_factory=dict)
    different_reactions: dict[str, ReactionDiff] = field(default_factory=dict)
    different_surrogates: dict[str, ReactionDiff] = field(default_factory=dict)
    different_readouts: dict[str, DerivedDiff] = field(default_factory=dict)
    different_derived: dict[str, DerivedDiff] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __str__(self) -> str:
        """Return a human-readable string representation of the diff."""
        content = ["Model Diff", "----------"]

        # Parameters
        if self.missing_parameters:
            content.append(
                "Missing Parameters: {}".format(", ".join(self.missing_parameters))
            )
        if self.different_parameters:
            content.append("Different Parameters:")
            for k, (v1, v2) in self.different_parameters.items():
                content.append(f"  {k}: {v1} != {v2}")

        # Variables
        if self.missing_variables:
            content.append(
                "Missing Variables: {}".format(", ".join(self.missing_variables))
            )
        if self.different_variables:
            content.append("Different Variables:")
            for k, (v1, v2) in self.different_variables.items():
                content.append(f"  {k}: {v1} != {v2}")

        # Derived
        if self.missing_derived:
            content.append(
                "Missing Derived: {}".format(", ".join(self.missing_derived))
            )
        if self.different_derived:
            content.append("Different Derived:")
            for k, diff in self.different_derived.items():
                content.append(f"  {k}:")
                if diff.args1 != diff.args2:
                    content.append(f"    Args: {diff.args1} != {diff.args2}")

        # Reactions
        if self.missing_reactions:
            content.append(
                "Missing Reactions: {}".format(", ".join(self.missing_reactions))
            )
        if self.different_reactions:
            content.append("Different Reactions:")
            for k, diff in self.different_reactions.items():
                content.append(f"  {k}:")
                if diff.args1 != diff.args2:
                    content.append(f"    Args: {diff.args1} != {diff.args2}")
                if diff.stoichiometry1 != diff.stoichiometry2:
                    content.append(
                        f"    Stoichiometry: {diff.stoichiometry1} != {diff.stoichiometry2}"
                    )

        # Surrogates
        if self.missing_surrogates:
            content.append(
                "Missing Surrogates: {}".format(", ".join(self.missing_surrogates))
            )
        if self.different_surrogates:
            content.append("Different Surrogates:")
            for k, diff in self.different_surrogates.items():
                content.append(f"  {k}:")
                if diff.args1 != diff.args2:
                    content.append(f"    Args: {diff.args1} != {diff.args2}")
                if diff.stoichiometry1 != diff.stoichiometry2:
                    content.append(
                        f"    Stoichiometry: {diff.stoichiometry1} != {diff.stoichiometry2}"
                    )
        return "\n".join(content)


def _soft_eq_stoichiometries(
    s1: Mapping[str, float | Derived], s2: Mapping[str, float | Derived]
) -> bool:
    """Check if two stoichiometries are equal, ignoring the functions."""
    if s1.keys() != s2.keys():
        return False

    for k, v1 in s1.items():
        v2 = s2[k]
        if isinstance(v1, Derived):
            if not isinstance(v2, Derived):
                return False
            if v1.args != v2.args:
                return False
        elif v1 != v2:
            return False

    return True


def soft_eq(m1: Model, m2: Model) -> bool:
    """Check if two models are equal, ignoring the functions."""
    if m1._parameters != m2._parameters:  # noqa: SLF001
        return False
    if m1._variables != m2._variables:  # noqa: SLF001
        return False
    for k, d1 in m1._derived.items():  # noqa: SLF001
        if (d2 := m2._derived.get(k)) is None:  # noqa: SLF001
            return False
        if d1.args != d2.args:
            return False
    for k, r1 in m1._readouts.items():  # noqa: SLF001
        if (r2 := m2._readouts.get(k)) is None:  # noqa: SLF001
            return False
        if r1.args != r2.args:
            return False
    for k, v1 in m1._reactions.items():  # noqa: SLF001
        if (v2 := m2._reactions.get(k)) is None:  # noqa: SLF001
            return False
        if v1.args != v2.args:
            return False
        if not _soft_eq_stoichiometries(v1.stoichiometry, v2.stoichiometry):
            return False
    for k, s1 in m1._surrogates.items():  # noqa: SLF001
        if (s2 := m2._surrogates.get(k)) is None:  # noqa: SLF001
            return False
        if s1.args != s2.args:
            return False
        if s1.stoichiometries != s2.stoichiometries:
            return False
    return True


def model_diff(m1: Model, m2: Model) -> ModelDiff:
    """Compute the difference between two models."""
    diff = ModelDiff()

    for k, v1 in m1._parameters.items():  # noqa: SLF001
        if (v2 := m2._parameters.get(k)) is None:  # noqa: SLF001
            diff.missing_parameters.add(k)
        elif v1 != v2:
            diff.different_parameters[k] = (v1, v2)  # type: ignore

    for k, v1 in m1._variables.items():  # noqa: SLF001
        if (v2 := m2._variables.get(k)) is None:  # noqa: SLF001
            diff.missing_variables.add(k)
        elif v1 != v2:
            diff.different_variables[k] = (v1, v2)  # type: ignore

    for k, v1 in m1._readouts.items():  # noqa: SLF001
        if (v2 := m2._readouts.get(k)) is None:  # noqa: SLF001
            diff.missing_readouts.add(k)
        elif v1.args != v2.args:
            diff.different_readouts[k] = DerivedDiff(v1.args, v2.args)

    for k, v1 in m1._derived.items():  # noqa: SLF001
        if (v2 := m2._derived.get(k)) is None:  # noqa: SLF001
            diff.missing_derived.add(k)
        elif v1.args != v2.args:
            diff.different_derived[k] = DerivedDiff(v1.args, v2.args)

    for k, v1 in m1._reactions.items():  # noqa: SLF001
        if (v2 := m2._reactions.get(k)) is None:  # noqa: SLF001
            diff.missing_reactions.add(k)
        else:
            if v1.args != v2.args:
                rxn_diff: ReactionDiff = diff.different_reactions.get(k, ReactionDiff())
                rxn_diff.args1 = v1.args
                rxn_diff.args2 = v2.args
                diff.different_reactions[k] = rxn_diff
            if v1.stoichiometry != v2.stoichiometry:
                rxn_diff = diff.different_reactions.get(k, ReactionDiff())
                rxn_diff.stoichiometry1 = dict(v1.stoichiometry)
                rxn_diff.stoichiometry2 = dict(v2.stoichiometry)
                diff.different_reactions[k] = rxn_diff

    for k, v1 in m1._surrogates.items():  # noqa: SLF001
        if (v2 := m2._surrogates.get(k)) is None:  # noqa: SLF001
            diff.missing_surrogates.add(k)
        else:
            if v1.args != v2.args:
                rxn_diff = diff.different_surrogates.get(k, ReactionDiff())
                rxn_diff.args1 = v1.args
                rxn_diff.args2 = v2.args
            if v1.stoichiometries != v2.stoichiometries:
                rxn_diff = diff.different_surrogates.get(k, ReactionDiff())
                rxn_diff.stoichiometry1 = dict(v1.stoichiometries)  # type: ignore
                rxn_diff.stoichiometry2 = dict(v2.stoichiometries)  # type: ignore

    return diff
