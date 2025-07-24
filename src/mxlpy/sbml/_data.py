from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wadler_lindig import pformat

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "AtomicUnit",
    "Compartment",
    "CompositeUnit",
    "Compound",
    "Derived",
    "Function",
    "Parameter",
    "Reaction",
]


@dataclass
class AtomicUnit:
    kind: str
    exponent: int
    scale: int
    multiplier: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class CompositeUnit:
    sbml_id: str
    units: list

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Parameter:
    value: float
    is_constant: bool

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Compartment:
    name: str
    dimensions: int
    size: float
    units: str
    is_constant: bool

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Compound:
    compartment: str | None
    initial_amount: float
    substance_units: str | None
    has_only_substance_units: bool
    has_boundary_condition: bool
    is_constant: bool
    is_concentration: bool

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Derived:
    body: str
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Function:
    body: str
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Reaction:
    body: str
    stoichiometry: Mapping[str, float | str]
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)
