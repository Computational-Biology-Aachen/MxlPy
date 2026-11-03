# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Array nodes: reductions and indexing over 1-D arrays (mxl-schemas ADR 0014).

Every other node broadcasts element-wise over arrays; these reduce an array to
a number or select from it. Each holds its operands in ``children``, in the
order given in the docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Base, Nary

__all__ = [
    "At",
    "Index",
    "MaxOf",
    "Mean",
    "MinOf",
    "Simpson",
    "Slice",
    "Std",
    "Sum",
]


def _children_mxlweb(name: str, children: list[Base]) -> str:
    return f"new {name}([{', '.join(c.to_mxlweb() for c in children)}])"


@dataclass
class Sum(Nary):
    """``[a]``: sum of all elements."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Sum", self.children)


@dataclass
class Mean(Nary):
    """``[a]``: arithmetic mean."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Mean", self.children)


@dataclass
class Std(Nary):
    """``[a]``: sample standard deviation (ddof = 1)."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Std", self.children)


@dataclass
class MaxOf(Nary):
    """``[a]``: largest element."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("MaxOf", self.children)


@dataclass
class MinOf(Nary):
    """``[a]``: smallest element."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("MinOf", self.children)


@dataclass
class Index(Nary):
    """``[a, i]``: element ``i`` (0-based, negative counts from the end)."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Index", self.children)


@dataclass
class Slice(Nary):
    """``[a, start, stop]``: elements ``start`` up to (excluding) ``stop``."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Slice", self.children)


@dataclass
class At(Nary):
    """``[a, t, times]``: the element of ``a`` at the sample of ``times`` nearest ``t``."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("At", self.children)


@dataclass
class Simpson(Nary):
    """``[y, x]``: integral of ``y`` over ``x`` by Simpson's rule (``scipy.integrate.simpson``)."""

    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return _children_mxlweb("Simpson", self.children)
