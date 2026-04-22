from __future__ import annotations

from dataclasses import dataclass, field

from .base import Base, Nary

__all__ = [
    "Add",
    "And",
    "Divide",
    "Eq",
    "GreaterEqual",
    "GreaterThan",
    "IntDivide",
    "LessEqual",
    "LessThan",
    "Max",
    "Min",
    "Minus",
    "Mul",
    "Not",
    "NotEqual",
    "Or",
    "Piecewise",
    "Rem",
    "Xor",
]


def _children_mxlweb(children: list[Base]) -> str:
    return f"[{', '.join(c.to_mxlweb() for c in children)}]"


@dataclass
class Max(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Max({_children_mxlweb(self.children)})"


@dataclass
class Min(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Min({_children_mxlweb(self.children)})"


@dataclass
class Piecewise(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Piecewise({_children_mxlweb(self.children)})"


@dataclass
class Rem(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Rem({_children_mxlweb(self.children)})"


@dataclass
class And(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new And({_children_mxlweb(self.children)})"


@dataclass
class Not(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Not({_children_mxlweb(self.children)})"


@dataclass
class Or(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Or({_children_mxlweb(self.children)})"


@dataclass
class Xor(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Xor({_children_mxlweb(self.children)})"


@dataclass
class Eq(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Eq({_children_mxlweb(self.children)})"


@dataclass
class GreaterEqual(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new GreaterEqual({_children_mxlweb(self.children)})"


@dataclass
class GreaterThan(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new GreaterThan({_children_mxlweb(self.children)})"


@dataclass
class LessEqual(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new LessEqual({_children_mxlweb(self.children)})"


@dataclass
class LessThan(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new LessThan({_children_mxlweb(self.children)})"


@dataclass
class NotEqual(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new NotEqual({_children_mxlweb(self.children)})"


@dataclass
class Add(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Add({_children_mxlweb(self.children)})"


@dataclass
class Minus(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Minus({_children_mxlweb(self.children)})"


@dataclass
class Mul(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Mul({_children_mxlweb(self.children)})"


@dataclass
class Divide(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new Divide({_children_mxlweb(self.children)})"


@dataclass
class IntDivide(Nary):
    children: list[Base] = field(default_factory=list)

    def to_mxlweb(self) -> str:
        return f"new IntDivide({_children_mxlweb(self.children)})"
