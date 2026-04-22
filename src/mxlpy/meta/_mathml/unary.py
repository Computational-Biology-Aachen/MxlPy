from __future__ import annotations

from dataclasses import dataclass

from .base import Base, Unary

__all__ = [
    "Abs",
    "Acos",
    "Acot",
    "ArcCosh",
    "ArcCoth",
    "ArcCsc",
    "ArcCsch",
    "ArcSec",
    "ArcSech",
    "ArcSinh",
    "ArcTanh",
    "Asin",
    "Atan",
    "Ceiling",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Exp",
    "Factorial",
    "Floor",
    "Ln",
    "RateOf",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
]


@dataclass(slots=True)
class Abs(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Abs({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Ceiling(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Ceiling({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Exp(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Exp({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Factorial(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Factorial({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Floor(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Floor({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Ln(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Ln({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Sin(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Sin({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Cos(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Cos({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Tan(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Tan({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Sec(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Sec({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Csc(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Csc({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Cot(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Cot({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Asin(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Asin({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Acos(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Acos({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Atan(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Atan({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Acot(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Acot({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcSec(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcSec({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcCsc(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcCsc({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Sinh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Sinh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Cosh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Cosh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Tanh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Tanh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Sech(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Sech({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Csch(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Csch({self.child.to_mxlweb()})"


@dataclass(slots=True)
class Coth(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new Coth({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcSinh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcSinh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcCosh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcCosh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcTanh(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcTanh({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcCsch(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcCsch({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcSech(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcSech({self.child.to_mxlweb()})"


@dataclass(slots=True)
class ArcCoth(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new ArcCoth({self.child.to_mxlweb()})"


@dataclass(slots=True)
class RateOf(Unary):
    child: Base

    def to_mxlweb(self) -> str:
        return f"new RateOf({self.child.to_mxlweb()})"
