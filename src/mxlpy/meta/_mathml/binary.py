from __future__ import annotations

from dataclasses import dataclass

from .base import Base, Binary

__all__ = ["Implies", "Pow"]


@dataclass(slots=True)
class Pow(Binary):
    left: Base
    right: Base

    def to_mxlweb(self) -> str:
        return f"new Pow({self.left.to_mxlweb()}, {self.right.to_mxlweb()})"


@dataclass(slots=True)
class Implies(Binary):
    left: Base
    right: Base

    def to_mxlweb(self) -> str:
        return f"new Implies({self.left.to_mxlweb()}, {self.right.to_mxlweb()})"
