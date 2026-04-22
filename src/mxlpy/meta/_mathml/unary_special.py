from __future__ import annotations

from dataclasses import dataclass

from .base import Base

__all__ = ["Log", "Sqrt"]


@dataclass
class Log(Base):
    child: Base
    base: Base

    def to_mxlweb(self) -> str:
        return f"new Log({self.child.to_mxlweb()}, {self.base.to_mxlweb()})"


@dataclass
class Sqrt(Base):
    child: Base
    base: Base

    def to_mxlweb(self) -> str:
        return f"new Sqrt({self.child.to_mxlweb()}, {self.base.to_mxlweb()})"
