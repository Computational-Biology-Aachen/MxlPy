from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = ["Base", "Binary", "Name", "Nary", "Nullary", "Num", "Unary"]


@dataclass(slots=True)
class Base(ABC):
    @abstractmethod
    def to_mxlweb(self) -> str: ...


@dataclass(slots=True)
class Nullary(Base): ...


@dataclass(slots=True)
class Unary(Base): ...


@dataclass(slots=True)
class Binary(Base): ...


@dataclass(slots=True)
class Nary(Base): ...


###############################################################################
## Nullary fns
## Also didn't belive that is the term, but check it
## https://en.wikipedia.org/wiki/Arity
###############################################################################


@dataclass(slots=True)
class Name(Nullary):
    name: str

    def to_mxlweb(self) -> str:
        return f"new Name({self.name})"


@dataclass(slots=True)
class Num(Nullary):
    value: float

    def to_mxlweb(self) -> str:
        return f"new Num({self.value})"
