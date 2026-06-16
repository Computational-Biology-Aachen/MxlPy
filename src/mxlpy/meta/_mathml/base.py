from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

__all__ = [
    "Base",
    "Binary",
    "Bool",
    "E",
    "Name",
    "Nary",
    "Nullary",
    "Num",
    "Pi",
    "Unary",
]

# Dataclass field name -> JSON key. ``Name`` stores its symbol in ``name`` but
# the shared mxlpy/mxlweb format spells leaf payloads ``value``.
_FIELD_TO_KEY = {"name": "value"}


@dataclass(slots=True)
class Base(ABC):
    @abstractmethod
    def to_mxlweb(self) -> str: ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise this node into a JSON-compatible, arity-accurate dict."""
        out: dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):  # type: ignore[arg-type]
            key = _FIELD_TO_KEY.get(f.name, f.name)
            value = getattr(self, f.name)
            if isinstance(value, Base):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [c.to_dict() for c in value]
            else:
                out[key] = value
        return out


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


@dataclass(slots=True)
class Pi(Nullary):
    def to_mxlweb(self) -> str:
        return "new Pi()"


@dataclass(slots=True)
class E(Nullary):
    def to_mxlweb(self) -> str:
        return "new E()"


@dataclass(slots=True)
class Bool(Nullary):
    value: bool

    def to_mxlweb(self) -> str:
        return f"new Bool({str(self.value).lower()})"
