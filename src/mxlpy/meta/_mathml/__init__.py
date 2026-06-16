from __future__ import annotations

from dataclasses import fields
from typing import Any

from .base import (
    _FIELD_TO_KEY,
    Base,
    Binary,
    Bool,
    E,
    Name,
    Nary,
    Nullary,
    Num,
    Pi,
    Unary,
)
from .binary import Implies, Pow
from .nary import (
    Add,
    And,
    Divide,
    Eq,
    GreaterEqual,
    GreaterThan,
    IntDivide,
    LessEqual,
    LessThan,
    Max,
    Min,
    Minus,
    Mul,
    Not,
    NotEqual,
    Or,
    Piecewise,
    Rem,
    Xor,
)
from .unary import (
    Abs,
    Acos,
    Acot,
    ArcCosh,
    ArcCoth,
    ArcCsc,
    ArcCsch,
    ArcSec,
    ArcSech,
    ArcSinh,
    ArcTanh,
    Asin,
    Atan,
    Ceiling,
    Cos,
    Cosh,
    Cot,
    Coth,
    Csc,
    Csch,
    Exp,
    Factorial,
    Floor,
    Ln,
    RateOf,
    Sec,
    Sech,
    Sin,
    Sinh,
    Tan,
    Tanh,
)
from .unary_special import Log, Sqrt

__all__ = [
    "Abs",
    "Acos",
    "Acot",
    "Add",
    "And",
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
    "Base",
    "Binary",
    "Bool",
    "Ceiling",
    "Cos",
    "Cosh",
    "Cot",
    "Coth",
    "Csc",
    "Csch",
    "Divide",
    "E",
    "Eq",
    "Exp",
    "Factorial",
    "Floor",
    "GreaterEqual",
    "GreaterThan",
    "Implies",
    "IntDivide",
    "LessEqual",
    "LessThan",
    "Ln",
    "Log",
    "Max",
    "Min",
    "Minus",
    "Mul",
    "Name",
    "Nary",
    "Not",
    "NotEqual",
    "Nullary",
    "Num",
    "Or",
    "Pi",
    "Piecewise",
    "Pow",
    "RateOf",
    "Rem",
    "Sec",
    "Sech",
    "Sin",
    "Sinh",
    "Sqrt",
    "Tan",
    "Tanh",
    "Unary",
    "Xor",
    "node_from_dict",
]

# Abstract intermediates that must never be instantiated from a dict.
_ABSTRACT_NODES = {"Base", "Nullary", "Unary", "Binary", "Nary"}
_NODE_REGISTRY: dict[str, type[Base]] = {}


def _build_registry() -> dict[str, type[Base]]:
    registry: dict[str, type[Base]] = {}
    stack = list(Base.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__name__ not in _ABSTRACT_NODES:
            registry[cls.__name__] = cls
    return registry


def node_from_dict(data: dict[str, Any]) -> Base:
    """Reconstruct an expression node tree from its arity-accurate dict form.

    Parameters
    ----------
    data
        Mapping with a ``"type"`` discriminator and the node's operands, as
        produced by :meth:`Base.to_dict`.

    Returns
    -------
    Base
        The reconstructed expression node.

    """
    if not _NODE_REGISTRY:
        _NODE_REGISTRY.update(_build_registry())

    node_type = data["type"]
    if (cls := _NODE_REGISTRY.get(node_type)) is None:
        msg = f"Unknown expression node type {node_type!r}"
        raise ValueError(msg)

    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        key = _FIELD_TO_KEY.get(f.name, f.name)
        raw = data[key]
        if isinstance(raw, dict) and "type" in raw:
            kwargs[f.name] = node_from_dict(raw)
        elif isinstance(raw, list):
            kwargs[f.name] = [node_from_dict(c) for c in raw]
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)
