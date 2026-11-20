# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Symbolic utilities."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .jacobian import jacobian_fn
from .symbolic_model import SymbolicModel, to_symbolic_model

if TYPE_CHECKING:
    from .strikepy import check_identifiability


def __getattr__(name: str) -> object:
    # strikepy needs symbtools, which simulating a model doesn't
    if name == "check_identifiability":
        return importlib.import_module(f"{__name__}.strikepy").check_identifiability
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "SymbolicModel",
    "check_identifiability",
    "jacobian_fn",
    "to_symbolic_model",
]
