# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Surrogate Models Module.

This module provides classes and functions for creating and training surrogate models
for metabolic simulations. It includes functionality for both steady-state and time-series
data using neural networks.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import contextlib

    with contextlib.suppress(ImportError):
        from . import _equinox as equinox
        from . import _keras as keras
        from . import _torch as torch
else:
    from mxlpy._lazy import backend_getattr

    __getattr__ = backend_getattr(
        __name__,
        {
            "equinox": ("_equinox", "jax"),
            "keras": ("_keras", "keras"),
            "torch": ("_torch", "torch"),
        },
    )


from . import _poly as poly
from . import _qss as qss
from ._io import load

__all__ = [
    "equinox",
    "keras",
    "load",
    "poly",
    "qss",
    "torch",
]
