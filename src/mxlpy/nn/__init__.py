# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Collection of neural network architectures."""

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


__all__ = [
    "equinox",
    "keras",
    "torch",
]
