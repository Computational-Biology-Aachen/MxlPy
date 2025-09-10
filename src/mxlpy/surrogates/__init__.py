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
    from lazy_import import lazy_module

    equinox = lazy_module(
        "mxlpy.surrogates._equinox",
        error_strings={"module": "equinox", "install_name": "mxlpy[equinox]"},
    )
    keras = lazy_module(
        "mxlpy.surrogates._keras",
        error_strings={"module": "keras", "install_name": "mxlpy[tf]"},
    )
    torch = lazy_module(
        "mxlpy.surrogates._torch",
        error_strings={"module": "torch", "install_name": "mxlpy[torch]"},
    )


from . import _poly as poly
from . import _qss as qss

__all__ = [
    "keras",
    "poly",
    "qss",
    "torch",
]
