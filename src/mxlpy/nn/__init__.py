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
    from lazy_import import lazy_module

    equinox = lazy_module(
        "mxlpy.nn._equinox",
        error_strings={"module": "equinox", "install_name": "mxlpy[equinox]"},
    )
    keras = lazy_module(
        "mxlpy.nn._keras",
        error_strings={"module": "keras", "install_name": "mxlpy[tf]"},
    )
    torch = lazy_module(
        "mxlpy.nn._torch",
        error_strings={"module": "torch", "install_name": "mxlpy[torch]"},
    )


__all__ = [
    "keras",
    "torch",
]
