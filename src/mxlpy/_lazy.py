# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Lazy access to optional-backend submodules (torch / keras / equinox)."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

__all__ = ["backend_getattr"]


def backend_getattr(
    package: str,
    backends: Mapping[str, tuple[str, str]],
) -> Callable[[str], ModuleType]:
    """Build a module ``__getattr__`` (PEP 562) importing backend submodules on first access.

    Parameters
    ----------
    package
        The package's ``__name__``.
    backends
        ``{public name: (submodule, extra)}``, e.g.
        ``{"torch": ("_torch", "torch")}`` makes ``package.torch`` import
        ``package._torch``, pointing at ``mxlpy[torch]`` if the backend's own
        dependencies are missing.

    Returns
    -------
    Callable[[str], ModuleType]
        The ``__getattr__`` to assign in the package.

    """

    def getattr_(name: str) -> ModuleType:
        if (spec := backends.get(name)) is None:
            msg = f"module {package!r} has no attribute {name!r}"
            raise AttributeError(msg)
        submodule, extra = spec
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except ModuleNotFoundError as e:
            # Only a missing third-party dependency, never a broken mxlpy module
            if e.name is None or e.name.split(".")[0] == "mxlpy":
                raise
            msg = (
                f"{package}.{name} requires {e.name!r}, which isn't installed. "
                f"Install it with `pip install mxlpy[{extra}]`."
            )
            raise ImportError(msg) from e
        # Cache, so __getattr__ isn't called again for this name
        setattr(sys.modules[package], name, module)
        return module

    return getattr_
