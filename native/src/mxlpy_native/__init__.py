# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Compiled solvers for MxlPy's Native integrator."""

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

__all__ = ["library_path"]


def library_path() -> Path:
    """Return the path of the compiled solvers, for loading with ctypes."""
    here = Path(__file__).parent
    for suffix in EXTENSION_SUFFIXES:
        if (path := here / f"_solvers{suffix}").exists():
            return path
    msg = f"compiled solvers not found in {here}"
    raise FileNotFoundError(msg)
