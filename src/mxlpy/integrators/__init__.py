# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Integrator Package.

This package provides integrators for solving ordinary differential equations (ODEs).
The default integrator is the first one available of Assimulo (conda only),
Sundials (`mxlpy[sundials]`) and Scipy. Both Assimulo and Sundials run CVODE.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from .abstract import IntegratorProtocol, IntegratorType
from .int_scipy import Scipy
from .utils import (
    OscillationDetector,
    detect_oscillations,
    detect_oscillations_scipy,
    no_oscillation_detection,
)

DefaultIntegrator: type = Scipy

# Assimulo and sundials4py are optional dependencies
with contextlib.suppress(ImportError):
    from .int_sundials import Sundials

    DefaultIntegrator = Sundials
with contextlib.suppress(ImportError):
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo

if TYPE_CHECKING:
    from .int_diffrax import Diffrax
    from .int_native import Native


def __getattr__(name: str) -> object:
    # Diffrax is an optional dependency that pulls in jax/equinox (~135 MB),
    # so it is only imported on first access
    if name == "Diffrax":
        try:
            from .int_diffrax import Diffrax  # noqa: PLC0415
        except ImportError as e:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg) from e
        return Diffrax
    # Native needs llvmlite or wasmtime, so it stays unavailable without them
    if name == "Native":
        try:
            from .int_native import Native  # noqa: PLC0415
        except ImportError as e:
            raise AttributeError(str(e)) from e
        return Native
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "Assimulo",
    "DefaultIntegrator",
    "Diffrax",
    "IntegratorProtocol",
    "IntegratorType",
    "Native",
    "OscillationDetector",
    "Scipy",
    "Sundials",
    "detect_oscillations",
    "detect_oscillations_scipy",
    "no_oscillation_detection",
]
