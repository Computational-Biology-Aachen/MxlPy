"""Integrator Package.

This package provides integrators for solving ordinary differential equations (ODEs).
It includes support for both Assimulo and Scipy integrators, with Assimulo being the default if available.
"""

from __future__ import annotations

import contextlib

from .abstract import IntegratorProtocol, IntegratorType
from .int_scipy import Scipy
from .utils import (
    OscillationDetector,
    detect_oscillations,
    detect_oscillations_scipy,
    no_oscillation_detection,
)

# Assimulo is optional dependency
try:
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo
except ImportError:
    DefaultIntegrator = Scipy

# Diffrax is optional dependency
with contextlib.suppress(ImportError):
    from .int_diffrax import Diffrax

__all__ = [
    "Assimulo",
    "DefaultIntegrator",
    "Diffrax",
    "IntegratorProtocol",
    "IntegratorType",
    "OscillationDetector",
    "Scipy",
    "detect_oscillations",
    "detect_oscillations_scipy",
    "no_oscillation_detection",
]
