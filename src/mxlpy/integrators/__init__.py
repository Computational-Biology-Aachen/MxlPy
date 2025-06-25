"""Integrator Package.

This package provides integrators for solving ordinary differential equations (ODEs).
It includes support for both Assimulo and Scipy integrators, with Assimulo being the default if available.
"""

from __future__ import annotations

from .int_diffrax import Diffrax
from .int_scipy import Scipy

try:
    from .int_assimulo import Assimulo

    DefaultIntegrator = Assimulo
except ImportError:
    DefaultIntegrator = Scipy

__all__ = [
    "Assimulo",
    "DefaultIntegrator",
    "Diffrax",
    "Scipy",
]
