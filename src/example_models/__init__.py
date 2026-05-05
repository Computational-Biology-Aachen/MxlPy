"""Example models.

This package provides functions for retrieving various predefined metabolic models.
It includes models for example systems, linear chains, and specific metabolic pathways.

"""

from __future__ import annotations

from .example1 import get_example1
from .linear_chain import (
    get_lin_chain_two_circles,
    get_linear_chain_1v,
    get_linear_chain_2v,
)
from .lotka_volterra import create_model as get_lotka_volterra
from .phase_plane import get_phase_plane
from .poolman2000 import get_model as get_poolman2000
from .sir import get_sir, get_sird
from .tpi import get_tpi_ald_model
from .upper_glycolysis import get_model as get_upper_glycolysis

__all__ = [
    "get_example1",
    "get_lin_chain_two_circles",
    "get_linear_chain_1v",
    "get_linear_chain_2v",
    "get_lotka_volterra",
    "get_phase_plane",
    "get_poolman2000",
    "get_sir",
    "get_sird",
    "get_tpi_ald_model",
    "get_upper_glycolysis",
]
