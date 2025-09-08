"""mxlpy: A Python Package for Metabolic Modeling and Analysis.

This package provides tools for creating, simulating and analyzing metabolic models
with features including:

Key Features:
    - Model creation and manipulation
    - Steady state and time-series simulations
    - Parameter fitting and optimization
    - Monte Carlo analysis
    - Metabolic Control Analysis (MCA)
    - SBML import/export support
    - Visualization tools

Core Components:
    Model: Core class for metabolic model representation
    Simulator: Handles model simulation and integration
    DefaultIntegrator: Standard ODE solver implementation
    LabelMapper: Maps between model components and labels
    Cache: Performance optimization through result caching

Simulation Features:
    - Steady state calculations
    - Time course simulations
    - Parameter scanning
    - Surrogate modeling with PyTorch

Analysis Tools:
    - Parameter fitting to experimental data
    - Monte Carlo methods for uncertainty analysis
    - Metabolic Control Analysis
    - Custom visualization functions

"""

from __future__ import annotations

import contextlib
import itertools as it
from typing import TYPE_CHECKING

import pandas as pd

from . import (
    compare,
    distributions,
    experimental,
    fit,
    fns,
    fuzzy,
    mc,
    mca,
    plot,
    report,
    sbml,
    scan,
    units,
)
from .integrators import DefaultIntegrator, Diffrax, Scipy
from .label_map import LabelMapper
from .linear_label_map import LinearLabelMapper
from .mc import Cache
from .model import Model
from .simulator import Simulator
from .symbolic import SymbolicModel, to_symbolic_model
from .types import (
    Derived,
    InitialAssignment,
    IntegratorProtocol,
    Parameter,
    Variable,
    unwrap,
)

with contextlib.suppress(ImportError):
    from .integrators import Assimulo

if TYPE_CHECKING:
    from mxlpy.types import ArrayLike

    from . import nn, npe, surrogates
else:
    from lazy_import import lazy_module

    nn = lazy_module("mxlpy.nn")
    npe = lazy_module("mxlpy.npe")
    surrogates = lazy_module("mxlpy.surrogates")


__all__ = [
    "Assimulo",
    "Cache",
    "DefaultIntegrator",
    "Derived",
    "Diffrax",
    "InitialAssignment",
    "IntegratorProtocol",
    "LabelMapper",
    "LinearLabelMapper",
    "Model",
    "Parameter",
    "Scipy",
    "Simulator",
    "SymbolicModel",
    "Variable",
    "cartesian_product",
    "compare",
    "distributions",
    "experimental",
    "fit",
    "fns",
    "fuzzy",
    "make_protocol",
    "mc",
    "mca",
    "nn",
    "npe",
    "plot",
    "report",
    "sbml",
    "scan",
    "surrogates",
    "symbolic",
    "to_symbolic_model",
    "units",
    "unwrap",
]


def cartesian_product(parameters: dict[str, ArrayLike]) -> pd.DataFrame:
    """Generate a cartesian product of the parameter values.

    Args:
        parameters: Dictionary containing parameter names and values.

    Returns:
        pd.DataFrame: DataFrame containing the cartesian product of the parameter values.

    """
    return pd.DataFrame(
        it.product(*parameters.values()),
        columns=list(parameters),
    )


def make_protocol(steps: list[tuple[float, dict[str, float]]]) -> pd.DataFrame:
    """Create protocol DataFrame from a dictionary of steps.

    Arguments:
        steps: dictionary of steps, where each key is the duration of
               the step in seconds and the value is a dictionary of all
               parameter values during that step.

    Examples:
        >>> make_protocol([
        ...     (1, {"k1": 1.0}),
        ...     (2, {"k1": 2.0}),
        ...     (3, {"k1": 1.0}),
        ... ])

        | Timedelta       |   k1 |
        |:----------------|-----:|
        | 0 days 00:00:01 |  1.0 |
        | 0 days 00:00:03 |  2.0 |
        | 0 days 00:00:06 |  1.0 |

        >>> make_protocol([
        ...     (1, {"k1": 1.0, "k2": 2.0}),
        ...     (2, {"k1": 2.0, "k2": 3.0}),
        ...     (3, {"k1": 1.0, "k2": 2.0}),
        ... ])

        | Timedelta       |   k1 |   k2 |
        |:----------------|-----:|-----:|
        | 0 days 00:00:01 |    1 |    2 |
        | 0 days 00:00:03 |    2 |    3 |
        | 0 days 00:00:06 |    1 |    2 |

    """
    data = {}
    t0 = pd.Timedelta(0)
    for step, pars in steps:
        t0 += pd.Timedelta(seconds=step)
        data[t0] = pars
    protocol = pd.DataFrame(data).T
    protocol.index.name = "Timedelta"
    return protocol
