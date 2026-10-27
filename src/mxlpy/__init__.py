# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

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
import logging
from typing import TYPE_CHECKING

import pandas as pd

# Registered before the eager imports below, since some of them import these
# modules themselves (e.g. databases imports sbml)
if TYPE_CHECKING:
    from mxlpy.types import ArrayLike

    from . import nn, npe, sbml, surrogates
    from .integrators import Diffrax
else:
    from lazy_import import lazy_module

    nn = lazy_module("mxlpy.nn")
    npe = lazy_module("mxlpy.npe")
    sbml = lazy_module("mxlpy.sbml")
    surrogates = lazy_module("mxlpy.surrogates")

# dsl imported after fns and model to avoid circular-import ordering issues
from . import (
    compare,
    databases,
    distributions,
    dsl,
    experimental,
    fit,
    fns,
    fuzzy,
    mc,
    mca,
    plot,
    qa,
    report,
    runners,
    scan,
    sensitivity,
    units,
)
from ._kinetic_builder import KineticModelBuilder
from ._ode_builder import OdeModelBuilder
from ._steady_state_builder import SteadyStateModelBuilder
from .composition import compose
from .dsl import from_dsl
from .integrators import DefaultIntegrator, Scipy
from .integrators.abstract import AbstractIntegrator, IntegratorProtocol
from .label_map import LabelMapper
from .linear_label_map import LinearLabelMapper
from .mc import Cache
from .minimizers.abstract import AbstractMinimizer, MinimizerProtocol
from .npe.abstract import AbstractEstimator, EstimatorProtocol
from .runners import EvaluationRunner, KineticRunner, OdeRunner
from .serialize import load, save
from .simulation import Simulation
from .simulator import Simulator
from .surrogates.abstract import AbstractSurrogate, SurrogateProtocol
from .surrogates.abstract_ode import AbstractOdeSurrogate, OdeSurrogateProtocol
from .symbolic import SymbolicModel, to_symbolic_model
from .types import (
    Annotation,
    Derived,
    Event,
    InitialAssignment,
    OscillationDetected,
    Parameter,
    Slider,
    Variable,
)
from .unit_inference import Conflict, UnitInference

with contextlib.suppress(ImportError):
    from .integrators import Assimulo

# Alias for backward compatability
Model = KineticModelBuilder


def __getattr__(name: str) -> object:
    # Diffrax is resolved lazily, see mxlpy.integrators.__getattr__
    if name == "Diffrax":
        from . import integrators  # noqa: PLC0415

        return integrators.Diffrax
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AbstractEstimator",
    "AbstractIntegrator",
    "AbstractMinimizer",
    "AbstractOdeSurrogate",
    "AbstractSurrogate",
    "Annotation",
    "Assimulo",
    "Cache",
    "Conflict",
    "DefaultIntegrator",
    "EvaluationRunner",
    "Derived",
    "Diffrax",
    "EstimatorProtocol",
    "Event",
    "InitialAssignment",
    "IntegratorProtocol",
    "KineticModelBuilder",
    "KineticRunner",
    "LabelMapper",
    "LinearLabelMapper",
    "MinimizerProtocol",
    "Model",
    "OdeModelBuilder",
    "OdeRunner",
    "OdeSurrogateProtocol",
    "OscillationDetected",
    "Parameter",
    "Scipy",
    "Simulation",
    "Simulator",
    "Slider",
    "SteadyStateModelBuilder",
    "SurrogateProtocol",
    "SymbolicModel",
    "UnitInference",
    "Variable",
    "cartesian_product",
    "compare",
    "compose",
    "configure_default_logging",
    "databases",
    "distributions",
    "dsl",
    "experimental",
    "fit",
    "fns",
    "from_dsl",
    "fuzzy",
    "load",
    "make_protocol",
    "mc",
    "mca",
    "nn",
    "npe",
    "plot",
    "qa",
    "report",
    "runners",
    "save",
    "sbml",
    "scan",
    "sensitivity",
    "surrogates",
    "symbolic",
    "to_symbolic_model",
    "units",
]


def cartesian_product(parameters: dict[str, ArrayLike]) -> pd.DataFrame:
    """Generate a cartesian product of the parameter values.

    Parameters
    ----------
    parameters
        Dictionary containing parameter names and values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cartesian product of the parameter values.

    """
    return pd.DataFrame(
        it.product(*parameters.values()),
        columns=list(parameters),
    )


def make_protocol(steps: list[tuple[float, dict[str, float]]]) -> pd.DataFrame:
    """Create protocol DataFrame from a dictionary of steps.

    Parameters
    ----------
    steps
        dictionary of steps, where each key is the duration of
        the step in seconds and the value is a dictionary of all
        parameter values during that step.

    Examples
    --------
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


def configure_default_logging(
    fmt: str = "%(levelname)s:%(name)s:%(message)s",
    level: logging._Level = logging.DEBUG,
) -> None:
    """Create a default logging config."""
    logger = logging.getLogger("mxlpy")
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
