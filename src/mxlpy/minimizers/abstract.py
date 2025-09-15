"""Types for minimizers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from wadler_lindig import pformat

from mxlpy.types import Result

__all__ = [
    "AbstractMinimizer",
    "Bounds",
    "InitialGuess",
    "LOGGER",
    "LossFn",
    "MinimizerProtocol",
    "OptimisationState",
    "Residual",
    "mock_minimizer",
]

if TYPE_CHECKING:
    import pandas as pd


LOGGER = logging.getLogger(__name__)


class Residual(Protocol):
    """Protocol for steady state residual functions.

    This is the internal version, which is produced by partial
    application of `settings` of `ResidualProtocol`
    """

    def __call__(
        self,
        updates: dict[str, float],
    ) -> float:
        """Calculate residual error between model steady state and experimental data."""
        ...


type InitialGuess = dict[str, float]

type Bounds = dict[str, tuple[float | None, float | None]]

type LossFn = Callable[
    [
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
    ],
    float,
]


class MinimizerProtocol(Protocol):
    """Protocol for minimizers."""

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize fn."""
        ...


class AbstractMinimizer(ABC):
    """Abstract minimizer."""

    @abstractmethod
    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize fn."""
        ...


@dataclass
class OptimisationState:
    """Result of a minimization operation."""

    parameters: dict[str, float]
    residual: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


def mock_minimizer(
    residual_fn: Residual,  # noqa: ARG001
    p0: dict[str, float],
    bounds: Bounds | None,  # noqa: ARG001
) -> Result[OptimisationState]:
    """Mock minimizer for testing purposes."""
    return Result(OptimisationState(parameters=p0, residual=0.0))
