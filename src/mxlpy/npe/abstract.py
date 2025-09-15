"""NPE Interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wadler_lindig import pformat

__all__ = ["AbstractEstimator"]

if TYPE_CHECKING:
    import pandas as pd


@dataclass(kw_only=True)
class AbstractEstimator:
    """Abstract class for parameter estimation using neural networks."""

    parameter_names: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @abstractmethod
    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
