"""Fitting types."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Protocol

from wadler_lindig import pformat

from mxlpy.model import Model

__all__ = [
    "EnsembleFit",
    "Fit",
    "FitResidual",
    "FitSettings",
    "InitialGuess",
    "JointFit",
    "LOGGER",
    "MixedSettings",
]

if TYPE_CHECKING:
    import pandas as pd

    from mxlpy.integrators.abstract import IntegratorType
    from mxlpy.minimizers.abstract import LossFn
    from mxlpy.model import Model

LOGGER = logging.getLogger(__name__)

type InitialGuess = dict[str, float]


class FitResidual(Protocol):
    """Protocol for fitting function.

    This is different to the minimizer.Residual, because we are wrapping
    `_Settings` with all fit functions.
    """

    def __call__(
        self,
        updates: dict[str, float],
        settings: _Settings,
    ) -> float:
        """Do minimization step."""
        ...


@dataclass
class FitSettings:
    """Settings for a fit."""

    model: Model
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None = None
    integrator: IntegratorType | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class MixedSettings:
    """Settings for a fit."""

    model: Model
    data: pd.Series | pd.DataFrame
    residual_fn: FitResidual
    y0: dict[str, float] | None = None
    integrator: IntegratorType | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class _Settings:
    """Non user-facing version of FitSettings."""

    model: Model
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None
    integrator: IntegratorType | None
    loss_fn: LossFn
    p_names: list[str]
    v_names: list[str]
    standard_scale: bool
    protocol: pd.DataFrame | None = None
    residual_fn: FitResidual | None = None

    @cached_property
    def mean(self) -> pd.Series | float:
        return self.data.mean()

    @cached_property
    def scale(self) -> pd.Series | float:
        return self.data.std()

    @cached_property
    def data_scaled(self) -> pd.Series | pd.DataFrame:
        return (self.data - self.mean) / self.scale

    def loss(self, prediction: pd.Series | pd.DataFrame) -> float:
        if self.standard_scale:
            return self.loss_fn(self.data_scaled, (prediction - self.mean) / self.scale)
        return self.loss_fn(self.data, prediction)


@dataclass
class Fit:
    """Result of a fit operation."""

    model: Model
    best_pars: dict[str, float]
    loss: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class EnsembleFit:
    """Result of a carousel fit operation."""

    fits: list[Fit]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_best_fit(self) -> Fit:
        """Get the best fit from the carousel."""
        return min(self.fits, key=lambda x: x.loss)


@dataclass
class JointFit:
    """Result of joint fit operation."""

    best_pars: dict[str, float]
    loss: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)
