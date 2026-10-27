# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting types."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Protocol

import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.runners import default_runner

__all__ = [
    "EnsembleFit",
    "Fit",
    "FitResidual",
    "FitSettings",
    "GroupFit",
    "InitialGuess",
    "JointFit",
    "LOGGER",
    "MixedSettings",
]


if TYPE_CHECKING:
    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy._steady_state_builder import SteadyStateModelBuilder
    from mxlpy.minimizers.abstract import LossFn
    from mxlpy.runners import SteadyStateRunner
    from mxlpy.simulation import Simulation

LOGGER = logging.getLogger(__name__)

type InitialGuess = dict[str, float]

# Defaults to the KineticModelBuilder, so a bare `Fit` keeps its meaning
M = TypeVar(
    "M",
    bound="KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder",
    default="KineticModelBuilder",
)


class FitResidual(Protocol):
    """Protocol for fitting function.

    This is different to the minimizer.Residual, because we are wrapping
    `_Settings` with all fit functions.
    """

    def __call__(
        self,
        updates: dict[str, float],
        settings: _Settings[Any],
    ) -> float:
        """Do minimization step."""
        ...


@dataclass
class FitSettings:
    """Settings for a fit."""

    model: KineticModelBuilder
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None = None
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class MixedSettings:
    """Settings for a fit."""

    model: KineticModelBuilder
    data: pd.Series | pd.DataFrame
    residual_fn: FitResidual
    y0: dict[str, float] | None = None
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


def _pick_runner(
    runner: SteadyStateRunner[Any, Any] | None,
    fallback: SteadyStateRunner[Any, Any] | None,
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
) -> SteadyStateRunner[Any, Any]:
    """Pick the per-settings runner, else the fallback, else the default runner."""
    if runner is not None:
        return runner
    if fallback is not None:
        return fallback
    return default_runner(model)


@dataclass
class _Settings[M: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder]:
    """Non user-facing version of FitSettings."""

    model: M
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None
    runner: SteadyStateRunner[Any, Any]
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
        """Calculate loss between prediction and stored data.

        Parameters
        ----------
        prediction
            Model prediction to compare against experimental data.

        Returns
        -------
        float
            Loss value computed by the configured loss function.

        """
        # The standard deviation of fewer than two data points is undefined
        if self.standard_scale and len(self.data) > 1:
            return self.loss_fn(self.data_scaled, (prediction - self.mean) / self.scale)
        return self.loss_fn(self.data, prediction)


@dataclass
class Fit(Generic[M]):
    """Result of a fit operation."""

    model: M
    best_pars: dict[str, float]
    loss: float

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class GroupFit:
    """Result of a carousel fit operation."""

    fits: list[Fit]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_losses(self) -> pd.Series:
        """Get losses of all fits."""
        return pd.Series({i: f.loss for i, f in enumerate(self.fits)})

    def get_best_fit(self) -> Fit:
        """Get the best fit from the carousel."""
        return min(self.fits, key=lambda x: x.loss)


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
