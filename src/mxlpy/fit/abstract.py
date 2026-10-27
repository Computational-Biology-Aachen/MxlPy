# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting types."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Protocol

import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.runners import default_runner
from mxlpy.simulator import _normalise_protocol_index

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
    from mxlpy.minimizers.abstract import LossFn
    from mxlpy.runners import SteadyStateRunner

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
class FitSettings(Generic[M]):
    """Settings of one model:data pair of a joint fit."""

    model: M
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None = None
    runner: SteadyStateRunner[M, Any] | None = None
    loss_fn: LossFn | None = None
    protocol: pd.DataFrame | None = None


@dataclass
class MixedSettings(Generic[M]):
    """Settings of one model:data:residual triple of a mixed joint fit."""

    model: M
    data: pd.Series | pd.DataFrame
    residual_fn: FitResidual
    y0: dict[str, float] | None = None
    runner: SteadyStateRunner[M, Any] | None = None
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


def _split_p0(
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
    p0: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Split initial guesses into parameter and variable names."""
    p_names = set(model.get_parameter_names())
    v_names = (
        set()
        if isinstance(model, SteadyStateModelBuilder)
        else set(model.get_variable_names())
    )
    return [i for i in p0 if i in p_names], [i for i in p0 if i in v_names]


def _joint_settings(
    settings: FitSettings[Any] | MixedSettings[Any],
    *,
    p0: dict[str, float],
    y0: dict[str, float] | None,
    runner: SteadyStateRunner[Any, Any] | None,
    loss_fn: LossFn,
    as_deepcopy: bool,
    standard_scale: bool,
    needs_dynamics: bool,
) -> _Settings[Any]:
    """Resolve the settings of one model of a joint fit.

    Per-model settings take precedence over the function-level ones. The
    function-level `y0` is skipped for `SteadyStateModelBuilder` models, so
    they can be fitted jointly with integrable models.
    """
    model = settings.model
    is_steady_state = isinstance(model, SteadyStateModelBuilder)
    if is_steady_state and needs_dynamics:
        msg = "A SteadyStateModelBuilder can only be fitted to steady-state data"
        raise TypeError(msg)
    if is_steady_state and settings.y0 is not None:
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)

    p_names, v_names = _split_p0(model, p0)
    return _Settings(
        model=deepcopy(model) if as_deepcopy else model,
        data=settings.data,
        y0=settings.y0 if settings.y0 is not None else None if is_steady_state else y0,
        runner=_pick_runner(settings.runner, runner, model),
        loss_fn=settings.loss_fn if settings.loss_fn is not None else loss_fn,
        protocol=(
            _normalise_protocol_index(protocol)
            if (protocol := settings.protocol) is not None
            else None
        ),
        residual_fn=getattr(settings, "residual_fn", None),
        p_names=p_names,
        v_names=v_names,
        standard_scale=standard_scale,
    )


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
class GroupFit(Generic[M]):
    """Result of a carousel fit operation."""

    fits: list[Fit[M]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_losses(self) -> pd.Series:
        """Get losses of all fits."""
        return pd.Series({i: f.loss for i, f in enumerate(self.fits)})

    def get_best_fit(self) -> Fit[M]:
        """Get the best fit from the carousel."""
        return min(self.fits, key=lambda x: x.loss)


@dataclass
class EnsembleFit(Generic[M]):
    """Result of a carousel fit operation."""

    fits: list[Fit[M]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_best_fit(self) -> Fit[M]:
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
