# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting types."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Protocol

import pandas as pd
from typing_extensions import TypeVar
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit.losses import GaussianNll, _is_likelihood
from mxlpy.runners import default_runner
from mxlpy.simulator import _normalise_protocol_index

__all__ = [
    "Condition",
    "Data",
    "EnsembleFit",
    "Fit",
    "FitResidual",
    "FitSettings",
    "GroupFit",
    "InitialGuess",
    "JointFit",
    "LOGGER",
    "MixedSettings",
    "PostprocessFn",
]


if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.losses import Sigma
    from mxlpy.minimizers.abstract import Bounds, LossFn
    from mxlpy.runners import SteadyStateRunner

LOGGER = logging.getLogger(__name__)

type InitialGuess = dict[str, float]

# Maps the combined simulation result into the space of the data, applied before the loss
type PostprocessFn = Callable[[pd.DataFrame], pd.DataFrame]

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


@dataclass(frozen=True, eq=False)
class Condition:
    """One data set and the experimental condition it was measured in.

    Functions fitting several data sets at once take a mapping of names to
    conditions (or plain data, a condition without settings).

    Parameters
    ----------
    data
        A Series for steady-state data, a DataFrame indexed by time for a
        time course (with `protocol`: a protocol time course)
    y0
        Initial values of the condition. Fitted initial values still override
        them
    protocol
        Protocol of a protocol time course, see `fit.protocol_time_course`
    parameters
        Parameter values set by the condition (e.g. an inhibitor
        concentration). They override fitted values in this condition, so
        parameters set by every condition can't be estimated
    local
        Condition-specific estimates: in this condition the model parameter
        (key) takes the value of the estimated name (value), e.g.
        `{"k1": "k1_mut"}` estimates `k1` separately for a mutant. The name
        has to be estimated (in `p0` / the fitted values) like a parameter
    sigma
        Standard deviation of the measurement noise (scalar, per observable
        or per data point). If set, the condition's loss is
        `GaussianNll(sigma)`, regardless of the function's `loss_fn`

    Examples
    --------
        >>> fit.Condition(df_inhibited, parameters={"I": 10.0}, sigma=0.05)
        >>> fit.Condition(df_mutant, local={"k1": "k1_mut"})

    """

    data: pd.Series | pd.DataFrame
    _: KW_ONLY
    y0: Mapping[str, float] | None = None
    protocol: pd.DataFrame | None = None
    parameters: Mapping[str, float] | None = None
    local: Mapping[str, str] | None = None
    sigma: Sigma | None = None

    def loss(self, loss_fn: LossFn | None) -> LossFn:
        """Loss of the condition: `GaussianNll(sigma)` if it has a sigma, else `loss_fn`."""
        if self.sigma is not None:
            return GaussianNll(self.sigma)
        if loss_fn is None:
            msg = "Pass loss_fn, or a sigma for every condition"
            raise ValueError(msg)
        return loss_fn

    @property
    def overridden(self) -> set[str]:
        """Model names whose values the condition sets or renames."""
        return set(self.parameters or {}) | set(self.local or {})


type Data = (
    pd.Series
    | pd.DataFrame
    | Condition
    | Mapping[str, pd.Series | pd.DataFrame | Condition]
)


def as_conditions(
    data: Data,
    *,
    y0: Mapping[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
) -> dict[str, Condition]:
    """Conditions by name of any form of `data`.

    Single data (or a single condition) is named `data`. `y0` and `protocol`
    apply to conditions without their own.
    """
    items: dict[str, pd.Series | pd.DataFrame | Condition]
    if isinstance(data, (pd.Series, pd.DataFrame, Condition)):
        items = {"data": data}
    else:
        items = dict(data)
        if not items:
            msg = "No data given"
            raise ValueError(msg)
    out = {}
    for name, item in items.items():
        cond = item if isinstance(item, Condition) else Condition(item)
        if (y0 is not None and cond.y0 is None) or (
            protocol is not None and cond.protocol is None
        ):
            cond = replace(
                cond,
                y0=cond.y0 if cond.y0 is not None else y0,
                protocol=cond.protocol if cond.protocol is not None else protocol,
            )
        if cond.protocol is not None and not isinstance(cond.data, pd.DataFrame):
            msg = f"Condition {name!r}: a protocol needs a time course (DataFrame)"
            raise TypeError(msg)
        out[str(name)] = cond
    return out


@dataclass
class FitSettings(Generic[M]):
    """Settings of one model:data pair of a joint fit."""

    model: M
    data: pd.Series | pd.DataFrame
    y0: dict[str, float] | None = None
    runner: SteadyStateRunner[M, Any] | None = None
    loss_fn: LossFn | None = None
    postprocess_fn: PostprocessFn | None = None
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
    postprocess_fn: PostprocessFn | None = None
    protocol: pd.DataFrame | None = None


# Weights of hidden-state blocks are signed, unlike the minimizers' default
WEIGHT_BOUNDS = (-1e6, 1e6)


def with_weight_bounds(
    models: Iterable[KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder],
    names: Iterable[str],
    bounds: Bounds | None,
) -> Bounds:
    """`bounds` plus `WEIGHT_BOUNDS` for the weights in `names` without bounds."""
    weights = {w for m in models for w in m.get_weight_names()}
    given = {} if bounds is None else dict(bounds)
    return {k: WEIGHT_BOUNDS for k in names if k in weights and k not in given} | given


def without_weights(
    models: Iterable[KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder],
    names: Iterable[str],
    transform: Transform | None,
) -> Transform | None:
    """A transform for all `names` as one per name, leaving weights linear.

    Weights are signed, so they can't be log-transformed.
    """
    if not isinstance(transform, str):
        return transform
    weights = {w for m in models for w in m.get_weight_names()}
    if not weights:
        return transform
    return {k: transform for k in names if k not in weights}


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
    p_names = set(model.get_addressable_names())
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
    postprocess_fn: PostprocessFn | None,
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
        postprocess_fn=(
            settings.postprocess_fn
            if settings.postprocess_fn is not None
            else postprocess_fn
        ),
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
    postprocess_fn: PostprocessFn | None = None

    @property
    def scaled(self) -> bool:
        """Whether data and prediction are compared after standard scaling."""
        return (
            self.standard_scale
            and len(self.data) > 1
            and not _is_likelihood(self.loss_fn)
        )

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
        # The standard deviation of fewer than two data points is undefined.
        # Likelihoods compare in data units, as their sigma is in data units
        if self.scaled:
            return self.loss_fn(self.data_scaled, (prediction - self.mean) / self.scale)
        return self.loss_fn(self.data, prediction)


@dataclass
class Fit(Generic[M]):
    """Result of a fit operation.

    Attributes
    ----------
    model
        The fitted model (a copy, unless fitted with `as_deepcopy=False`)
    best_pars
        Fitted parameters and initial values
    loss
        Loss of the data at `best_pars`, without the penalties
    penalty
        Sum of the priors' and penalties' terms at `best_pars`, so the
        minimized objective is `loss + penalty`

    """

    model: M
    best_pars: dict[str, float]
    loss: float
    penalty: float = 0.0

    @property
    def objective(self) -> float:
        """The minimized objective, `loss + penalty`."""
        return self.loss + self.penalty

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class GroupFit(Generic[M]):
    """Result of a multi-start fit.

    Attributes
    ----------
    fits
        Successful fits by start (the row labels of `p0`)
    failures
        Reason of every failed start by start, e.g. the minimizer's message
        or `timeout`

    """

    fits: dict[Hashable, Fit[M]]
    failures: dict[Hashable, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def n_failed(self) -> int:
        """Number of failed starts."""
        return len(self.failures)

    def get_losses(self) -> pd.Series:
        """Get losses of all successful fits by start, without their penalties."""
        return pd.Series({k: f.loss for k, f in self.fits.items()}, dtype=float)

    def get_objectives(self) -> pd.Series:
        """Get the minimized objectives (loss + penalty) of all successful fits by start."""
        return pd.Series({k: f.objective for k, f in self.fits.items()}, dtype=float)

    def get_best_fit(self) -> Fit[M]:
        """Get the fit with the lowest objective (loss + penalty)."""
        if not self.fits:
            reasons = sorted(set(self.failures.values()))
            msg = f"All {self.n_failed} starts failed: {reasons}"
            raise ValueError(msg)
        return min(self.fits.values(), key=lambda x: x.objective)


@dataclass
class EnsembleFit(Generic[M]):
    """Result of a carousel fit operation."""

    fits: list[Fit[M]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_best_fit(self) -> Fit[M]:
        """Get the fit with the lowest objective (loss + penalty)."""
        return min(self.fits, key=lambda x: x.objective)


@dataclass
class JointFit:
    """Result of joint fit operation.

    Attributes
    ----------
    best_pars
        Fitted parameters and initial values
    loss
        Summed loss of all data at `best_pars`, without the penalties
    penalty
        Sum of the priors' and penalties' terms at `best_pars`

    """

    best_pars: dict[str, float]
    loss: float
    penalty: float = 0.0

    @property
    def objective(self) -> float:
        """The minimized objective, `loss + penalty`."""
        return self.loss + self.penalty

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)
