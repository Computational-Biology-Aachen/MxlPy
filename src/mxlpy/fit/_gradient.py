# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Residuals providing residual vectors and derivatives for gradient-based minimizers.

Minimizers detect the optional `residuals`, `jacobian` and `gradient` methods,
so plain scalar residuals keep working unchanged.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from mxlpy import sensitivities as sens_mod
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy.fit import losses
from mxlpy.fit.residuals import (
    _align_time,
    _apply_updates,
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mxlpy.fit.abstract import FitResidual, _Settings
    from mxlpy.minimizers.abstract import Bounds, Residual
    from mxlpy.types import Array

__all__ = ["GradientResidual", "Sensitivities", "Transform", "transformed"]

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()

type Sensitivities = Literal["forward", "finite_difference"]
type Transform = Literal["log", "linear"] | dict[str, Literal["log", "linear"]]

_DEFAULT_BOUNDS = (1e-6, 1e6)


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOGGER.warning(msg)


def _loss_derivative(
    loss_fn: Callable[..., float], n: int
) -> Callable[[Array], Array] | None:
    """Analytic d loss / d residual for the built-in losses, else None."""
    if loss_fn is losses.mean_squared:
        return lambda r: 2 * r / n
    if loss_fn is losses.rmse:

        def d_rmse(r: Array) -> Array:
            value = np.sqrt(np.mean(np.square(r)))
            return r / (n * value) if value > 0 else np.zeros_like(r)

        return d_rmse
    if loss_fn is losses.mae:
        return lambda r: np.sign(r) / n
    return None


class GradientResidual:
    """Residual of a single fit, additionally providing derivatives.

    The residual vector is the (standard-scaled, if enabled) prediction minus
    the data, flattened, without missing data. Its Jacobian comes from the
    forward sensitivities of the simulation if possible, else from central
    finite differences of whole simulations.

    Parameters
    ----------
    residual_fn
        One of the default residual functions of `mxlpy.fit.residuals`
    settings
        Settings of the fit
    sensitivities
        `forward` or `finite_difference`
    rel_step
        Relative step of the finite differences

    """

    def __init__(
        self,
        residual_fn: FitResidual,
        settings: _Settings[Any],
        *,
        sensitivities: Sensitivities,
        rel_step: float = 1e-3,
    ) -> None:
        kinds = {
            steady_state_residual: "steady_state",
            time_course_residual: "time_course",
            protocol_time_course_residual: "protocol_time_course",
        }
        if (kind := kinds.get(cast(Any, residual_fn))) is None:
            msg = "Sensitivities need one of the residual functions of mxlpy.fit.residuals"
            raise ValueError(msg)
        self.kind = kind
        self.fn = partial(residual_fn, settings=settings)
        self.settings = settings
        self.rel_step = rel_step

        data = settings.data
        self.names: list[str] = list(
            data.index if isinstance(data, pd.Series) else data.columns
        )
        self.scaled = settings.standard_scale and len(data) > 1
        # Missing data and constant columns (zero std, so NaN when scaled) are
        # left out, as the scalar losses skip them as well
        scaled = np.asarray(self._scale(data), dtype=float).ravel()
        self.mask = np.isfinite(scaled)
        self.data_vector = scaled[self.mask]

        self.forward = sensitivities == "forward"
        if self.forward and (reason := self._forward_problem()) is not None:
            _warn_once(
                f"Using finite-difference sensitivities for fitting, as {reason}"
            )
            self.forward = False

        self._key: tuple[tuple[str, float], ...] | None = None
        self._prediction: pd.DataFrame | pd.Series | None = None
        self._jac_key: tuple[tuple[str, float], ...] | None = None
        self._jac: Array | None = None

    def _forward_problem(self) -> str | None:
        s = self.settings
        if not isinstance(s.model, (KineticModelBuilder, OdeModelBuilder)):
            return "the model has no dynamics"
        if self.kind == "protocol_time_course":
            return "protocol fits don't support forward sensitivities yet"
        if s.postprocess_fn is not None:
            return "a postprocess_fn can't be differentiated"
        if not hasattr(s.runner, "integrator"):
            return "the runner doesn't expose its integrator"
        return None

    def _scale(self, values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        s = self.settings
        if self.scaled:
            return (values - s.mean) / s.scale
        return values

    ###########################################################################
    # Simulations
    ###########################################################################

    def _simulate(self, updates: dict[str, float]) -> pd.DataFrame | pd.Series | None:
        s = self.settings
        _apply_updates(updates, s)
        runner = cast(Any, s.runner)
        if self.kind == "steady_state":
            res = runner.steady_state(s.model)
        else:
            # Replicates share time points, which the integrators reject
            times = np.unique(np.asarray(s.data.index, dtype=float))
            if self.kind == "time_course":
                res = runner.time_course(s.model, times)
            else:
                res = runner.protocol_time_course(s.model, s.protocol, times)
        if isinstance(val := res.value, Exception):
            return None
        prediction = val.get_combined()
        if s.postprocess_fn is not None:
            prediction = s.postprocess_fn(prediction)
        if self.kind == "steady_state":
            return prediction.iloc[-1].loc[self.names]
        return _align_time(prediction, s.data.index).loc[:, self.names]

    def _vector(self, prediction: pd.DataFrame | pd.Series | None) -> Array:
        if prediction is None:
            return np.full(int(self.mask.sum()), np.nan)
        return (
            np.asarray(self._scale(prediction), dtype=float).ravel()[self.mask]
            - self.data_vector
        )

    def _predict(self, updates: dict[str, float]) -> pd.DataFrame | pd.Series | None:
        key = tuple(updates.items())
        if key != self._key:
            self._prediction = self._simulate(updates)
            self._key = key
        return self._prediction

    ###########################################################################
    # Residual protocol
    ###########################################################################

    def __call__(self, updates: dict[str, float]) -> float:
        """Scalar loss, the same as the wrapped residual function.

        Goes through the prediction cache, so a following `gradient` or
        `jacobian` call at the same point doesn't simulate again.
        """
        if (prediction := self._predict(updates)) is None:
            return np.inf
        return float(self.settings.loss(prediction))

    def residuals(self, updates: dict[str, float]) -> Array:
        """Residual vector, all `nan` if the simulation failed.

        Entries whose prediction isn't finite are 0, as the scalar losses skip
        them as well.
        """
        r = self._vector(self._predict(updates))
        if self._prediction is None:
            return r
        return np.where(np.isfinite(r), r, 0.0)

    def jacobian(self, updates: dict[str, float]) -> Array:
        """Derivative of the residual vector, (n_residuals, n_parameters)."""
        key = tuple(updates.items())
        if key == self._jac_key and self._jac is not None:
            return self._jac
        jac = self._forward(updates) if self.forward else None
        if jac is None:
            jac = self._finite_difference(updates)
        # Rows of entries the losses skip, matching `residuals`
        jac[~np.isfinite(self._vector(self._predict(updates)))] = 0.0
        self._jac_key, self._jac = key, jac
        return jac

    def gradient(self, updates: dict[str, float]) -> Array:
        """Derivative of the scalar loss w.r.t. the parameters."""
        r = self._vector(self._predict(updates))
        valid = np.isfinite(r)
        if self._prediction is None or not valid.any():
            return np.full(len(updates), np.nan)
        jac = self.jacobian(updates)
        return self._d_loss(r[valid], valid) @ jac[valid]

    ###########################################################################
    # Derivatives
    ###########################################################################

    def _d_loss(self, r: Array, valid: NDArray[np.bool_]) -> Array:
        """D loss / d residuals, analytic for the built-in losses.

        `r` only contains the `valid` entries of the residual vector.
        """
        if (fn := _loss_derivative(self.settings.loss_fn, len(r))) is not None:
            return fn(r)
        # Central differences of the loss itself, no simulations needed
        data = self._scale(self.settings.data)
        positions = np.flatnonzero(self.mask)[valid]
        data_vector = self.data_vector[valid]

        def loss_at(vector: Array) -> float:
            full = np.full(self.mask.shape, np.nan)
            full[positions] = vector + data_vector
            pred = (
                pd.Series(full, index=data.index)
                if isinstance(data, pd.Series)
                else pd.DataFrame(
                    full.reshape(data.shape), index=data.index, columns=data.columns
                )
            )
            return self.settings.loss_fn(data, pred)

        out = np.empty_like(r)
        for i in range(len(r)):
            h = self.rel_step * max(1.0, abs(r[i]))
            up, down = r.copy(), r.copy()
            up[i] += h
            down[i] -= h
            out[i] = (loss_at(up) - loss_at(down)) / (2 * h)
        return out

    def _forward(self, updates: dict[str, float]) -> Array | None:
        s = self.settings
        _apply_updates(updates, s)
        runner = cast(Any, s.runner)
        wrt = list(updates)
        try:
            if self.kind == "steady_state":
                res = sens_mod.steady_state(
                    s.model,
                    wrt,
                    integrator=runner.integrator,
                    method="forward",
                    tolerance=runner.tolerance,
                    rel_norm=runner.rel_norm,
                )
            else:
                # Replicates share time points, which the integrators reject
                res = sens_mod.time_course(
                    s.model,
                    np.unique(np.asarray(s.data.index, dtype=float)),
                    wrt,
                    integrator=runner.integrator,
                    method="forward",
                )
        except Exception as e:  # noqa: BLE001
            # Models the sensitivity codegen can't handle (events, Native,
            # unparsable functions, ...)
            _warn_once(f"Using finite-difference sensitivities for fitting, as {e}")
            self.forward = False
            return None
        if isinstance(sens := res.value, Exception):
            return None
        # The prediction itself always comes from the runner, like the loss,
        # as the augmented solve differs within the integration tolerance
        columns = []
        for name in wrt:
            d = sens.sensitivities[name].loc[:, self.names]
            d = (
                d.iloc[-1]
                if self.kind == "steady_state"
                else _align_time(d, s.data.index)
            )
            if self.scaled:
                d = d / s.scale
            columns.append(np.asarray(d, dtype=float).ravel()[self.mask])
        return np.column_stack(columns)

    def _finite_difference(self, updates: dict[str, float]) -> Array:
        base = self._vector(self._predict(updates))
        columns = []
        for name, value in updates.items():
            h = self.rel_step * abs(value) if value != 0 else self.rel_step
            up = self._vector(self._simulate(updates | {name: value + h}))
            down = self._vector(self._simulate(updates | {name: value - h}))
            column = (up - down) / (2 * h)
            # One-sided differences if one of the perturbed simulations failed
            column = np.where(np.isfinite(column), column, (up - base) / h)
            column = np.where(np.isfinite(column), column, (base - down) / h)
            columns.append(column)
        # Leave the model at the requested point, matching the cache
        _apply_updates(updates, self.settings)
        return np.column_stack(columns)


###############################################################################
# Parameter transforms
###############################################################################


class _Transformed:
    """Residual of log10-transformed parameters."""

    def __init__(self, inner: Residual, log: set[str]) -> None:
        self.inner = inner
        self.log = log

    def to_inner(self, updates: dict[str, float]) -> dict[str, float]:
        return {k: 10**v if k in self.log else v for k, v in updates.items()}

    def __call__(self, updates: dict[str, float]) -> float:
        return self.inner(self.to_inner(updates))


class _TransformedGradient(_Transformed):
    """`_Transformed` passing derivatives through the chain rule."""

    def _factors(self, updates: dict[str, float]) -> Array:
        inner = self.to_inner(updates)
        return np.array(
            [inner[k] * np.log(10) if k in self.log else 1.0 for k in updates]
        )

    def residuals(self, updates: dict[str, float]) -> Array:
        return cast(Any, self.inner).residuals(self.to_inner(updates))

    def jacobian(self, updates: dict[str, float]) -> Array:
        jac = cast(Any, self.inner).jacobian(self.to_inner(updates))
        return jac * self._factors(updates)[None, :]

    def gradient(self, updates: dict[str, float]) -> Array:
        grad = cast(Any, self.inner).gradient(self.to_inner(updates))
        return grad * self._factors(updates)


def _log_names(p0: dict[str, float], transform: Transform | None) -> set[str]:
    if transform is None:
        return set()
    if isinstance(transform, str):
        transform = dict.fromkeys(p0, transform)
    if unknown := set(transform) - set(p0):
        msg = f"Transforms for parameters not in p0: {sorted(unknown)}"
        raise ValueError(msg)
    if bad := sorted(k for k, v in transform.items() if v not in ("log", "linear")):
        msg = f"Transforms must be 'log' or 'linear', not for {bad}"
        raise ValueError(msg)
    log = {k for k, v in transform.items() if v == "log"}
    if bad := sorted(k for k in log if p0[k] <= 0):
        msg = f"Log transform needs positive initial values, not for {bad}"
        raise ValueError(msg)
    return log


def transformed(
    residual: Residual,
    p0: dict[str, float],
    bounds: Bounds,
    transform: Transform | None,
) -> tuple[
    Residual, dict[str, float], Bounds, Callable[[dict[str, float]], dict[str, float]]
]:
    """Wrap a residual to work on log10 parameters where requested.

    Returns the wrapped residual, the transformed initial values and bounds
    and a function mapping minimizer results back. Every transformed
    parameter gets explicit bounds, as minimizer defaults apply to the
    untransformed values.
    """
    log = _log_names(p0, transform)
    if not log:
        return residual, p0, bounds, lambda x: x

    cls = _TransformedGradient if hasattr(residual, "jacobian") else _Transformed
    wrapped = cls(residual, log)
    new_p0 = {k: float(np.log10(v)) if k in log else v for k, v in p0.items()}
    new_bounds = dict(bounds)
    for k in log:
        lo, hi = bounds.get(k, (None, None))
        # Non-positive or missing bounds can't be transformed, use finite
        # defaults containing p0 instead (global minimizers need them)
        if lo is None or lo <= 0:
            lo = min(_DEFAULT_BOUNDS[0], p0[k])
        if hi is None:
            hi = max(_DEFAULT_BOUNDS[1], p0[k])
        new_bounds[k] = (float(np.log10(lo)), float(np.log10(hi)))
    return cast("Residual", wrapped), new_p0, new_bounds, wrapped.to_inner
