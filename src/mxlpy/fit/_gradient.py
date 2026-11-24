# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Residuals providing residual vectors and derivatives for gradient-based minimizers.

Minimizers detect the optional `residuals`, `jacobian`, `gradient` and
`hessian` methods, so plain scalar residuals keep working unchanged. The
derivatives come from the runner (see `mxlpy.runners.SensitivityRunner`)
or, for runners without them, from finite differences of its results.
Loss gradients come from the runner's vector-Jacobian products (adjoints)
instead of the residual Jacobian if the runner chooses adjoints.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from mxlpy.adjoint_sensitivities import uses_adjoint
from mxlpy.fit import losses
from mxlpy.fit.losses import GaussianNll, _sigma_array
from mxlpy.fit.residuals import (
    _apply_updates,
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.simulation import _align_time

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mxlpy.fit.abstract import FitResidual, _Settings
    from mxlpy.minimizers.abstract import Bounds, Residual
    from mxlpy.types import Array

__all__ = ["GradientResidual", "Transform", "transformed"]

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()

type Transform = Literal["log", "linear"] | dict[str, Literal["log", "linear"]]

_DEFAULT_BOUNDS = (1e-6, 1e6)


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOGGER.warning(msg)


def _loss_derivative(
    loss_fn: Callable[..., float], n: int
) -> Callable[[Array], Array] | None:
    """Analytic d loss / d residual for the built-in losses, else None.

    The residuals of a `GaussianNll` are already weighted by `1 / sigma`.
    """
    if isinstance(loss_fn, GaussianNll):
        return lambda r: r
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


def _loss_hessian(
    loss_fn: Callable[..., float], n: int
) -> Callable[[Array, Array], Array] | None:
    """Gauss-Newton Hessian for the least-squares losses, else None.

    Takes the residual vector and its Jacobian. For `mean_squared` this is
    2 JᵀJ / n. For `rmse` = sqrt(mean_squared) it is JᵀJ / (n rmse), leaving
    out the negative rank-one term of the square root: that term makes the
    Hessian singular along the Gauss-Newton step whenever the residual lies
    in the range of the Jacobian (e.g. close to a fit of noise-free data),
    while without it the Newton step is exactly the Gauss-Newton step.
    """
    if isinstance(loss_fn, GaussianNll):
        return lambda _, jac: jac.T @ jac
    if loss_fn is losses.mean_squared:
        return lambda _, jac: 2 * jac.T @ jac / n
    if loss_fn is losses.rmse:

        def h_rmse(r: Array, jac: Array) -> Array:
            value = max(float(np.sqrt(np.mean(np.square(r)))), np.finfo(float).tiny)
            return jac.T @ jac / (n * value)

        return h_rmse
    return None


_KINDS = {
    steady_state_residual: "steady_state",
    time_course_residual: "time_course",
    protocol_time_course_residual: "protocol_time_course",
}


class GradientResidual:
    """Residual of a single fit, additionally providing derivatives.

    The residual vector is the (standard-scaled, if enabled) prediction minus
    the data, flattened, without missing data. For a `GaussianNll` loss it is
    weighted by `1 / sigma` instead, so its sum of squares is twice the loss. Its Jacobian comes from the
    runner's sensitivities if it provides them, else from central finite
    differences of whole simulations.

    Parameters
    ----------
    residual_fn
        One of the default residual functions of `mxlpy.fit.residuals`
    settings
        Settings of the fit

    """

    def __init__(self, residual_fn: FitResidual, settings: _Settings[Any]) -> None:
        if (kind := _KINDS.get(cast(Any, residual_fn))) is None:
            msg = (
                "Derivatives need one of the residual functions of mxlpy.fit.residuals"
            )
            raise ValueError(msg)
        self.kind = kind
        self.fn = partial(residual_fn, settings=settings)
        self.settings = settings
        self.displacement = float(getattr(settings.runner, "displacement", 1e-3))

        data = settings.data
        self.names: list[str] = list(
            data.index if isinstance(data, pd.Series) else data.columns
        )
        self.scaled = settings.scaled
        # Missing data and constant columns (zero std, so NaN when scaled) are
        # left out, as the scalar losses skip them as well
        scaled = np.asarray(self._scale(data), dtype=float).ravel()
        self.mask = np.isfinite(scaled)
        self.data_vector = scaled[self.mask]
        loss_fn = settings.loss_fn
        self.weights: Array | None = (
            1 / _sigma_array(loss_fn.sigma, data).ravel()[self.mask]
            if isinstance(loss_fn, GaussianNll)
            else None
        )

        self.forward = (reason := self._forward_problem()) is None
        if reason is not None:
            self._fall_back(reason)

        self._key: tuple[tuple[str, float], ...] | None = None
        self._prediction: pd.DataFrame | pd.Series | None = None
        self._jac_key: tuple[tuple[str, float], ...] | None = None
        self._jac: Array | None = None
        self._adjoint_failed = False

    @staticmethod
    def supports(residual_fn: FitResidual) -> bool:
        """Whether `residual_fn` is one of the residuals derivatives exist for."""
        return cast(Any, residual_fn) in _KINDS

    def _forward_problem(self) -> str | None:
        """Why the runner's sensitivities can't be used, if they can't."""
        s = self.settings
        method = (
            "steady_state_sensitivities"
            if self.kind == "steady_state"
            else "time_course_sensitivities"
        )
        if not callable(getattr(s.runner, method, None)):
            return "the runner doesn't provide sensitivities"
        if getattr(s.runner, "derivatives", None) == "finite_difference":
            return "the runner uses finite differences"
        if self.kind == "protocol_time_course":
            return "protocol fits don't support forward sensitivities yet"
        if s.postprocess_fn is not None:
            return "a postprocess_fn can't be differentiated"
        return None

    def _fall_back(self, reason: str) -> None:
        """Use finite differences of the runner's results from now on.

        Only warns if the runner was explicitly asked for forward
        sensitivities, as finite differences are the documented fallback.
        """
        self.forward = False
        msg = f"Using finite-difference derivatives for fitting, as {reason}"
        if getattr(self.settings.runner, "derivatives", None) == "forward":
            _warn_once(msg)
        else:
            _LOGGER.debug(msg)

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
        prediction = val.get_combined(include_terms=True)
        if s.postprocess_fn is not None:
            prediction = s.postprocess_fn(prediction)
        if self.kind == "steady_state":
            return prediction.iloc[-1].loc[self.names]
        return _align_time(prediction, s.data.index).loc[:, self.names]

    def _vector(self, prediction: pd.DataFrame | pd.Series | None) -> Array:
        if prediction is None:
            return np.full(int(self.mask.sum()), np.nan)
        r = (
            np.asarray(self._scale(prediction), dtype=float).ravel()[self.mask]
            - self.data_vector
        )
        return r if self.weights is None else r * self.weights

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
        """Scalar loss of the wrapped residual function.

        Calls it directly, so fits not using derivatives behave exactly as
        without this wrapper.
        """
        return self.fn(updates)

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
        """Derivative of the scalar loss w.r.t. the parameters.

        From the runner's vector-Jacobian product if it uses adjoints, else
        from the residual Jacobian.
        """
        r = self._vector(self._predict(updates))
        valid = np.isfinite(r)
        if self._prediction is None or not valid.any():
            return np.full(len(updates), np.nan)
        d_loss = self._d_loss(r[valid], valid)
        # A Jacobian computed for this point anyway (e.g. for the Hessian)
        # gives the gradient for free
        cached = tuple(updates.items()) == self._jac_key and self._jac is not None
        if (
            not cached
            and self._uses_adjoint(len(updates))
            and (grad := self._adjoint(updates, d_loss, valid)) is not None
        ):
            return grad
        return d_loss @ self.jacobian(updates)[valid]

    def squares_factor(self) -> float | None:
        """`c` with loss = `c Σr² / 2` of the residual vector, None if it isn't one.

        Least squares needs it to weight priors and penalties like the loss.
        """
        if isinstance(self.settings.loss_fn, GaussianNll):
            return 1.0
        if self.settings.loss_fn is losses.mean_squared:
            return 2 / max(int(self.mask.sum()), 1)
        return None

    def hessian(self, updates: dict[str, float]) -> Array | None:
        """Gauss-Newton approximation of the loss Hessian.

        Positive semi-definite and shares the sensitivities with `gradient`.
        None for losses other than `mean_squared` and `rmse`.
        """
        # Checked before simulating, so asking whether there is one is cheap
        if _loss_hessian(self.settings.loss_fn, 1) is None:
            return None
        r = self._vector(self._predict(updates))
        valid = np.isfinite(r)
        fn = cast("Callable", _loss_hessian(self.settings.loss_fn, int(valid.sum())))
        if self._prediction is None or not valid.any():
            return np.full((len(updates), len(updates)), np.nan)
        return fn(r[valid], self.jacobian(updates)[valid])

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
            h = self.displacement * max(1.0, abs(r[i]))
            up, down = r.copy(), r.copy()
            up[i] += h
            down[i] -= h
            out[i] = (loss_at(up) - loss_at(down)) / (2 * h)
        return out

    def _uses_adjoint(self, n_wrt: int) -> bool:
        """Whether loss gradients come from the runner's adjoints."""
        s = self.settings
        if (
            self._adjoint_failed
            or not callable(getattr(s.runner, f"{self.kind}_vjp", None))
            or s.postprocess_fn is not None
        ):
            return False
        derivatives = getattr(s.runner, "derivatives", None)
        if derivatives == "adjoint":
            return True
        # Steady-state sensitivities cost one linear solve, shared with the
        # Gauss-Newton Hessian, so only time courses switch
        return (
            derivatives == "auto"
            and self.kind != "steady_state"
            and uses_adjoint(
                cast(Any, s.model),
                n_wrt,
                getattr(s.runner, "integrator", None),
                protocol=self.kind == "protocol_time_course",
            )
        )

    def _cotangents(
        self, d_loss: Array, valid: NDArray[np.bool_]
    ) -> pd.DataFrame | pd.Series:
        """D loss / d prediction, from d loss / d (valid) residuals.

        Replicates share time points, so their entries are summed.
        """
        s = self.settings
        full = np.zeros(self.mask.shape)
        positions = np.flatnonzero(self.mask)[valid]
        full[positions] = (
            d_loss if self.weights is None else d_loss * self.weights[valid]
        )
        data = s.data
        if isinstance(data, pd.Series):
            out: pd.DataFrame | pd.Series = pd.Series(full, index=self.names)
        else:
            frame = pd.DataFrame(
                full.reshape(data.shape),
                index=np.asarray(data.index, dtype=float),
                columns=self.names,
            )
            out = cast(pd.DataFrame, frame.groupby(level=0).sum())
        if self.scaled:
            out = cast(pd.DataFrame | pd.Series, out / s.scale)
        return out.fillna(0.0)

    def _adjoint(
        self, updates: dict[str, float], d_loss: Array, valid: NDArray[np.bool_]
    ) -> Array | None:
        """Loss gradient from the runner's vector-Jacobian product."""
        s = self.settings
        _apply_updates(updates, s)
        runner = cast(Any, s.runner)
        wrt = list(updates)
        cotangents = self._cotangents(d_loss, valid)
        try:
            if self.kind == "steady_state":
                res = runner.steady_state_vjp(s.model, cotangents, wrt)
            elif self.kind == "time_course":
                res = runner.time_course_vjp(s.model, cotangents, wrt)
            else:
                res = runner.protocol_time_course_vjp(
                    s.model, s.protocol, cotangents, wrt
                )
        except Exception as e:  # noqa: BLE001
            # Models the adjoint codegen can't handle with derivatives="adjoint"
            self._adjoint_failed = True
            msg = f"Using sensitivities for loss gradients, as adjoints failed: {e}"
            if getattr(s.runner, "derivatives", None) == "adjoint":
                _warn_once(msg)
            else:
                _LOGGER.debug(msg)
            return None
        if isinstance(val := res.value, Exception):
            return None
        return val.gradient.loc[wrt].to_numpy(dtype=float)

    def _forward(self, updates: dict[str, float]) -> Array | None:
        s = self.settings
        _apply_updates(updates, s)
        runner = cast(Any, s.runner)
        wrt = list(updates)
        try:
            if self.kind == "steady_state":
                res = runner.steady_state_sensitivities(s.model, wrt)
            else:
                # Replicates share time points, which the integrators reject
                res = runner.time_course_sensitivities(
                    s.model, np.unique(np.asarray(s.data.index, dtype=float)), wrt
                )
        except Exception as e:  # noqa: BLE001
            # Models the sensitivity codegen can't handle with
            # derivatives="forward" (events, Native, unparsable functions, ...)
            self._fall_back(str(e))
            return None
        if isinstance(sens := res.value, Exception):
            return None
        # The prediction itself always comes from the runner, like the loss,
        # as the augmented solve differs within the integration tolerance
        columns = []
        for name in wrt:
            d = sens.derivatives[name].loc[:, self.names]
            d = (
                d.iloc[-1]
                if self.kind == "steady_state"
                else _align_time(d, s.data.index)
            )
            if self.scaled:
                d = d / s.scale
            columns.append(np.asarray(d, dtype=float).ravel()[self.mask])
        jac = np.column_stack(columns)
        return jac if self.weights is None else jac * self.weights[:, None]

    def _finite_difference(self, updates: dict[str, float]) -> Array:
        base = self._vector(self._predict(updates))
        columns = []
        for name, value in updates.items():
            h = self.displacement * abs(value) if value != 0 else self.displacement
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

    def hessian(self, updates: dict[str, float]) -> Array | None:
        # The Gauss-Newton Hessian of the transformed residual Jacobian J·f,
        # which stays positive semi-definite. The exact chain rule would add
        # the gradient times d²θ/du² along the diagonal, which is indefinite
        # away from the optimum and vanishes at it.
        h = cast(Any, self.inner).hessian(self.to_inner(updates))
        if h is None:
            return None
        f = self._factors(updates)
        return h * np.outer(f, f)


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
        msg = (
            f"Log transform needs positive initial values, not for {bad}; "
            "pass transform per parameter, e.g. {name: 'linear'}, for these"
        )
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
