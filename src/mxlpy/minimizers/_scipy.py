# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.optimize import (
    BFGS,
    basinhopping,
    differential_evolution,
    direct,
    dual_annealing,
    least_squares,
    minimize,
    shgo,
)

from mxlpy.types import Array, FitFailure, Result

from .abstract import AbstractMinimizer, Bounds, OptimisationState, Residual

__all__ = [
    "GaussNewtonMinimizer",
    "GlobalScipyMinimizer",
    "LOGGER",
    "LeastSquaresMinimizer",
    "LocalScipyMinimizer",
    "ScipyResidualFn",
]

# scipy methods using a gradient, which then gets passed as `jac`
GRADIENT_METHODS = frozenset(
    {"CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP", "trust-constr"}
)
_DEFAULT_BOUNDS = (1e-6, 1e6)

if TYPE_CHECKING:
    from scipy.optimize._optimize import OptimizeResult


LOGGER = logging.getLogger(__name__)


type ScipyResidualFn = Callable[[Array], float]


def _pack_updates(
    par_values: Array,
    par_names: list[str],
) -> dict[str, float]:
    return dict(
        zip(
            par_names,
            par_values,
            strict=True,
        )
    )


class _FiniteResidual:
    """Scipy residual function replacing non-finite residuals by a finite penalty.

    Residual functions return `inf` if e.g. a simulation fails. Gradient-based
    methods can't handle that: finite differences turn it into `nan` and the
    line search aborts, while scipy still reports success at the initial
    guess. A finite penalty instead lets the line search backtrack.

    A fixed large penalty would destroy the precision of the relative
    convergence criteria, so it is scaled from the largest finite residual
    seen so far. It only ever grows, so a failing region never looks more
    attractive than any evaluated point, even for global methods sampling
    the whole parameter space.
    """

    fallback_penalty: float = 1e10
    penalty_factor: float = 1e3

    def __init__(self, residual_fn: Residual, par_names: list[str]) -> None:
        self.residual_fn = residual_fn
        self.par_names = par_names
        self.penalty: float | None = None
        self.returned_penalties: set[float] = set()

    def __call__(self, par_values: Array) -> float:
        value = self.residual_fn(_pack_updates(par_values, self.par_names))
        if np.isfinite(value):
            self.penalty = max(
                self.penalty or 0.0, self.penalty_factor * max(1.0, abs(value))
            )
            return value
        penalty = self.fallback_penalty if self.penalty is None else self.penalty
        self.returned_penalties.add(penalty)
        return penalty

    def is_penalty(self, value: float) -> bool:
        """Check whether a returned value is a penalty instead of a residual."""
        return not np.isfinite(value) or value in self.returned_penalties

    @property
    def has_gradient(self) -> bool:
        """Whether the residual function provides analytic gradients."""
        return callable(getattr(self.residual_fn, "gradient", None))

    def gradient(self, par_values: Array) -> Array:
        """Gradient of the residual, zero where it isn't finite.

        Non-finite residuals are replaced by a penalty, which is flat, so the
        line search backtracks.
        """
        grad = np.asarray(
            self.residual_fn.gradient(_pack_updates(par_values, self.par_names)),  # type: ignore[attr-defined]
            dtype=float,
        )
        if not np.isfinite(grad).all():
            return np.zeros_like(grad)
        return grad

    def hessian(self, par_values: Array) -> Array:
        """Hessian of the residual, zero where it isn't finite."""
        hess = np.asarray(
            self.residual_fn.hessian(_pack_updates(par_values, self.par_names)),  # type: ignore[attr-defined]
            dtype=float,
        )
        if not np.isfinite(hess).all():
            return np.zeros_like(hess)
        return hess

    def jac_for(self, method: str) -> Callable[[Array], Array] | None:
        """`jac` argument for `scipy.optimize.minimize` with `method`."""
        if self.has_gradient and method in GRADIENT_METHODS:
            return self.gradient
        return None


@dataclass(kw_only=True, slots=True)
class LocalScipyMinimizer(AbstractMinimizer):
    """Local multivariate minimization using scipy.optimize.

    By default (`method=None`), Nelder-Mead is used to fit a single parameter
    and L-BFGS-B otherwise, as the first L-BFGS-B step is of unit length,
    which for a single parameter frequently lands on its bounds.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization

    """

    tol: float = 1e-6
    method: (
        Literal[
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "COBYQA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        | None
    ) = None

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Call minimzer."""
        if (method := self.method) is None:
            method = "Nelder-Mead" if len(p0) == 1 else "L-BFGS-B"
        res_fn = _FiniteResidual(residual_fn, list(p0))

        res: OptimizeResult = minimize(
            res_fn,
            x0=list(p0.values()),
            jac=res_fn.jac_for(method),
            bounds=[bounds.get(name, _DEFAULT_BOUNDS) for name in p0],
            method=method,
            tol=self.tol,
        )
        if res_fn.is_penalty(res.fun):
            LOGGER.warning("Minimisation failed, no finite residual found")
            return Result(FitFailure(extra_info=["No finite residual found"]))
        if res.success:
            return Result(
                OptimisationState(
                    parameters=dict(
                        zip(
                            p0,
                            res.x,
                            strict=True,
                        ),
                    ),
                    residual=res.fun,
                )
            )
        LOGGER.warning("Minimisation failed due to %s", res.message)
        return Result(FitFailure(extra_info=[res.message]))


@dataclass(kw_only=True, slots=True)
class GlobalScipyMinimizer(AbstractMinimizer):
    """Global iate minimization using scipy.optimize.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization

    """

    tol: float = 1e-6
    method: Literal[
        "basinhopping",
        "differential_evolution",
        "shgo",
        "dual_annealing",
        "direct",
    ] = "basinhopping"

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize residual fn."""
        res: OptimizeResult
        res_fn = _FiniteResidual(residual_fn, list(p0))
        # The global methods need finite (low, high) pairs in parameter order
        bounds_seq = [bounds.get(name, _DEFAULT_BOUNDS) for name in p0]

        if self.method == "basinhopping":
            res = basinhopping(
                res_fn,
                x0=list(p0.values()),
            )
        elif self.method == "differential_evolution":
            res = differential_evolution(res_fn, bounds_seq)
        elif self.method == "shgo":
            res = shgo(res_fn, bounds_seq)
        elif self.method == "dual_annealing":
            res = dual_annealing(res_fn, bounds_seq)
        elif self.method == "direct":
            res = direct(res_fn, bounds_seq)
        else:
            msg = f"Unknown minimizer method {self.method!r} - valid methods: 'basinhopping', 'differential_evolution', 'shgo', 'dual_annealing', 'direct'"
            raise NotImplementedError(msg)
        if res_fn.is_penalty(res.fun):
            LOGGER.warning("Minimisation failed, no finite residual found")
            return Result(FitFailure(extra_info=["No finite residual found"]))
        if res.success:
            return Result(
                OptimisationState(
                    parameters=dict(
                        zip(
                            p0,
                            res.x,
                            strict=True,
                        ),
                    ),
                    residual=res.fun,
                )
            )

        LOGGER.warning("Minimisation failed.")
        return Result(FitFailure(extra_info=[res.message]))


@dataclass(kw_only=True, slots=True)
class LeastSquaresMinimizer(AbstractMinimizer):
    """Nonlinear least squares using `scipy.optimize.least_squares`.

    Minimizes the sum of squares of the residual vector (the standard-scaled
    prediction minus the data), using its Jacobian. This needs a residual
    providing them, i.e. a fit with `sensitivities` set. The reported loss
    is the value of the fit's loss function, so results stay comparable to
    other minimizers.

    Parameters
    ----------
    method
        `trf` (trust region reflective, supports bounds), `dogbox` or `lm`
        (Levenberg-Marquardt, MINPACK). `lm` ignores bounds, so parameters
        can become negative unless fitted with `transform="log"`
    tol
        Tolerance for the change of the cost, the parameters and the gradient
    max_nfev
        Maximal number of residual evaluations, None uses the scipy default

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data,
        ...                 minimizer=LeastSquaresMinimizer(),
        ...                 sensitivities="forward")

    """

    method: Literal["trf", "dogbox", "lm"] = "trf"
    tol: float = 1e-8
    max_nfev: int | None = None

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize the sum of squared residuals."""
        if not callable(getattr(residual_fn, "residuals", None)):
            msg = "LeastSquaresMinimizer needs residual vectors, fit with sensitivities='forward' or 'finite_difference'"
            raise TypeError(msg)
        residuals = cast(
            Callable[[dict[str, float]], Array],
            residual_fn.residuals,  # type: ignore[attr-defined]
        )
        jacobian = getattr(residual_fn, "jacobian", None)
        names = list(p0)

        if self.method == "lm":
            LOGGER.debug("The lm method ignores bounds")
            lb, ub = -np.inf, np.inf
        else:
            pairs = [bounds.get(name, _DEFAULT_BOUNDS) for name in names]
            lb = np.array([-np.inf if lo is None else lo for lo, _ in pairs])
            ub = np.array([np.inf if hi is None else hi for _, hi in pairs])

        penalty = {"value": 1e5}

        def fun(x: Array) -> Array:
            r = np.asarray(residuals(_pack_updates(x, names)), dtype=float)
            if np.isfinite(r).all():
                penalty["value"] = max(
                    penalty["value"], 1e3 * max(1.0, float(np.abs(r).max()))
                )
                return r
            return np.full_like(r, penalty["value"])

        def jac(x: Array) -> Array:
            updates = _pack_updates(x, names)
            j = np.asarray(cast(Callable, jacobian)(updates), dtype=float)
            if not np.isfinite(j).all() or not np.isfinite(residuals(updates)).all():
                return np.zeros_like(j)
            return j

        # Like L-BFGS-B, start from the closest feasible point
        x0 = np.clip(np.array(list(p0.values()), dtype=float), lb, ub)
        try:
            res = least_squares(
                fun,
                x0=x0,
                jac=cast(Any, jac if callable(jacobian) else "2-point"),
                bounds=(lb, ub),
                method=self.method,
                ftol=self.tol,
                xtol=self.tol,
                gtol=self.tol,
                max_nfev=self.max_nfev,
            )
        except ValueError as e:
            return Result(FitFailure(extra_info=[str(e)]))

        parameters = dict(zip(names, res.x, strict=True))
        loss = residual_fn(parameters)
        if not res.success or not np.isfinite(loss):
            LOGGER.warning("Minimisation failed due to %s", res.message)
            return Result(FitFailure(extra_info=[res.message]))
        return Result(OptimisationState(parameters=parameters, residual=float(loss)))


@dataclass(kw_only=True, slots=True)
class GaussNewtonMinimizer(AbstractMinimizer):
    """Trust-region minimization of the loss with a Gauss-Newton Hessian.

    Uses the exact gradient and the Gauss-Newton approximation of the
    Hessian (Jᵀ (d²loss/dr²) J, positive semi-definite), both from the same
    sensitivity solve. Unlike `LeastSquaresMinimizer`, which minimizes the
    sum of squares, it minimizes the fit's loss function itself. This needs
    a fit with `sensitivities` set and the `rmse` or `mean_squared` loss;
    other differentiable losses fall back to BFGS updates of the Hessian
    (`trust-constr` only).

    Parameters
    ----------
    method
        `trust-constr` (supports bounds) or `trust-exact` (ignores bounds)
    tol
        Tolerance for termination
    max_iter
        Maximal number of iterations

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data,
        ...                 minimizer=GaussNewtonMinimizer(),
        ...                 sensitivities="forward")

    """

    method: Literal["trust-constr", "trust-exact"] = "trust-constr"
    tol: float = 1e-8
    max_iter: int = 1000

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize the loss."""
        if not callable(getattr(residual_fn, "hessian", None)):
            msg = "GaussNewtonMinimizer needs derivatives, fit with sensitivities='forward' or 'finite_difference'"
            raise TypeError(msg)
        res_fn = _FiniteResidual(residual_fn, list(p0))
        x0 = np.array(list(p0.values()), dtype=float)

        hess: Any = res_fn.hessian
        if residual_fn.hessian(p0) is None:  # type: ignore[attr-defined]
            if self.method != "trust-constr":
                msg = "Only the rmse and mean_squared losses have a Gauss-Newton Hessian, use method='trust-constr' for others"
                raise ValueError(msg)
            LOGGER.debug("No Gauss-Newton Hessian for this loss, using BFGS updates")
            hess = BFGS()

        kwargs: dict[str, Any] = {}
        if self.method == "trust-constr":
            pairs = [bounds.get(name, _DEFAULT_BOUNDS) for name in p0]
            lb = np.array([-np.inf if lo is None else lo for lo, _ in pairs])
            ub = np.array([np.inf if hi is None else hi for _, hi in pairs])
            x0 = np.clip(x0, lb, ub)
            kwargs["bounds"] = list(zip(lb, ub, strict=True))
            # A small barrier, so bounds only matter close to them. The
            # default of 0.1 slows convergence down to the gradient tolerance
            kwargs["options"] = {
                "maxiter": self.max_iter,
                "xtol": self.tol,
                "gtol": self.tol,
                "initial_barrier_parameter": 1e-6,
                "barrier_tol": self.tol,
            }
        else:
            LOGGER.debug("The trust-exact method ignores bounds")
            kwargs["options"] = {"maxiter": self.max_iter, "gtol": self.tol}

        res: OptimizeResult = minimize(
            res_fn,
            x0=x0,
            jac=res_fn.gradient,
            hess=hess,
            method=self.method,
            **kwargs,
        )
        if res_fn.is_penalty(res.fun):
            LOGGER.warning("Minimisation failed, no finite residual found")
            return Result(FitFailure(extra_info=["No finite residual found"]))
        # trust-exact stops with status 2 if no step improves the loss any
        # more, e.g. at a perfect fit, where the loss is at machine precision
        if not (res.success or (self.method == "trust-exact" and res.status == 2)):  # noqa: PLR2004
            LOGGER.warning("Minimisation failed due to %s", res.message)
            return Result(FitFailure(extra_info=[res.message]))
        return Result(
            OptimisationState(
                parameters=dict(zip(p0, res.x, strict=True)),
                residual=float(res.fun),
            )
        )
