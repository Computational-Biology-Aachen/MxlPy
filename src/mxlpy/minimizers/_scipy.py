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
    "GlobalScipyMinimizer",
    "LOGGER",
    "LeastSquaresMinimizer",
    "LocalScipyMinimizer",
    "ScipyResidualFn",
]

# scipy methods using a gradient, which then gets passed as `jac`
_GRADIENT_METHODS = frozenset(
    {
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
    }
)
# scipy methods using a Hessian, which then gets passed as `hess`
_HESSIAN_METHODS = frozenset(
    {"Newton-CG", "trust-constr", "dogleg", "trust-ncg", "trust-krylov", "trust-exact"}
)
# ... of which these can't do without one
_HESSIAN_REQUIRED = frozenset({"dogleg", "trust-ncg", "trust-krylov", "trust-exact"})
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
        if self.has_gradient and method in _GRADIENT_METHODS:
            return self.gradient
        return None

    def hess_for(self, method: str, p0: dict[str, float]) -> Any:
        """`hess` argument for `scipy.optimize.minimize` with `method`.

        The Gauss-Newton Hessian of the residual if it has one (`rmse` and
        `mean_squared` losses), else BFGS updates for `trust-constr`.
        """
        if method not in _HESSIAN_METHODS or not self.has_gradient:
            return None
        hessian = getattr(self.residual_fn, "hessian", None)
        if callable(hessian) and hessian(p0) is not None:
            return self.hessian
        if method in _HESSIAN_REQUIRED:
            msg = f"Only the rmse and mean_squared losses have a Gauss-Newton Hessian, use method='trust-constr' instead of {method!r} for others"
            raise ValueError(msg)
        return BFGS() if method == "trust-constr" else None


@dataclass(kw_only=True, slots=True)
class LocalScipyMinimizer(AbstractMinimizer):
    """Local multivariate minimization using scipy.optimize.

    By default (`method=None`), Nelder-Mead is used to fit a single parameter
    and L-BFGS-B otherwise, as the first L-BFGS-B step is of unit length,
    which for a single parameter frequently lands on its bounds.

    Gradient-based methods use the derivatives of the fit (see
    `mxlpy.runners`), and methods using a Hessian its Gauss-Newton
    approximation `Jᵀ (d²loss/dr²) J` for the `rmse` and `mean_squared`
    losses (BFGS updates for other losses with `trust-constr`). Unlike
    `LeastSquaresMinimizer`, which minimizes the sum of squares, these
    minimize the fit's loss function itself. `trust-exact` ignores bounds.

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data,
        ...                 minimizer=LocalScipyMinimizer(method="trust-constr"))

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
        pairs = [bounds.get(name, _DEFAULT_BOUNDS) for name in p0]
        x0 = np.array(list(p0.values()), dtype=float)
        kwargs: dict[str, Any] = {"bounds": pairs}
        if method == "trust-constr":
            # Start feasible, with a small barrier so bounds only matter close
            # to them. The default of 0.1 slows convergence down to the
            # gradient tolerance
            lb = np.array([-np.inf if lo is None else lo for lo, _ in pairs])
            ub = np.array([np.inf if hi is None else hi for _, hi in pairs])
            x0 = np.clip(x0, lb, ub)
            kwargs["options"] = {"initial_barrier_parameter": 1e-6}
        elif method == "trust-exact":
            LOGGER.debug("The trust-exact method ignores bounds")
            kwargs = {}

        res: OptimizeResult = minimize(
            res_fn,
            x0=x0,
            jac=res_fn.jac_for(method),
            hess=res_fn.hess_for(method, p0),
            method=method,
            tol=self.tol,
            **kwargs,
        )
        if res_fn.is_penalty(res.fun):
            LOGGER.warning("Minimisation failed, no finite residual found")
            return Result(FitFailure(extra_info=["No finite residual found"]))
        # trust-exact stops with status 2 if no step improves the loss any
        # more, e.g. at a perfect fit, where the loss is at machine precision
        if res.success or (method == "trust-exact" and res.status == 2):  # noqa: PLR2004
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
    prediction minus the data), using its Jacobian (see `mxlpy.runners` for
    how it is computed). This needs one of the residual functions of
    `mxlpy.fit.residuals`. The reported loss is the value of the fit's loss
    function, so results stay comparable to other minimizers.

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
    backend
        `scipy` or `native`, which runs `lm` with the model and its forward
        sensitivities entirely in compiled code (needs `mxlpy[native]`'s
        llvm backend, falls back to scipy otherwise). It integrates with the
        method and tolerances of the runner's `Native` integrator, or
        Native's defaults

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data,
        ...                 minimizer=LeastSquaresMinimizer())
        >>> fit.time_course(model, p0=p0, data=data, transform="log",
        ...                 minimizer=LeastSquaresMinimizer(method="lm", backend="native"))

    """

    method: Literal["trf", "dogbox", "lm"] = "trf"
    tol: float = 1e-8
    max_nfev: int | None = None
    backend: Literal["scipy", "native"] = "scipy"

    def __post_init__(self) -> None:
        """Check that the backend supports the method."""
        if self.backend == "native" and self.method != "lm":
            msg = "The native backend only implements method='lm'"
            raise ValueError(msg)

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Minimize the sum of squared residuals."""
        if not callable(getattr(residual_fn, "residuals", None)):
            msg = "LeastSquaresMinimizer needs residual vectors, fit with one of the residual functions of mxlpy.fit.residuals"
            raise TypeError(msg)
        if self.backend == "native":
            from mxlpy.fit._native import least_squares as native  # noqa: PLC0415 (imports this module)

            return native(
                residual_fn,
                p0,
                bounds,
                max_nfev=100 * (len(p0) + 1)
                if self.max_nfev is None
                else self.max_nfev,
            )
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
