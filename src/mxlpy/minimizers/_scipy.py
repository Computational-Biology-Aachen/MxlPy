# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    direct,
    dual_annealing,
    minimize,
    shgo,
)

from mxlpy.types import Array, FitFailure, Result

from .abstract import AbstractMinimizer, Bounds, OptimisationState, Residual

__all__ = ["GlobalScipyMinimizer", "LOGGER", "LocalScipyMinimizer", "ScipyResidualFn"]

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
            bounds=[bounds.get(name, (1e-6, 1e6)) for name in p0],
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

        if self.method == "basinhopping":
            res = basinhopping(
                res_fn,
                x0=list(p0.values()),
            )
        elif self.method == "differential_evolution":
            res = differential_evolution(res_fn, bounds)
        elif self.method == "shgo":
            res = shgo(res_fn, bounds)
        elif self.method == "dual_annealing":
            res = dual_annealing(res_fn, bounds)
        elif self.method == "direct":
            res = direct(res_fn, bounds)
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
