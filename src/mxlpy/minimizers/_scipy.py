from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


@dataclass(kw_only=True, slots=True)
class LocalScipyMinimizer(AbstractMinimizer):
    """Local multivariate minimization using scipy.optimize.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization

    """

    tol: float = 1e-6
    method: Literal[
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
    ] = "L-BFGS-B"

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Call minimzer."""
        par_names = list(p0.keys())

        res: OptimizeResult = minimize(
            lambda par_values: residual_fn(_pack_updates(par_values, par_names)),
            x0=list(p0.values()),
            bounds=[bounds.get(name, (1e-6, 1e6)) for name in p0],
            method=self.method,
            tol=self.tol,
        )
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
        par_names = list(p0.keys())
        res_fn: ScipyResidualFn = lambda par_values: residual_fn(  # noqa: E731
            _pack_updates(par_values, par_names)
        )

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
            msg = f"Unknown method {self.method}"
            raise NotImplementedError(msg)
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
