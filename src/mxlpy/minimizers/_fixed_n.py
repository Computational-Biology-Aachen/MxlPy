"""Fixed-N minimizer: run exactly N optimizer iterations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.optimize import minimize

from mxlpy.types import Array, Result

from .abstract import AbstractMinimizer, Bounds, OptimisationState, Residual

__all__ = ["FixedNMinimizer"]


def _pack_updates(
    par_values: Array,
    par_names: list[str],
) -> dict[str, float]:
    return dict(zip(par_names, par_values, strict=True))


@dataclass(kw_only=True, slots=True)
class FixedNMinimizer(AbstractMinimizer):
    """Minimizer that performs exactly ``n_steps`` optimizer iterations.

    Unlike :class:`LocalScipyMinimizer`, this minimizer never fails: it always
    returns the best parameter state reached after ``n_steps`` iterations,
    regardless of whether convergence was achieved.  This makes the runtime
    predictable and bounded, which is desirable for Monte Carlo robustness
    checks and Pareto-optimality tests.

    Parameters
    ----------
    n_steps
        Number of optimizer iterations to perform.
    method
        Gradient-based scipy method to use.  All listed methods support the
        ``maxiter`` option and respect parameter bounds.
    tol
        Tolerance passed to ``scipy.optimize.minimize``.  ``None`` uses the
        scipy default.  The iteration cap takes priority: even if ``tol`` is
        loose, the minimizer stops after ``n_steps`` iterations.

    Examples
    --------
    >>> minimizer = FixedNMinimizer(n_steps=10)
    >>> result = minimizer(lambda p: sum(v**2 for v in p.values()), {"x": 1.0}, {})
    >>> result.unwrap_or_err().parameters["x"] < 1.0
    True

    """

    n_steps: int
    method: Literal["L-BFGS-B", "BFGS", "CG", "TNC", "SLSQP"] = "L-BFGS-B"
    tol: float | None = None

    def __call__(
        self,
        residual_fn: Residual,
        p0: dict[str, float],
        bounds: Bounds,
    ) -> Result[OptimisationState]:
        """Run exactly ``n_steps`` optimizer iterations and return the result.

        Parameters
        ----------
        residual_fn
            Callable that maps a parameter dict to a scalar residual.
        p0
            Initial parameter values.
        bounds
            Per-parameter bounds.  Keys absent from ``bounds`` default to
            ``(1e-6, 1e6)``.

        Returns
        -------
        Result[OptimisationState]
            Always succeeds - returns the parameter state after ``n_steps``
            iterations together with the residual at that point.

        """
        par_names = list(p0.keys())

        res = minimize(
            lambda par_values: residual_fn(_pack_updates(par_values, par_names)),
            x0=list(p0.values()),
            bounds=[bounds.get(name, (1e-6, 1e6)) for name in p0],
            method=self.method,
            tol=self.tol,
            options={"maxiter": self.n_steps},
        )
        return Result(
            OptimisationState(
                parameters=dict(zip(p0, res.x, strict=True)),
                residual=float(res.fun),
            )
        )
