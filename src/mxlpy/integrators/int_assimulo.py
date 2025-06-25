"""Assimulo integrator for solving ODEs."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

with contextlib.redirect_stderr(open(os.devnull, "w")):  # noqa: PTH123
    from assimulo.problem import Explicit_Problem  # type: ignore
    from assimulo.solvers import CVode  # type: ignore
    from assimulo.solvers.sundials import CVodeError  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Array, ArrayLike, Rhs


__all__ = [
    "Assimulo",
]


@dataclass
class Assimulo:
    """Assimulo integrator for solving ODEs.

    Attributes:
        rhs: Right-hand side function of the ODE.
        y0: Initial conditions.
        atol: Absolute tolerance for the solver.
        rtol: Relative tolerance for the solver.
        maxnef: Maximum number of error failures.
        maxncf: Maximum number of convergence failures.
        verbosity: Verbosity level of the solver.

    Methods:
        integrate: Integrate the ODE system.

    """

    rhs: Rhs
    y0: tuple[float, ...]
    jacobian: Callable | None = None
    atol: float = 1e-8
    rtol: float = 1e-8
    maxnef: int = 4  # max error failures
    maxncf: int = 1  # max convergence failures
    verbosity: Literal[50, 40, 30, 20, 10] = 50

    def __post_init__(self) -> None:
        """Post-initialization method for setting up the CVode integrator with the provided parameters.

        This method initializes the CVode integrator with an explicit problem defined by the
        right-hand side function (`self.rhs`) and the initial conditions (`self.y0`). It also
        sets various integrator options such as absolute tolerance (`self.atol`), relative
        tolerance (`self.rtol`), maximum number of error test failures (`self.maxnef`), maximum
        number of convergence failures (`self.maxncf`), and verbosity level (`self.verbosity`).

        """
        problem = Explicit_Problem(self.rhs, self.y0)
        if self.jacobian is not None:
            problem.jac = self.jacobian

        self.integrator = CVode(problem)
        self.integrator.atol = self.atol
        self.integrator.rtol = self.rtol
        self.integrator.maxnef = self.maxnef
        self.integrator.maxncf = self.maxncf
        self.integrator.verbosity = self.verbosity

    def reset(self) -> None:
        """Reset the integrator."""
        self.integrator.reset()

    def integrate(
        self,
        *,
        t_end: float,
        steps: int | None = None,
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the ODE system.

        Args:
            t_end: Terminal time point for the integration.
            steps: Number of steps for the integration.
            time_points: Time points for the integration.

        Returns:
            np.ndarray: Array of integrated values.

        """
        if steps is None:
            steps = 0
        try:
            t, y = self.integrator.simulate(t_end, steps)
            return (
                np.atleast_1d(np.array(t, dtype=float)),
                np.atleast_2d(np.array(y, dtype=float)),
            )
        except CVodeError:
            return None, None

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the ODE system over a time course.

        Args:
            time_points: Time points for the integration.

        Returns:
            tuple[ArrayLike | None, ArrayLike | None]: Tuple containing the time points and the integrated values.

        """
        try:
            t, y = self.integrator.simulate(time_points[-1], 0, time_points)
            return (
                np.atleast_1d(np.array(t, dtype=float)),
                np.atleast_2d(np.array(y, dtype=float)),
            )
        except CVodeError:
            return None, None

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        t_max: float = 1_000_000_000,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the ODE system to steady state.

        Args:
            tolerance: Tolerance for determining steady state.
            rel_norm: Whether to use relative normalization.
            t_max: Maximum time point for the integration (default: 1,000,000,000).

        Returns:
            tuple[float | None, ArrayLike | None]: Tuple containing the final time point and the integrated values at steady state.

        """
        self.reset()

        try:
            for t_end in np.geomspace(1000, t_max, 3):
                t, y = self.integrator.simulate(t_end)
                diff = (y[-1] - y[-2]) / y[-1] if rel_norm else y[-1] - y[-2]
                if np.linalg.norm(diff, ord=2) < tolerance:
                    return t[-1], y[-1]
        except CVodeError:
            return None, None
        return None, None
