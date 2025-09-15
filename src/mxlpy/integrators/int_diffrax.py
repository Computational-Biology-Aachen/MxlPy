"""Diffrax integrator for solving ODEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)

from mxlpy.integrators.abstract import AbstractIntegrator, TimeCourse
from mxlpy.types import ArrayLike, NoSteadyState, Result

__all__ = ["Diffrax"]

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Rhs


@dataclass
class Diffrax(AbstractIntegrator):
    """Diffrax integrator for solving ODEs."""

    rhs: Rhs
    y0: tuple[float, ...]
    jac: Callable | None = None
    solver: AbstractSolver = field(default=Kvaerno5())
    stepsize_controller: AbstractStepSizeController = field(
        default=PIDController(rtol=1e-8, atol=1e-8)
    )
    t0: float = 0.0

    def __post_init__(self) -> None:
        """Create copy of initial state.

        This method creates a copy of the initial state `y0` and stores it in the `_y0_orig` attribute.
        This is useful for preserving the original initial state for future reference or reset operations.

        """
        self._y0_orig = self.y0

    def reset(self) -> None:
        """Reset the integrator."""
        self.t0 = 0
        self.y0 = self._y0_orig

    def integrate(
        self,
        *,
        t_end: float,
        steps: int | None = None,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system over a time course."""
        steps = 100 if steps is None else steps

        return self.integrate_time_course(
            time_points=np.linspace(self.t0, t_end, steps, dtype=float)
        )

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system over a time course.

        Args:
            time_points: Time points for the integration.

        Returns:
            tuple[Array, Array]: Tuple containing the time points and the integrated values.

        """
        if time_points[0] != self.t0:
            time_points = np.insert(time_points, 0, self.t0)

        res = diffeqsolve(
            ODETerm(lambda t, y, _: self.rhs(t, y)),  # type: ignore
            solver=self.solver,
            t0=time_points[0],
            t1=time_points[-1],
            dt0=None,
            y0=self.y0,
            max_steps=None,
            saveat=SaveAt(ts=time_points),  # type: ignore
            stepsize_controller=self.stepsize_controller,
        )

        t = np.atleast_1d(np.array(res.ts, dtype=float))
        y = np.atleast_2d(np.array(res.ys, dtype=float).T)

        self.t0 = t[-1]
        self.y0 = y[-1]
        return Result(TimeCourse(time=t, values=y))

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        t_max: float = 1_000_000_000,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system to steady state.

        Args:
            tolerance: Tolerance for determining steady state.
            rel_norm: Whether to use relative normalization.
            t_max: Maximum time point for the integration (default: 1,000,000,000).

        Returns:
            tuple[float | None, Array | None]: Tuple containing the final time point and the integrated values at steady state.

        """
        t_start = 0.0
        for t_end in np.geomspace(1000, t_max, 3):
            res = self.integrate_time_course(
                time_points=np.linspace(t_start, t_end, 1000, dtype=float)
            )
            match res.value:
                case TimeCourse(t, y):
                    diff = (y[-1] - y[-2]) / y[-1] if rel_norm else y[-1] - y[-2]
                    if np.linalg.norm(diff, ord=2) < tolerance:
                        return Result(
                            TimeCourse(
                                time=np.array([t[-1]], dtype=float),
                                values=np.array([y[-1]], dtype=float),
                            )
                        )
                case _:
                    return Result(res.value)
            t_start = t_end

        return Result(NoSteadyState())
