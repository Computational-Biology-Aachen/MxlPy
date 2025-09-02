"""Scipy integrator for solving ODEs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import scipy.integrate as spi

from mxlpy.types import Array, ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Rhs


__all__ = [
    "Scipy",
]


@dataclass
class Scipy:
    """Scipy integrator for solving ODEs.

    Attributes:
        rhs: Right-hand side function of the ODE.
        y0: Initial conditions.
        atol: Absolute tolerance for the solver.
        rtol: Relative tolerance for the solver.
        t0: Initial time point.
        _y0_orig: Original initial conditions.

    Methods:
        __post_init__: Initialize the Scipy integrator.
        reset: Reset the integrator.
        integrate: Integrate the ODE system.
        integrate_to_steady_state: Integrate the ODE system to steady state.

    """

    rhs: Rhs
    y0: tuple[float, ...]
    jacobian: Callable | None = None
    atol: float = 1e-8
    rtol: float = 1e-8
    t0: float = 0.0
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "LSODA"
    _y0_orig: tuple[float, ...] = field(default_factory=tuple)

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
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the ODE system.

        Args:
            t_end: Terminal time point for the integration.
            steps: Number of steps for the integration.
            time_points: Array of time points for the integration.

        Returns:
            tuple[ArrayLike | None, ArrayLike | None]: Tuple containing the time points and the integrated values.

        """
        # Scipy counts the total amount of return points rather than steps as assimulo
        steps = 100 if steps is None else steps + 1

        return self.integrate_time_course(
            time_points=np.linspace(self.t0, t_end, steps, dtype=float)
        )

    def integrate_time_course(
        self, *, time_points: ArrayLike
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the ODE system over a time course.

        Args:
            time_points: Time points for the integration.

        Returns:
            tuple[ArrayLike, ArrayLike]: Tuple containing the time points and the integrated values.

        """
        if time_points[0] != self.t0:
            time_points = np.insert(time_points, 0, self.t0)

        res = spi.solve_ivp(
            fun=self.rhs,
            y0=self.y0,
            t_span=(time_points[0], time_points[-1]),
            t_eval=time_points,
            jac=self.jacobian,
            atol=self.atol,
            rtol=self.rtol,
            method=self.method,
        )

        if res.success:
            t = np.atleast_1d(np.array(res.t, dtype=float))
            y = np.atleast_2d(np.array(res.y, dtype=float).T)

            self.t0 = t[-1]
            self.y0 = y[-1]
            return t, y
        return None, None

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        step_size: int = 100,
        max_steps: int = 1000,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the ODE system to steady state.

        Args:
            tolerance: Tolerance for determining steady state.
            rel_norm: Whether to use relative normalization.
            step_size: Step size for the integration (default: 100).
            max_steps: Maximum number of steps for the integration (default: 1,000).
            integrator: Name of the integrator to use (default: "lsoda").

        Returns:
            tuple[float | None, ArrayLike | None]: Tuple containing the final time point and the integrated values at steady state.

        """
        self.reset()

        # If rhs returns a tuple, we get weird errors, so we need
        # to wrap this in a list for some reason
        integ = spi.ode(lambda t, x: list(self.rhs(t, x)), jac=self.jacobian)
        integ.set_integrator(name=self.method)
        integ.set_initial_value(self.y0)

        t = self.t0 + step_size
        y1 = copy.deepcopy(self.y0)
        for _ in range(max_steps):
            y2 = integ.integrate(t)
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1
            if np.linalg.norm(diff, ord=2) < tolerance:
                return t, cast(ArrayLike, y2)
            y1 = y2
            t += step_size
        return None, None
