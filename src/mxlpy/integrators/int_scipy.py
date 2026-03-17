"""Scipy integrator for solving ODEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.integrate as spi

from mxlpy.integrators.abstract import (
    AbstractIntegrator,
    TimeCourse,
)
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.types import ArrayLike, IntegrationFailure, NoSteadyState, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Rhs


__all__ = [
    "Scipy",
]


@dataclass
class Scipy(AbstractIntegrator):
    """Scipy integrator for solving ODEs.

    Attributes
    ----------
    rhs
        Right-hand side function of the ODE.
    y0
        Initial conditions.
    atol
        Absolute tolerance for the solver.
    rtol
        Relative tolerance for the solver.
    t0
        Initial time point.
    _y0_orig
        Original initial conditions.

    Methods
    -------
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
    ) -> Result[TimeCourse]:
        """Integrate the ODE system.

        Parameters
        ----------
        t_end
            Terminal time point for the integration.
        steps
            Number of steps for the integration.
        time_points
            Array of time points for the integration.

        Returns
        -------
        tuple[ArrayLike | None, ArrayLike | None]
            Tuple containing the time points and the integrated values.

        """
        # Scipy counts the total amount of return points rather than steps as assimulo
        steps = 100 if steps is None else steps + 1

        return self.integrate_time_course(
            time_points=np.linspace(self.t0, t_end, steps, dtype=float)
        )

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system over a time course.

        Parameters
        ----------
        time_points
            Time points for the integration.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Tuple containing the time points and the integrated values.

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
            return Result(TimeCourse(time=t, values=y))
        return Result(IntegrationFailure())

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        oscillation_detector: OscillationDetector | None = detect_oscillations,
        step_size: int = 100,
        max_steps: int = 1000,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system to steady state.

        Parameters
        ----------
        tolerance
            Tolerance for determining steady state.
        rel_norm
            Whether to use relative normalization.
        oscillation_detector
            Callable that analyses each trajectory segment and returns an
            :class:`~mxlpy.types.OscillationDetected` exception when
            oscillatory behaviour is found, or ``None`` otherwise.  Pass
            :func:`no_oscillation_detection` to disable detection entirely.
            Default: :func:`detect_oscillations` (autocorrelation-based).
        step_size
            Step size for the integration (default: 100).
        max_steps
            Maximum number of steps for the integration (default: 1,000).

        Returns
        -------
        tuple[float | None, ArrayLike | None]
            Tuple containing the final time point and the integrated values at steady state.

        """
        self.reset()

        y1 = np.array(self.y0, dtype=float)
        t0 = self.t0

        for _ in range(max_steps):
            t_end = t0 + step_size
            # No t_eval: solver returns all internal steps, giving a dense
            # trajectory that is later used for oscillation detection.
            res = spi.solve_ivp(
                fun=self.rhs,
                y0=y1,
                t_span=(t0, t_end),
                method=self.method,
                atol=self.atol,
                rtol=self.rtol,
                jac=self.jacobian,
            )
            if not res.success:
                return Result(IntegrationFailure())

            y2 = res.y[:, -1]
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1

            if np.linalg.norm(diff, ord=2) < tolerance:
                return Result(
                    TimeCourse(
                        time=np.array([t_end], dtype=float),
                        values=np.array([y2], dtype=float),
                    )
                )

            # Not converging - check the dense trajectory from this step for
            # oscillatory behaviour and return early if detected.
            if oscillation_detector is not None:
                hist = res.y.T  # (N_internal, n_vars)
                var_names = [str(i) for i in range(hist.shape[1])]
                if (
                    osc := oscillation_detector(hist, var_names, times=res.t)
                ) is not None:
                    return Result(osc)

            y1 = y2
            t0 = t_end

        return Result(NoSteadyState())
