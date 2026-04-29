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
from mxlpy.types import ArrayLike, Event, IntegrationFailure, NoSteadyState, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Array, Rhs


__all__ = [
    "Scipy",
]

_DIRECTION: dict[str, int] = {"rising": 1, "falling": -1, "both": 0}


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

    """

    rhs: Rhs
    y0: tuple[float, ...]
    jacobian: Callable | None = None
    atol: float = 1e-8
    rtol: float = 1e-8
    t0: float = 0.0
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "LSODA"
    _y0_orig: tuple[float, ...] = field(default_factory=tuple)
    _events: dict[str, Event] = field(default_factory=dict)
    _var_names: list[str] = field(default_factory=list)
    _param_values: dict[str, float] = field(default_factory=dict)
    _param_update_callback: Callable[[str, float], None] | None = field(default=None)

    def __post_init__(self) -> None:
        """Create copy of initial state."""
        self._y0_orig = self.y0

    def reset(self) -> None:
        """Reset the integrator."""
        self.t0 = 0
        self.y0 = self._y0_orig

    def _build_scipy_event(self, event: Event) -> Callable[[float, Array], float]:
        """Build a scipy-compatible event callable from an Event.

        Parameters
        ----------
        event
            The event to convert.

        Returns
        -------
        Callable
            A function ``f(t, y) -> float`` with ``.terminal`` and
            ``.direction`` attributes set for scipy's event API.

        """
        var_names = self._var_names
        param_values = self._param_values

        def scipy_event(t: float, y: Array) -> float:
            args: dict[str, float] = (
                dict(zip(var_names, y, strict=False)) | param_values | {"time": t}
            )
            return event.evaluate_trigger(args)

        scipy_event.terminal = True  # type: ignore[attr-defined]
        scipy_event.direction = _DIRECTION[event.direction]  # type: ignore[attr-defined]
        return scipy_event

    def _integrate_with_events(self, time_points: Array) -> Result[TimeCourse]:
        """Segment-restart integration loop with event handling.

        Parameters
        ----------
        time_points
            Requested output time points (monotonically increasing).

        Returns
        -------
        Result[TimeCourse]
            Concatenated trajectory across all segments.

        """
        t_end = float(time_points[-1])
        all_t: list[Array] = []
        all_y: list[Array] = []

        t_current = self.t0
        y_current = np.array(self.y0, dtype=float)
        fired_names: set[str] = set()
        # Events suppressed for one segment after firing to prevent immediate
        # re-detection when the trigger is at zero at the restart point.
        suppressed_once: set[str] = set()

        while t_current < t_end - 1e-12:
            active = {
                name: ev
                for name, ev in self._events.items()
                if name not in fired_names and name not in suppressed_once
            }
            suppressed_once.clear()
            scipy_events = [self._build_scipy_event(ev) for ev in active.values()]

            mask = time_points > t_current
            seg_eval = time_points[mask]
            if len(seg_eval) == 0:
                break
            if seg_eval[0] != t_current:
                seg_eval = np.concatenate([[t_current], seg_eval])

            res = spi.solve_ivp(
                fun=self.rhs,
                y0=tuple(y_current),
                t_span=(t_current, t_end),
                t_eval=seg_eval,
                events=scipy_events or None,
                jac=self.jacobian,
                atol=self.atol,
                rtol=self.rtol,
                method=self.method,
            )

            if not res.success and res.status != 1:
                return Result(IntegrationFailure())

            t_seg = np.array(res.t, dtype=float)
            y_seg = np.array(res.y, dtype=float).T

            start = 1 if all_t else 0
            if len(t_seg) > start:
                all_t.append(t_seg[start:])
                all_y.append(y_seg[start:])

            if res.status != 1:
                break

            # Find first event that fired
            t_event = float("inf")
            fired_idx = -1
            event_names = list(active.keys())
            for i, t_arr in enumerate(res.t_events):
                if len(t_arr) > 0 and float(t_arr[0]) < t_event:
                    t_event = float(t_arr[0])
                    fired_idx = i

            if fired_idx == -1:
                break

            y_at_event = np.array(res.y_events[fired_idx][0], dtype=float)
            args: dict[str, float] = (
                dict(zip(self._var_names, y_at_event, strict=False))
                | self._param_values
                | {"time": t_event}
            )

            fired_event = active[event_names[fired_idx]]
            y_post = y_at_event.copy()
            for name, derived in fired_event.assignments.items():
                value = derived.calculate(args)
                if name in self._var_names:
                    y_post[self._var_names.index(name)] = value
                else:
                    self._param_values[name] = value
                    if self._param_update_callback is not None:
                        self._param_update_callback(name, value)
                args[name] = value

            # Record discontinuity: post-assignment state at event time
            all_t.append(np.array([t_event], dtype=float))
            all_y.append(y_post[np.newaxis, :])

            if not fired_event.persistent:
                fired_names.add(event_names[fired_idx])
            else:
                # Suppress this event for the next segment: after firing, the
                # trigger value is at or near zero and floating-point noise can
                # cause an immediate spurious re-detection.
                suppressed_once.add(event_names[fired_idx])

            # Advance slightly past t_event so time-based triggers (t - t0 = 0
            # at restart) are not immediately re-detected as crossings.
            eps = 1e-10 * max(t_end - t_event, 1.0)
            t_current = t_event + eps
            y_current = y_post

        if not all_t:
            return Result(IntegrationFailure())

        t_final = np.concatenate(all_t)
        y_final = np.concatenate(all_y)

        self.t0 = float(t_final[-1])
        self.y0 = tuple(float(v) for v in y_final[-1])

        return Result(TimeCourse(time=t_final, values=y_final))

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

        Returns
        -------
        Result[TimeCourse]
            Integration result containing time points and values.

        """
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
        Result[TimeCourse]
            Integration result containing time points and values.

        """
        time_points = np.asarray(time_points, dtype=float)

        if self._events:
            if time_points[0] != self.t0:
                time_points = np.concatenate([[self.t0], time_points])
            return self._integrate_with_events(time_points)

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
        Result[TimeCourse]
            Integration result containing the steady-state time and values.

        """
        self.reset()

        y1 = np.array(self.y0, dtype=float)
        t0 = self.t0

        for _ in range(max_steps):
            t_end = t0 + step_size
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

            if oscillation_detector is not None:
                hist = res.y.T
                var_names = [str(i) for i in range(hist.shape[1])]
                if (
                    osc := oscillation_detector(hist, var_names, times=res.t)
                ) is not None:
                    return Result(osc)

            y1 = y2
            t0 = t_end

        return Result(NoSteadyState())
