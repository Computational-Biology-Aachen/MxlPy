"""Scipy integrator for solving ODEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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

# Absolute tolerance for treating another active event's trigger value as
# "also crossing zero" at the winning event's (t_event, y_at_event) state.
_TIE_TRIGGER_ATOL = 1e-6


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
    _get_dependent: Callable[[float, Array], dict[str, float]] | None = field(
        default=None
    )
    _param_update_callback: Callable[[str, float], None] | None = field(default=None)
    # Names of non-persistent events that have already fired. Persists across
    # integrate()/integrate_time_course() calls (not just within one) so a
    # one-shot event stays fired across e.g. repeated Simulator.simulate()
    # calls in simulate_protocol().
    _fired_names: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Create copy of initial state."""
        self._y0_orig = self.y0

    def reset(self) -> None:
        """Reset the integrator."""
        self.t0 = 0
        self.y0 = self._y0_orig
        self._fired_names = set()

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
        get_dependent = self._get_dependent
        assert get_dependent is not None  # noqa: S101

        def scipy_event(t: float, y: Array) -> float:
            return event.evaluate_trigger(get_dependent(t, y))

        scipy_event.terminal = True  # type: ignore[attr-defined]
        scipy_event.direction = _DIRECTION[event.direction]  # type: ignore[attr-defined]
        return scipy_event

    def _apply_tied_events(
        self,
        active: dict[str, Event],
        res: Any,
        fired_names: set[str],
        suppressed_once: set[str],
    ) -> tuple[float, Array] | None:
        """Apply the winning event plus any other active event tied at the same instant.

        scipy's `solve_ivp` stops as soon as any terminal event's root is
        found and does not keep searching for other events' roots within
        that same step - so a second event that (mathematically) crosses
        zero at the exact same instant is never populated in
        `res.t_events`, even though it did cross. Ties are therefore not
        detected by comparing scipy's per-event root arrays (there is only
        ever one populated), but by evaluating every *other* active event's
        trigger directly at the winning event's `(t_event, y_at_event)`
        state: a near-zero value there means it crossed at the same instant.
        All tied events are then applied together via `Event.apply_assignments`,
        which chains through a shared ``args`` dict (later assignments see
        earlier ones' new values).

        Parameters
        ----------
        active
            Events considered for this segment, keyed by name.
        res
            The `scipy.integrate.solve_ivp` result for the segment.
        fired_names
            Names of one-shot events already fired; extended in place for any
            non-persistent event applied here.
        suppressed_once
            Names to suppress for a short buffer window after firing
            (floating-point hazard right at the restart point); extended in
            place for any persistent event applied here.

        Returns
        -------
        tuple[float, Array] | None
            ``(t_event, y_post)`` to resume integration from, or None if no
            event fired in this segment.

        """
        event_names = list(active.keys())
        t_event = float("inf")
        fired_idx = -1
        for i, t_arr in enumerate(res.t_events):
            if len(t_arr) > 0 and float(t_arr[0]) < t_event:
                t_event = float(t_arr[0])
                fired_idx = i

        if fired_idx == -1:
            return None

        y_at_event = np.array(res.y_events[fired_idx][0], dtype=float)
        get_dependent = self._get_dependent
        assert get_dependent is not None  # noqa: S101
        args: dict[str, float] = get_dependent(t_event, y_at_event)

        tied_idxs = [fired_idx]
        for i, name in enumerate(event_names):
            if i == fired_idx:
                continue
            if abs(active[name].evaluate_trigger(args)) < _TIE_TRIGGER_ATOL:
                tied_idxs.append(i)

        y_post = y_at_event.copy()
        for idx in tied_idxs:
            fired_event = active[event_names[idx]]
            updates = fired_event.apply_assignments(args)
            for name, value in updates.items():
                if name in self._var_names:
                    y_post[self._var_names.index(name)] = value
                elif self._param_update_callback is not None:
                    self._param_update_callback(name, value)
            args.update(updates)

            if fired_event.persistent:
                suppressed_once.add(event_names[idx])
            else:
                fired_names.add(event_names[idx])

        return t_event, y_post

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
        fired_names = self._fired_names
        # Events suppressed for a short buffer window right after firing, to
        # avoid scipy re-detecting their (near-)zero trigger at the exact
        # restart point as a fresh crossing. Bounded rather than open-ended
        # so the event is re-armed shortly after and isn't lost for the rest
        # of the simulation when it is the only active event.
        suppressed_once: set[str] = set()

        while t_current < t_end - 1e-12:
            active = {
                name: ev
                for name, ev in self._events.items()
                if name not in fired_names and name not in suppressed_once
            }
            scipy_events = [self._build_scipy_event(ev) for ev in active.values()]

            seg_end = (
                min(t_end, t_current + 1e-6 * max(abs(t_current), 1.0))
                if suppressed_once
                else t_end
            )

            if seg_end < t_end:
                # Bounded buffer sub-step: only need the endpoint state to
                # re-arm the suppressed events from, not a sampled trajectory.
                res = spi.solve_ivp(
                    fun=self.rhs,
                    y0=tuple(y_current),
                    t_span=(t_current, seg_end),
                    events=scipy_events or None,
                    jac=self.jacobian,
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                )
                if not res.success and res.status != 1:
                    return Result(IntegrationFailure())

                if res.status != 1:
                    # No (non-suppressed) event fired inside the buffer
                    # window: re-arm and retry from here with the real
                    # segment end.
                    t_current = float(res.t[-1])
                    y_current = np.array(res.y[:, -1], dtype=float)
                    suppressed_once = set()
                    continue
            else:
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

            fired = self._apply_tied_events(active, res, fired_names, suppressed_once)
            if fired is None:
                break
            t_event, y_post = fired

            # Record discontinuity: post-assignment state at event time
            all_t.append(np.array([t_event], dtype=float))
            all_y.append(y_post[np.newaxis, :])

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

        if self._events:
            return self._integrate_to_steady_state_with_events(
                tolerance=tolerance,
                rel_norm=rel_norm,
                oscillation_detector=oscillation_detector,
                step_size=step_size,
                max_steps=max_steps,
            )

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

    def _integrate_to_steady_state_with_events(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        oscillation_detector: OscillationDetector | None,
        step_size: int,
        max_steps: int,
    ) -> Result[TimeCourse]:
        """Steady-state search with event handling.

        Integrates in `step_size` increments, same as the event-free path.
        Any events that cross zero inside an increment are applied via the
        same segment-restart machinery as `_integrate_with_events` before the
        increment's endpoint is used for the convergence check, so a shutoff/
        dosing event mid-search is not silently skipped. `fired_names`
        persists across increments (not just within one) so a one-shot event
        stays fired for the rest of the search.
        """
        y1 = np.array(self.y0, dtype=float)
        t0 = self.t0
        fired_names = self._fired_names
        suppressed_once: set[str] = set()

        for _ in range(max_steps):
            t_end = t0 + step_size
            t_current = t0
            y_current = y1.copy()
            hist_t: list[Array] = []
            hist_y: list[Array] = []

            while t_current < t_end - 1e-12:
                active = {
                    name: ev
                    for name, ev in self._events.items()
                    if name not in fired_names and name not in suppressed_once
                }
                scipy_events = [self._build_scipy_event(ev) for ev in active.values()]

                seg_end = (
                    min(t_end, t_current + 1e-6 * max(abs(t_current), 1.0))
                    if suppressed_once
                    else t_end
                )

                res = spi.solve_ivp(
                    fun=self.rhs,
                    y0=tuple(y_current),
                    t_span=(t_current, seg_end),
                    events=scipy_events or None,
                    jac=self.jacobian,
                    atol=self.atol,
                    rtol=self.rtol,
                    method=self.method,
                )
                if not res.success and res.status != 1:
                    return Result(IntegrationFailure())

                if seg_end < t_end and res.status != 1:
                    # No (non-suppressed) event fired inside the buffer
                    # window: re-arm and retry from here with the real
                    # increment end.
                    t_current = float(res.t[-1])
                    y_current = np.array(res.y[:, -1], dtype=float)
                    suppressed_once = set()
                    continue

                t_seg = np.array(res.t, dtype=float)
                y_seg = np.array(res.y, dtype=float).T
                hist_t.append(t_seg)
                hist_y.append(y_seg)
                t_current = float(t_seg[-1])
                y_current = y_seg[-1]

                if res.status != 1:
                    break

                fired = self._apply_tied_events(
                    active, res, fired_names, suppressed_once
                )
                if fired is None:
                    break
                t_event, y_post = fired

                hist_t.append(np.array([t_event], dtype=float))
                hist_y.append(y_post[np.newaxis, :])

                eps = 1e-10 * max(t_end - t_event, 1.0)
                t_current = t_event + eps
                y_current = y_post

            y2 = y_current
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1

            if np.linalg.norm(diff, ord=2) < tolerance:
                return Result(
                    TimeCourse(
                        time=np.array([t_end], dtype=float),
                        values=np.array([y2], dtype=float),
                    )
                )

            if oscillation_detector is not None:
                hist = np.concatenate(hist_y)
                times = np.concatenate(hist_t)
                var_names = [str(i) for i in range(hist.shape[1])]
                if (
                    osc := oscillation_detector(hist, var_names, times=times)
                ) is not None:
                    return Result(osc)

            y1 = y2
            t0 = t_end

        return Result(NoSteadyState())
