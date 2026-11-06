# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""SUNDIALS (CVODE) integrator for solving ODEs, via the official sundials4py bindings."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sundials4py.core import (  # type: ignore
    SUN_COMM_NULL,
    N_VGetNumpyArray,
    N_VNew_Serial,
    SUNContext_ClearErrHandlers,
    SUNContext_Create,
    SUNDenseMatrix,
    SUNDenseMatrix_Data,
    SUNLinSol_Dense,
)
from sundials4py.cvodes import (  # type: ignore
    CV_ADAMS,
    CV_BDF,
    CV_NORMAL,
    CV_ONE_STEP,
    CVode,
    CVodeCreate,
    CVodeGetReturnFlagName,
    CVodeInit,
    CVodeReInit,
    CVodeSetJacFn,
    CVodeSetLinearSolver,
    CVodeSetMaxConvFails,
    CVodeSetMaxErrTestFails,
    CVodeSetMaxNumSteps,
    CVodeSetStopTime,
    CVodeSStolerances,
)

from mxlpy.integrators.abstract import (
    AbstractIntegrator,
    TimeCourse,
)
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.types import IntegrationFailure, NoSteadyState, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.types import Array, ArrayLike, Rhs


__all__ = [
    "Sundials",
    "SundialsError",
]

_METHODS = {"BDF": CV_BDF, "Adams": CV_ADAMS}


class SundialsError(IntegrationFailure):
    """CVODE failed, carrying its return flag (and the Python error behind it, if any)."""

    def __init__(self, flag: int) -> None:
        """Initialise from a CVODE return flag."""
        Exception.__init__(self, f"CVODE failed with {CVodeGetReturnFlagName(flag)}")
        self.flag = flag


def _weak(method: Callable[..., int]) -> Callable[..., int]:
    """Callback that doesn't keep its integrator alive.

    sundials4py holds callbacks from C, where the GC can't see a cycle through
    them, so passing a bound method would leak every integrator.
    """
    ref = weakref.WeakMethod(method)

    def callback(*args: Any) -> int:
        return ref()(*args)  # type: ignore[misc]

    return callback


def _check(status: int, fn: str) -> None:
    if status < 0:
        msg = f"{fn} failed with {CVodeGetReturnFlagName(status)}"
        raise RuntimeError(msg)


@dataclass
class Sundials(AbstractIntegrator):
    """CVODE from SUNDIALS, through the official sundials4py bindings.

    The same solver as `Assimulo`, with the same defaults, but installable
    from PyPI (`pip install mxlpy[sundials]`).

    Attributes
    ----------
    rhs
        Right-hand side function of the ODE.
    y0
        Initial conditions.
    jacobian
        Optional analytic Jacobian `f(t, y) -> (n, n)`; finite differences otherwise.
    atol
        Absolute tolerance for the solver.
    rtol
        Relative tolerance for the solver.
    maxnef
        Maximum number of error test failures per step.
    maxncf
        Maximum number of convergence failures per step.
    maxsteps
        Maximum number of internal steps between two output time points
        (CVODE's mxstep, as in Assimulo).
    method
        Linear multistep method: BDF (stiff) or Adams (non-stiff).

    """

    rhs: Rhs
    y0: tuple[float, ...]
    jacobian: Callable | None = None
    atol: float = 1e-8
    rtol: float = 1e-8
    maxnef: int = 4  # max error failures
    maxncf: int = 1  # max convergence failures
    maxsteps: int = 10_000
    method: Literal["BDF", "Adams"] = "BDF"
    _t: float = field(default=0.0, init=False, repr=False)
    _cb_error: BaseException | None = field(default=None, init=False, repr=False)
    _objs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Set up CVODE with a dense Newton linear solver."""
        if self.method not in _METHODS:
            msg = f"method must be one of {list(_METHODS)}, got {self.method!r}"
            raise ValueError(msg)
        n = len(self.y0)
        status, ctx = SUNContext_Create(SUN_COMM_NULL)
        _check(status, "SUNContext_Create")
        # Failures are reported through the returned Result instead
        _check(SUNContext_ClearErrHandlers(ctx), "SUNContext_ClearErrHandlers")

        y = N_VNew_Serial(n, ctx)
        self._y = N_VGetNumpyArray(y)
        self._y[:] = self.y0

        cvode = CVodeCreate(_METHODS[self.method], ctx)
        mem = cvode.get()
        _check(CVodeInit(mem, _weak(self._rhs), 0.0, y), "CVodeInit")
        A = SUNDenseMatrix(n, n, ctx)
        LS = SUNLinSol_Dense(y, A, ctx)
        _check(CVodeSetLinearSolver(mem, LS, A), "CVodeSetLinearSolver")
        if self.jacobian is not None:
            _check(CVodeSetJacFn(mem, _weak(self._jac)), "CVodeSetJacFn")

        # SUNDIALS objects are freed by the GC, so keep them alive with self
        self._objs = {"ctx": ctx, "y": y, "cvode": cvode, "A": A, "LS": LS}
        self._mem = mem

    def _rhs(self, t: float, yv: Any, ydotv: Any, _: Any) -> int:
        try:
            N_VGetNumpyArray(ydotv)[:] = self.rhs(t, N_VGetNumpyArray(yv))
        except Exception as e:  # noqa: BLE001
            # Python errors escaping the C solver would leave it in an undefined
            # state, so stop it with an unrecoverable flag and report e as the cause
            self._cb_error = e
            return -1
        return 0

    def _jac(self, t: float, yv: Any, _fy: Any, jac: Any, *_: Any) -> int:
        try:
            SUNDenseMatrix_Data(jac)[:, :] = np.asarray(
                self.jacobian(t, N_VGetNumpyArray(yv)),  # type: ignore[misc]
                dtype=float,
            )
        except Exception as e:  # noqa: BLE001
            self._cb_error = e
            return -1
        return 0

    def _failure(self, flag: int) -> Result[TimeCourse]:
        err = SundialsError(flag)
        err.__cause__, self._cb_error = self._cb_error, None
        return Result(err)

    def reset(self) -> None:
        """Reset the integrator to its initial state."""
        self._t = 0.0
        self._y[:] = self.y0

    def _restart(self, t_stop: float) -> None:
        """Restart CVODE from the current state, never stepping past `t_stop`.

        Like Assimulo, every call starts afresh, so parameter changes between
        calls (e.g. protocol steps) don't trip CVODE's step-history error test,
        and changed settings (e.g. `sim.integrator.rtol`) take effect.
        """
        mem = self._mem
        _check(CVodeReInit(mem, self._t, self._objs["y"]), "CVodeReInit")
        _check(CVodeSStolerances(mem, self.rtol, self.atol), "CVodeSStolerances")
        _check(CVodeSetMaxErrTestFails(mem, self.maxnef), "CVodeSetMaxErrTestFails")
        _check(CVodeSetMaxConvFails(mem, self.maxncf), "CVodeSetMaxConvFails")
        _check(CVodeSetMaxNumSteps(mem, self.maxsteps), "CVodeSetMaxNumSteps")
        _check(CVodeSetStopTime(mem, t_stop), "CVodeSetStopTime")

    def _solve_at(self, time_points: Array) -> Result[TimeCourse]:
        """Values at the current time followed by each of `time_points`."""
        ts, ys = [self._t], [self._y.copy()]
        if len(time_points) > 0:
            self._restart(float(time_points[-1]))
        for tout in time_points:
            status, t = CVode(self._mem, tout, self._objs["y"], CV_NORMAL)
            if status < 0:
                return self._failure(status)
            self._t = t
            ts.append(t)
            ys.append(self._y.copy())
        return Result(TimeCourse(time=np.array(ts), values=np.array(ys)))

    def _solve_steps(self, t_end: float) -> Result[TimeCourse]:
        """Values at every internal step, ending exactly at `t_end`."""
        ts, ys = [self._t], [self._y.copy()]
        if self._t < t_end:
            self._restart(t_end)
        while self._t < t_end:
            status, t = CVode(self._mem, t_end, self._objs["y"], CV_ONE_STEP)
            if status < 0:
                return self._failure(status)
            self._t = t
            ts.append(t)
            ys.append(self._y.copy())
        return Result(TimeCourse(time=np.array(ts), values=np.array(ys)))

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
            Number of equidistant output steps. If None (or 0), every
            internal solver step is returned.

        Returns
        -------
        Result[TimeCourse]
            Values from the current time up to `t_end`, current time included.

        """
        if not steps:
            return self._solve_steps(t_end)
        return self._solve_at(np.linspace(self._t, t_end, steps + 1)[1:])

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system over a time course.

        Parameters
        ----------
        time_points
            Time points for the integration, in increasing order.

        Returns
        -------
        Result[TimeCourse]
            Values at the current time followed by each requested time point.

        """
        t_out = np.asarray(time_points, dtype=float)
        if (np.diff(t_out) < 0).any() or t_out[0] < self._t:
            msg = "Sundials needs time points in increasing order, starting at or after the current time"
            raise ValueError(msg)
        if t_out[0] == self._t:
            t_out = t_out[1:]
        return self._solve_at(t_out)

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        oscillation_detector: OscillationDetector | None = detect_oscillations,
        t_max: float = 1_000_000_000,
        max_detect_samples: int = 1000,
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
        t_max
            Maximum time point for the integration (default: 1,000,000,000).
        max_detect_samples
            Maximum number of trajectory samples forwarded to the oscillation
            detector.  Longer segments are uniformly downsampled before
            analysis.  Default: 1,000.

        Returns
        -------
        Result[TimeCourse]
            The final time point and the values at steady state.

        """
        self.reset()

        for t_end in np.geomspace(1000, t_max, 3):
            res = self._solve_steps(float(t_end))
            if not isinstance(tc := res.value, TimeCourse):
                return res
            t = tc.time
            y = np.asarray(tc.values)
            diff = (y[-1] - y[-2]) / y[-1] if rel_norm else y[-1] - y[-2]
            if np.linalg.norm(diff, ord=2) < tolerance:
                return Result(
                    TimeCourse(
                        time=np.array([t[-1]], dtype=float),
                        values=np.array([y[-1]], dtype=float),
                    )
                )

            if oscillation_detector is not None:
                stride = max(1, len(y) // max_detect_samples)
                var_names = [str(i) for i in range(y.shape[1])]
                if (
                    osc := oscillation_detector(
                        y[::stride],
                        var_names,
                        times=t[::stride],
                    )
                ) is not None:
                    return Result(osc)

        return Result(NoSteadyState())
