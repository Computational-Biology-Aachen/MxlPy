# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Native integrator: compiles a lowerable model and solves it outside Python.

This is far faster than the Python integrators, but only works for models that can be
represented symbolically. Install with `mxlpy[native]` (llvmlite and the compiled
solvers from mxlpy-native, where wheels exist, otherwise the WASM backend) or
`mxlpy[native-wasm]` (wasmtime only, works everywhere but slower). If a backend is
installed but broken, the error says how to fix it; `python -m mxlpy_native` diagnoses
the llvm backend.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import numpy as np
import pandas as pd

from mxlpy.integrators._native.backends import installed, problem
from mxlpy.integrators._native.lower import (
    NativeLoweringError,
    content_key,
    fingerprint,
    lower,
    lower_outputs,
)
from mxlpy.integrators.abstract import AbstractIntegrator, TimeCourse
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.parallel import register_worker_state
from mxlpy.types import ArrayLike, IntegrationFailure, NoSteadyState, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from mxlpy.types import Array, Rhs

__all__ = ["Native", "NativeLoweringError", "args_time_course"]

_AVAILABLE = installed()
if not _AVAILABLE:
    msg = (
        'Native needs an optional backend: pip install "mxlpy[native]" (fastest), or '
        '"mxlpy[native-wasm]" (no compiled dependencies, works everywhere)'
    )
    raise ImportError(msg)

Method = Literal["Radau", "LSODA", "CVODE_BDF", "CVODE_Adams"]
METHODS: tuple[str, ...] = get_args(Method)

_LOGGER = logging.getLogger(__name__)
_CACHE_SIZE = 32
_CHECK_RTOL = 1e-8
_CHECK_STATES = 3
# The solver keeps global state, so only one solve can run at a time in a process.
_LOCK = threading.RLock()


class _Compiled(Protocol):
    blob: bytes

    def rhs(self, t: float, y: Array, pars: Array) -> Array: ...

    def solve(
        self,
        y0: Array,
        pars: Array,
        t_out: Array,
        *,
        rtol: float,
        atol: float,
        nmax: int,
        method: str,
        reuse_jacobian: bool,
    ) -> tuple[int, Array]: ...

    def eval_rows(self, t: Array, y: Array, pars: Array, n_out: int) -> Array: ...


@dataclass
class _Entry:
    """Compiled code, with what's needed to load it in another process."""

    backend: str
    blob: bytes
    par_names: list[str]
    # The output names, for the compiled simulation outputs
    names: list[str] = field(default_factory=list)
    _compiled: _Compiled | None = None

    @property
    def compiled(self) -> _Compiled:
        if (compiled := self._compiled) is None:
            compiled = self._compiled = _backend_module(self.backend).load(self.blob)
        return compiled


# Compiled simulation outputs are None for models that need the Python path.
type _Cached = _Entry | None

# By `fingerprint`: hits are certain, but only for the very same model objects.
_CACHE: OrderedDict[Hashable, _Cached] = OrderedDict()
# By `content_key`: survives pickling, so it serves a model's copies in parallel
# workers (and is sent to them), but hits have to be checked.
_SHARED: OrderedDict[Hashable, _Cached] = OrderedDict()
# wasmtime objects can't be freed once the interpreter has started shutting down.
atexit.register(_CACHE.clear)
atexit.register(_SHARED.clear)


def _backend_module(backend: str) -> Any:
    return importlib.import_module(f"mxlpy.integrators._native.{backend}_backend")


def _store(
    cache: OrderedDict[Hashable, _Cached], key: Hashable, entry: _Cached
) -> None:
    cache[key] = entry
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def _shared_hit(
    key: Hashable | None, check: Callable[[_Entry], None]
) -> _Cached | bool:
    """Return a checked `_SHARED` entry, or False if there's none to use."""
    if key is None or key not in _SHARED:
        return False
    if (entry := _SHARED[key]) is None:
        return None
    try:
        check(entry)
    except (NativeLoweringError, ArithmeticError, ValueError) as e:
        _LOGGER.debug("Native recompiles a model whose content key matched: %s", e)
        return False
    except Exception as e:  # noqa: BLE001  loading code shipped by another process
        _LOGGER.debug("Native can't load shared compiled code: %r", e)
        return False
    _SHARED.move_to_end(key)
    return entry


def _export() -> (
    list[tuple[Hashable, tuple[str, bytes, list[str], list[str]] | None]] | None
):
    """The shared compiled code, for parallel workers."""
    with _LOCK:
        return [
            (k, None if e is None else (e.backend, e.blob, e.par_names, e.names))
            for k, e in _SHARED.items()
        ] or None


def _install(
    entries: list[tuple[Hashable, tuple[str, bytes, list[str], list[str]] | None]],
) -> None:
    with _LOCK:
        for key, entry in entries:
            _store(_SHARED, key, None if entry is None else _Entry(*entry))


register_worker_state(_export, _install)


def _parameters(model: Any, names: list[str]) -> Array:
    values = model.get_parameter_values()
    # The solver reads pars even when a model has none, so never hand it an empty array.
    return np.array([values[p] for p in names] or [0.0], dtype=float)


def _check_rhs(
    compiled: _Compiled, model: Any, names: list[str], t: float, y0: Array
) -> None:
    """Compare compiled code with the model itself, to catch lowering errors.

    Uses perturbed states, because at a steady state the right-hand side is only
    rounding noise, which can't be compared. Models that can't evaluate a state skip it.
    """
    pars = _parameters(model, names)
    rng = np.random.default_rng(0)
    scale = np.where(y0 > 0, y0, 1.0)
    for _ in range(_CHECK_STATES):
        y = scale * rng.uniform(0.5, 1.5, len(y0))
        try:
            expected = np.asarray(model(t, y), dtype=float)
        except (ArithmeticError, ValueError):
            continue
        got = compiled.rhs(t, y, pars)
        finite = expected[np.isfinite(expected)]
        atol = _CHECK_RTOL * (np.abs(finite).max() if finite.size else 1.0) + 1e-12
        close = np.isclose(got, expected, rtol=_CHECK_RTOL, atol=atol, equal_nan=True)
        if not close.all():
            msg = f"Compiled model disagrees with the model at variables {np.flatnonzero(~close).tolist()}"
            raise NativeLoweringError(msg)


def _pick_backend() -> str:
    """Return the fastest working backend, warning about installed ones that are broken."""
    broken = {}
    for backend in _AVAILABLE:
        if (reason := problem(backend)) is None:
            for skipped, why in broken.items():
                _LOGGER.warning(
                    "Native uses %s, not the broken %s backend: %s",
                    backend,
                    skipped,
                    why,
                )
            return backend
        broken[backend] = reason
    details = "\n".join(f"  {b}: {why}" for b, why in broken.items())
    msg = f"None of Native's installed backends can be used:\n{details}"
    raise ImportError(msg)


def _compile(model: Any, backend: str, t: float, y0: Array) -> _Entry:
    key = (backend, fingerprint(model))
    with _LOCK:
        if (hit := _CACHE.get(key)) is not None:
            _CACHE.move_to_end(key)
            return hit
        shared = None if (c := content_key(model)) is None else ("rhs", backend, c)
        entry = _shared_hit(
            shared,
            lambda e: _check_rhs(e.compiled, model, e.par_names, t, y0),
        )
        if not isinstance(entry, _Entry):
            low = lower(model)
            compiled = _backend_module(backend).compile_model(low)
            _check_rhs(compiled, model, low.par_names, t, y0)
            entry = _Entry(backend, compiled.blob, low.par_names, _compiled=compiled)
            if shared is not None:
                _store(_SHARED, shared, entry)
        _store(_CACHE, key, entry)
        return entry


def _check_outputs(
    compiled: _Compiled,
    model: Any,
    par_names: list[str],
    variables: pd.DataFrame,
    python: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Compare compiled outputs with the Python path, at perturbed states."""
    rng = np.random.default_rng(0)
    values = variables.to_numpy(dtype=float)[:1]
    scale = np.where(values > 0, values, 1.0)
    states = scale * rng.uniform(0.5, 1.5, (_CHECK_STATES, values.shape[1]))
    index = np.asarray(variables.index[:1], dtype=float)[0] + np.arange(
        _CHECK_STATES, dtype=float
    )
    expected = python(pd.DataFrame(states, index=index, columns=variables.columns))
    got = compiled.eval_rows(
        index, states, _parameters(model, par_names), expected.shape[1]
    )
    want = expected.to_numpy(dtype=float)
    finite = want[np.isfinite(want)]
    atol = _CHECK_RTOL * (np.abs(finite).max() if finite.size else 1.0) + 1e-12
    close = np.isclose(got, want, rtol=_CHECK_RTOL, atol=atol, equal_nan=True).all(
        axis=0
    )
    if not close.all():
        msg = f"Compiled outputs disagree with the model for {list(expected.columns[~close])}"
        raise NativeLoweringError(msg)


def _compile_outputs(
    model: Any,
    backend: str,
    variables: pd.DataFrame,
    python: Callable[[pd.DataFrame], pd.DataFrame],
) -> _Entry | None:
    """Compile every reportable value of a model, or return None if that's not possible."""
    key = ("outputs", backend, fingerprint(model))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    shared = None if (c := content_key(model)) is None else ("outputs", backend, c)
    names: list[str] = []

    def check(entry: _Entry) -> None:
        if entry.names != names:
            msg = "Compiled outputs have other names"
            raise NativeLoweringError(msg)
        _check_outputs(entry.compiled, model, entry.par_names, variables, python)

    entry: _Entry | None = None
    try:
        names = [str(c) for c in python(variables.iloc[:1]).columns]
        if (hit := _shared_hit(shared, check)) is not False:
            entry = hit  # type: ignore[assignment]
        else:
            low = lower_outputs(model, names)
            compiled = _backend_module(backend).compile_model(low)
            _check_outputs(compiled, model, low.par_names, variables, python)
            entry = _Entry(backend, compiled.blob, low.par_names, names, compiled)
            if shared is not None:
                _store(_SHARED, shared, entry)
    except (NativeLoweringError, ArithmeticError, ValueError) as e:
        _LOGGER.debug("Native falls back to Python for simulation outputs: %s", e)
        if shared is not None:
            _store(_SHARED, shared, None)
    _store(_CACHE, key, entry)
    return entry


def args_time_course(
    model: Any,
    variables: pd.DataFrame,
    python: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    backend: str,
) -> pd.DataFrame | None:
    """Compute a simulation's reportable values in compiled code.

    `python` is the regular computation. It defines the columns and validates the
    compiled code once per model. Returns None if the model has to use it instead.
    """
    with _LOCK:
        if (entry := _compile_outputs(model, backend, variables, python)) is None:
            return None
        values = entry.compiled.eval_rows(
            np.asarray(variables.index, dtype=float),
            variables.to_numpy(dtype=float),
            _parameters(model, entry.par_names),
            len(entry.names),
        )
    return pd.DataFrame(values, index=variables.index, columns=entry.names)


@dataclass
class Native(AbstractIntegrator):
    """Integrator running compiled model code and a C solver outside Python.

    Attributes
    ----------
    rhs
        The model. Plain callables can't be compiled.
    y0
        Initial conditions.
    jacobian
        Ignored, the solver builds its own numerical Jacobian.
    atol
        Absolute tolerance for the solver.
    rtol
        Relative tolerance for the solver.
    t0
        Initial time point.
    backend
        `llvm`, `wasm`, or `auto` to take the fastest one installed.
    method
        `Radau` (RADAU5, for stiff models), `LSODA`, which switches between Adams
        and BDF methods as stiffness changes (SciPy's own C code), or SUNDIALS' CVODE
        with BDF (`CVODE_BDF`, stiff) or Adams-Moulton (`CVODE_Adams`, non-stiff)
        methods, the solver Assimulo's `CVode` runs too.
    reuse_jacobian
        Radau only. Reuse Jacobians for longer. Faster on larger stiff models, but less accurate
        on some small ones, and too rough for fits that use finite differences.
    nmax
        Maximal number of solver steps.

    """

    rhs: Rhs
    y0: tuple[float, ...]
    jacobian: Callable | None = None
    atol: float = 1e-8
    rtol: float = 1e-8
    t0: float = 0.0
    backend: Literal["auto", "llvm", "wasm"] = "auto"
    method: Method = "Radau"
    reuse_jacobian: bool = False
    nmax: int = 2_000_000
    _y0_orig: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Pick a backend and compile, so an unsupported model fails right away."""
        self._y0_orig = self.y0
        if self.method not in METHODS:
            msg = (
                f"Native supports the methods {', '.join(METHODS)}, not {self.method!r}"
            )
            raise ValueError(msg)
        if self.reuse_jacobian and self.method != "Radau":
            msg = "reuse_jacobian only applies to Radau"
            raise ValueError(msg)
        if self.backend not in ("auto", "llvm", "wasm"):
            msg = f"Native's backend must be 'auto', 'llvm' or 'wasm', not {self.backend!r}"
            raise ValueError(msg)
        if self.backend == "auto":
            self.backend = _pick_backend()  # type: ignore[assignment]
        elif (reason := problem(self.backend)) is not None:
            msg = f"Native's {self.backend} backend can't be used: {reason}"
            raise ImportError(msg)
        _compile(self.rhs, self.backend, self.t0, np.asarray(self.y0, dtype=float))

    @property
    def args_fn(self) -> Callable:
        """Compiled replacement for the Python computation of simulation outputs."""
        return partial(args_time_course, backend=self.backend)

    def reset(self) -> None:
        """Reset the integrator."""
        self.t0 = 0.0
        self.y0 = self._y0_orig

    def _solve(self, t_out: Array) -> tuple[Array, Array] | None:
        y0 = np.asarray(self.y0, dtype=float)
        if t_out[-1] == t_out[0]:
            return t_out, np.tile(y0, (len(t_out), 1))
        with _LOCK:
            entry = _compile(self.rhs, self.backend, self.t0, y0)
            idid, values = entry.compiled.solve(
                y0,
                _parameters(self.rhs, entry.par_names),
                t_out,
                rtol=self.rtol,
                atol=self.atol,
                nmax=self.nmax,
                method=self.method,
                reuse_jacobian=self.reuse_jacobian,
            )
        return None if idid <= 0 else (t_out, values)

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
            End time of the integration.
        steps
            Number of evenly spaced steps. Defaults to 99 if None, like Scipy.

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
        """Integrate the ODE system, returning values at the given time points.

        Parameters
        ----------
        time_points
            Time points for the integration, in increasing order.

        Returns
        -------
        Result[TimeCourse]
            Integration result containing time points and values.

        """
        t_out = np.asarray(time_points, dtype=float)
        if t_out[0] != self.t0:
            t_out = np.insert(t_out, 0, self.t0)
        if (np.diff(t_out) < 0).any():
            msg = (
                "Native needs time points in increasing order, starting at or after t0"
            )
            raise ValueError(msg)
        if (res := self._solve(t_out)) is None:
            return Result(IntegrationFailure())
        t, y = res
        self.t0 = float(t[-1])
        self.y0 = tuple(y[-1])
        return Result(TimeCourse(time=t, values=y))

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
        oscillation_detector: OscillationDetector | None = detect_oscillations,
        step_size: int = 100,
        max_steps: int = 1000,
        n_time_points: int = 100,
    ) -> Result[TimeCourse]:
        """Integrate the ODE system to steady state.

        Parameters
        ----------
        tolerance
            Tolerance for determining steady state.
        rel_norm
            Whether to use relative normalization.
        oscillation_detector
            Callable that returns an exception if a segment oscillates, or None.
            Pass `no_oscillation_detection` to disable detection.
        step_size
            Length of each integration segment.
        max_steps
            Maximum number of segments.
        n_time_points
            Number of points per segment that the oscillation detector sees.

        Returns
        -------
        Result[TimeCourse]
            Integration result containing the steady-state time and values.

        """
        self.reset()
        y1 = np.asarray(self.y0, dtype=float)
        for _ in range(max_steps):
            t_start = self.t0
            t_end = t_start + step_size
            if (res := self._solve(np.linspace(t_start, t_end, n_time_points))) is None:
                return Result(IntegrationFailure())
            t, y = res
            y2 = y[-1]
            diff = (y2 - y1) / y1 if rel_norm else y2 - y1
            if np.linalg.norm(diff, ord=2) < tolerance:
                return Result(
                    TimeCourse(
                        time=np.array([t_end], dtype=float),
                        values=np.array([y2], dtype=float),
                    )
                )
            if oscillation_detector is not None:
                var_names = [str(i) for i in range(y.shape[1])]
                if (osc := oscillation_detector(y, var_names, times=t)) is not None:
                    return Result(osc)
            y1 = y2
            self.t0 = t_end
            self.y0 = tuple(y2)
        return Result(NoSteadyState())
