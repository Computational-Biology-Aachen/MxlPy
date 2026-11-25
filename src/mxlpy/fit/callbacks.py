# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Progress events of fits.

Every fit function takes `callback=`, which is called with a `FitEvent`
after every evaluation of the objective and when the fit finished or
failed. Multi-start, joint and the identifiability and selection analyses
pass it on to each of their fits and tag the events with the `start`.

Fits running in parallel worker processes get a pickled copy of the
callback, so its effects have to leave the process, e.g. `JsonlCallback`,
which appends the events to a file. An in-memory sink like `events.append`
only collects events with `parallel=False`: each worker appends to its own
copy, which is lost.

Examples
--------
    >>> events = []
    >>> fit.time_course(model, p0=p0, data=data, callback=events.append)
    >>> fit.group_time_course(model, p0=starts, data=data,
    ...                       callback=fit.JsonlCallback("events.jsonl"))

"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from mxlpy.minimizers.abstract import Residual
    from mxlpy.types import Array

__all__ = ["Callback", "EventKind", "FitEvent", "JsonlCallback", "emit"]

type EventKind = Literal["evaluation", "finished", "failed"]


@dataclass(frozen=True, kw_only=True, slots=True)
class FitEvent:
    """Progress of a fit.

    Attributes
    ----------
    kind
        `evaluation` after every evaluation of the objective, `finished` or
        `failed` once at the end
    start
        Label of the start (multi-start fits) or task (analyses running many
        fits), None for a single fit
    n_evaluations
        Evaluations of the objective so far. Least-squares minimizers
        evaluate residual vectors, which count as well
    value
        Objective (loss + penalty) of this evaluation, or of the optimum for
        `finished`. NaN if unknown, e.g. for residual vectors of losses that
        aren't sums of squares
    best
        Lowest objective so far (`inf` before the first finite one)
    improved
        Whether this evaluation improved `best`
    parameters
        Parameter values of `best`, only set if `improved` and for `finished`
    duration
        Seconds this evaluation took (the whole fit for `finished` and
        `failed`)
    message
        Reason of a failure
    time
        Wall-clock time of the event (`time.time()`)

    """

    kind: EventKind
    start: str | None = None
    n_evaluations: int = 0
    value: float = math.nan
    best: float = math.inf
    improved: bool = False
    parameters: dict[str, float] | None = None
    duration: float = 0.0
    message: str | None = None
    time: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict, with non-finite floats as None."""
        out = asdict(self)
        for k in ("value", "best"):
            if not math.isfinite(out[k]):
                out[k] = None
        if out["parameters"] is not None:
            out["parameters"] = {
                k: (float(v) if math.isfinite(v) else None)
                for k, v in out["parameters"].items()
            }
        return out


type Callback = Callable[[FitEvent], None]


@dataclass(frozen=True)
class JsonlCallback:
    """Append events to a JSON-lines file, one object per line.

    Picklable, so it works from parallel workers: every event is written
    with a single append, which doesn't interleave with other processes'
    lines. Plain evaluations are written at most every `interval` seconds
    per fit, improvements and the end of every fit always.

    Parameters
    ----------
    path
        File to append to
    interval
        Minimal seconds between two written evaluations without improvement
    kinds
        Kinds of events to write
    fields
        Extra fields added to every line, e.g. `{"step": "fit"}`

    """

    path: str | os.PathLike[str]
    interval: float = 1.0
    kinds: Collection[EventKind] = ("evaluation", "finished", "failed")
    fields: dict[str, Any] = field(default_factory=dict)
    _last: dict[str | None, float] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __call__(self, event: FitEvent) -> None:
        """Write the event, unless filtered out."""
        if event.kind not in self.kinds:
            return
        if event.kind == "evaluation" and not event.improved:
            last = self._last.get(event.start)
            if last is not None and event.time - last < self.interval:
                return
        self._last[event.start] = event.time
        line = json.dumps({**self.fields, **event.to_dict()}) + "\n"
        path = Path(self.path)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class _Tagged:
    """Callback setting the `start` of every event (picklable)."""

    callback: Callback
    start: str

    def __call__(self, event: FitEvent) -> None:
        self.callback(replace(event, start=self.start))


def tagged(callback: Callback | None, start: object) -> Callback | None:
    """The callback with `start` set on every event, None without a callback."""
    if callback is None:
        return None
    return _Tagged(callback, str(start))


def emit(callback: Callback | None, event: FitEvent) -> None:
    """Call the callback if there is one."""
    if callback is not None:
        callback(event)


def _squares_factor(residual: Any) -> float | None:
    """Factor c with objective = c Σr² / 2 of a residual's vector, if known.

    Walks the wrappers (transforms, penalties) to the residual providing it;
    penalties' pseudo-residuals are scaled to keep the relation.
    """
    while residual is not None:
        fn = getattr(type(residual), "squares_factor", None)
        if callable(fn):
            return cast("float | None", fn(residual))
        residual = (
            residual.__dict__.get("inner") if hasattr(residual, "__dict__") else None
        )
    return None


class _Tracked:
    """Residual reporting every evaluation to a callback.

    Forwards everything else (derivatives, `to_inner`, ...) to the wrapped
    residual, so minimizers detect the same capabilities.
    """

    def __init__(
        self,
        inner: Residual,
        *,
        callback: Callback,
        back: Callable[[dict[str, float]], dict[str, float]],
    ) -> None:
        self.inner = inner
        self.callback = callback
        self.back = back
        self.n_evaluations = 0
        self.best = math.inf
        self.best_parameters: dict[str, float] | None = None
        self._factor: float | bool | None = False

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _record(self, updates: dict[str, float], value: float, duration: float) -> None:
        self.n_evaluations += 1
        improved = math.isfinite(value) and value < self.best
        parameters = None
        if improved:
            self.best = value
            parameters = self.best_parameters = self.back(dict(updates))
        self.callback(
            FitEvent(
                kind="evaluation",
                n_evaluations=self.n_evaluations,
                value=value,
                best=self.best,
                improved=improved,
                parameters=parameters,
                duration=duration,
            )
        )

    def __call__(self, updates: dict[str, float]) -> float:
        t0 = time.perf_counter()
        value = self.inner(updates)
        self._record(updates, float(value), time.perf_counter() - t0)
        return value


class _TrackedVector(_Tracked):
    """`_Tracked` for residuals providing residual vectors."""

    def residuals(self, updates: dict[str, float]) -> Array:
        t0 = time.perf_counter()
        r = np.asarray(cast(Any, self.inner).residuals(updates), dtype=float)
        duration = time.perf_counter() - t0
        if self._factor is False:
            self._factor = _squares_factor(self.inner)
        value = math.nan
        if isinstance(self._factor, float) and np.isfinite(r).all():
            value = self._factor * float(np.square(r).sum()) / 2
        self._record(updates, value, duration)
        return r


def tracked(
    residual: Residual,
    callback: Callback | None,
    *,
    back: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> Residual:
    """Wrap a residual to report its evaluations, unchanged without a callback.

    `back` maps the minimizer's values to the fit's (log transforms).
    """
    if callback is None:
        return residual
    cls = _TrackedVector if callable(getattr(residual, "residuals", None)) else _Tracked
    return cast(
        "Residual",
        cls(residual, callback=callback, back=back or (lambda x: x)),
    )


def untracked(residual: Any) -> Any:
    """The residual without the tracking wrapper."""
    return residual.inner if isinstance(residual, _Tracked) else residual


def finished(
    callback: Callback | None,
    residual: Residual,
    *,
    start: str | None,
    value: float,
    parameters: dict[str, float],
    duration: float,
) -> None:
    """Report the end of a successful fit."""
    if callback is None:
        return
    n = residual.n_evaluations if isinstance(residual, _Tracked) else 0
    best = min(value, residual.best) if isinstance(residual, _Tracked) else value
    emit(
        callback,
        FitEvent(
            kind="finished",
            start=start,
            n_evaluations=n,
            value=value,
            best=best,
            parameters=parameters,
            duration=duration,
        ),
    )


def failed(
    callback: Callback | None,
    residual: Residual | None,
    *,
    start: str | None,
    message: str,
    duration: float = 0.0,
) -> None:
    """Report a failed fit."""
    if callback is None:
        return
    tracker = residual if isinstance(residual, _Tracked) else None
    emit(
        callback,
        FitEvent(
            kind="failed",
            start=start,
            n_evaluations=0 if tracker is None else tracker.n_evaluations,
            best=math.inf if tracker is None else tracker.best,
            message=message,
            duration=duration,
        ),
    )


def failure_message(error: Exception) -> str:
    """Readable reason of a failed fit."""
    extra = getattr(error, "extra_info", None)
    if extra:
        return "; ".join(str(i) for i in extra)
    return str(error) or type(error).__name__
