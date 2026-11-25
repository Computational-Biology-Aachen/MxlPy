# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Chart and table data of a lab run, for the web UI.

Everything here returns plain JSON-compatible dicts, so the frontend only
draws. Charts share one format::

    {"id": str, "title": str, "kind": "xy" | "category",
     "x": {"label": str, "log": bool}, "y": {"label": str, "log": bool},
     "series": [{"label": str, "type": "points" | "line" | "band" | "bar",
                 "x": [...], "y": [...],              # "band": "lower"/"upper"
                 "color": int, "dashed": bool}],
     "hlines": [{"y": float, "label": str}]}

Tables are `{"id", "title", "columns": [...], "rows": [[...], ...]}`.
`Progress` aggregates the progress stream (`events.jsonl`) of a run into
loss traces and multi-start waterfalls while it runs.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy.lab._core import STEPS, TITLES, Current, simulate_conditions, step_status
from mxlpy.lab.store import to_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mxlpy.lab._core import StepResult
    from mxlpy.lab._lab import FittingLab

__all__ = [
    "Progress",
    "StepProgress",
    "fit_charts",
    "overview",
    "step_view",
]

# Points per series of a progress trace (older points are thinned out)
_MAX_TRACE = 400


###############################################################################
# Building blocks
###############################################################################


def _floats(values: Iterable[Any]) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(f if math.isfinite(f) else None)
    return out


def chart(
    chart_id: str,
    title: str,
    series: list[dict[str, Any]],
    *,
    x: str = "",
    y: str = "",
    kind: str = "xy",
    x_log: bool = False,
    y_log: bool = False,
    hlines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A chart in the format of this module."""
    return {
        "id": chart_id,
        "title": title,
        "kind": kind,
        "x": {"label": x, "log": x_log},
        "y": {"label": y, "log": y_log},
        "series": series,
        "hlines": hlines or [],
    }


def series(
    label: str,
    x: Iterable[Any],
    y: Iterable[Any],
    *,
    kind: str = "line",
    color: int = 0,
    dashed: bool = False,
    **extra: Iterable[Any],
) -> dict[str, Any]:
    """A series of a chart (`lower`/`upper` for bands)."""
    out: dict[str, Any] = {
        "label": label,
        "type": kind,
        "x": list(x) if kind == "bar" else _floats(x),
        "y": _floats(y),
        "color": color,
        "dashed": dashed,
    }
    out |= {k: _floats(v) for k, v in extra.items()}
    return out


def table(table_id: str, title: str, obj: Any) -> dict[str, Any] | None:
    """A table of a DataFrame, Series, dict of dicts or list of records."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        df = obj.to_frame()
    elif isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    elif (
        isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values())
    ):
        df = pd.DataFrame.from_dict(obj, orient="index")
    elif isinstance(obj, dict):
        df = pd.Series(obj, dtype=object).to_frame("value")
    else:
        return None
    if df.empty:
        return None
    index_name = (
        " / ".join(str(n) for n in df.index.names if n is not None)
        if any(n is not None for n in df.index.names)
        else ""
    )
    keep_index = not isinstance(df.index, pd.RangeIndex)
    columns = ([index_name] if keep_index else []) + [str(c) for c in df.columns]
    rows = []
    for idx, row in zip(df.index, df.itertuples(index=False), strict=True):
        label = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        rows.append(([label] if keep_index else []) + [to_json(v) for v in row])
    return {"id": table_id, "title": title, "columns": columns, "rows": rows}


###############################################################################
# Steps
###############################################################################


def fit_charts(
    current: Current,
    predictions: Mapping[str, pd.DataFrame | None],
    *,
    prefix: str = "fit",
    dashed: Mapping[str, pd.DataFrame | None] | None = None,
) -> list[dict[str, Any]]:
    """Data (points) and predictions (lines) per data set."""
    out = []
    for name, cond in current.conditions.items():
        data = cast(pd.DataFrame, cond.data)
        pred = predictions.get(name)
        alt = None if dashed is None else dashed.get(name)
        lines = []
        for i, col in enumerate(data.columns):
            t = np.asarray(data.index, dtype=float)
            lines.append(series(col, t, data[col], kind="points", color=i))
            if pred is not None and col in pred:
                lines.append(series(col, pred.index, pred[col], color=i))
            if alt is not None and col in alt:
                lines.append(series(col, alt.index, alt[col], color=i, dashed=True))
        out.append(chart(f"{prefix}/{name}", name, lines, x="time", y="value"))
    return out


def _setup(result: StepResult) -> tuple[list, list]:
    s = result.summary
    current = cast(Current | None, result.artifacts.get("current"))
    charts = [] if current is None else fit_charts(current, {}, prefix="data")
    targets = {
        col: {"target": target, "sigma": _sigma_of(s.get("sigma", {}), col)}
        for col, target in s.get("targets", {}).items()
    }
    return charts, [
        table("targets", "Observables", targets),
        table("n_data", "Data points", s.get("n_data")),
    ]


def _sigma_of(sigma: Mapping[str, Any], column: str) -> str:
    values = []
    for name, v in sigma.items():
        if isinstance(v, dict):
            if column in v:
                values.append(f"{name}: {v[column]:.3g}")
        elif isinstance(v, float | int):
            values.append(f"{name}: {v:.3g}")
        else:
            values.append(f"{name}: {v}")
    return ", ".join(values)


def _parameters(result: StepResult, chosen: list[str]) -> tuple[list, list]:
    s = result.summary
    params = cast(dict[str, dict[str, Any]], s.get("parameters", {}))
    names = list(params)
    picked = [k for k in names if k in chosen]
    other = [k for k in names if k not in chosen]
    bars = [
        series(
            "estimated",
            picked,
            [params[k]["importance"] for k in picked],
            kind="bar",
            color=0,
        ),
        series(
            "fixed",
            other,
            [params[k]["importance"] for k in other],
            kind="bar",
            color=7,
        ),
    ]
    charts = [
        chart(
            "importance",
            "Sensitivity of the candidates",
            [b for b in bars if b["x"]],
            kind="category",
            y="RMS scaled sensitivity",
            y_log=True,
        )
    ]
    return charts, [
        table("parameters", "Candidates", params),
        table("limits", "Limits of the number of parameters", s.get("limits")),
    ]


def waterfall(
    objectives: Iterable[float], chart_id: str, title: str
) -> dict[str, Any] | None:
    """Sorted final objectives of the starts of a multi-start fit."""
    values = sorted(v for v in objectives if v is not None and math.isfinite(v))
    if not values:
        return None
    return chart(
        chart_id,
        title,
        [series("starts", range(1, len(values) + 1), values, kind="points")],
        x="start (sorted)",
        y="objective",
        y_log=values[0] > 0,
    )


def _fit(result: StepResult) -> tuple[list, list]:
    s = result.summary
    ev = result.artifacts.get("evaluation")
    local = result.artifacts.get("local")
    charts = []
    current = cast(Current | None, result.artifacts.get("current"))
    if ev is not None and current is not None:
        dashed = None if local is None else local[2].predictions
        charts.extend(fit_charts(current, ev.predictions, dashed=dashed))
    charts.append(waterfall(s.get("objectives", []), "waterfall", "Multi-start"))
    joint = s.get("joint", {})
    tables = [
        table("estimates", "Estimates", s.get("estimates")),
        table(
            "gof",
            "Goodness of fit",
            {k: joint.get(k) for k in ("nll", "chi2", "dof", "chi2_p", "passed")},
        ),
        table("residuals", "Residual checks", joint.get("residuals")),
        table("nrmse", "Normalised RMSE", joint.get("nrmse")),
    ]
    if "local" in s:
        tables.append(
            table("local", "Condition-specific estimates", s["local"]["estimates"])
        )
    return charts, tables


def _diagnose(result: StepResult) -> tuple[list, list]:
    s = result.summary
    charts: list[dict[str, Any] | None] = []
    hidden = cast(dict[str, dict[str, Any]], s.get("hidden", {}))
    if hidden:
        losses = {k: v.get("loss") for k, v in hidden.items() if "loss" in v}
        charts.append(
            chart(
                "hidden",
                "Loss by number of hidden states",
                [series("loss", list(losses), list(losses.values()), kind="bar")],
                kind="category",
                y="loss",
            )
        )
    chosen = s.get("chosen_hidden")
    augmented = result.artifacts.get("augmented", {})
    if chosen is not None and chosen in augmented:
        res = augmented[chosen]
        correction = cast(pd.DataFrame, res.correction)
        for i, col in enumerate(correction.columns):
            scale = res.scales.get(col, 1.0) or 1.0
            lines = []
            for j, (name, frame) in enumerate(correction[col].groupby(level=0)):
                t = frame.index.get_level_values(-1)
                lines.append(series(str(name), t, frame.to_numpy() / scale, color=j))
            charts.append(
                chart(
                    f"correction/{i}",
                    f"Correction of {col}",
                    lines,
                    x="time",
                    y="relative correction",
                )
            )
    return charts, [
        table("hidden", "Hidden states", hidden),
        table("correction", "RMS correction", s.get("correction")),
        table("variants", "Structural variants", s.get("variants")),
    ]


def _identifiability(result: StepResult) -> tuple[list, list]:
    s = result.summary
    charts = []
    profiles = result.artifacts.get("profiles")
    if profiles is not None:
        for name, profile in profiles.profiles.items():
            delta = profile.delta
            charts.append(
                chart(
                    f"profile/{name}",
                    name,
                    [series("Δ objective", delta.index, delta.to_numpy())],
                    x=name,
                    y="Δ objective",
                    x_log=bool(profile.log),
                    hlines=[{"y": float(profile.threshold), "label": "threshold"}],
                )
            )
    fisher = s.get("fisher", {})
    pulling = s.get("pulling", {})
    tables = [
        table("profile_ci", "Profile likelihood CIs", s.get("profiles", {}).get("ci")),
        table("wald_ci", "Wald CIs (Fisher information)", fisher.get("wald_ci")),
        table("displacement", "Pulling test displacement", pulling.get("displacement")),
        table("related", "Identifiable combinations", s.get("related")),
    ]
    return charts, tables


def _uncertainty(lab: FittingLab, result: StepResult) -> tuple[list, list]:
    charts = []
    for key, band in result.summary.get("bands", {}).items():
        name, _, col = key.partition("/")
        lines = [
            series(
                "band",
                band["time"],
                band["value"],
                kind="band",
                lower=band["lower"],
                upper=band["upper"],
            )
        ]
        cond = lab.conditions.get(name)
        if cond is not None and col in cond.data:
            data = cast(pd.DataFrame, cond.data)
            lines.append(
                series(
                    "data",
                    np.asarray(data.index, dtype=float),
                    data[col],
                    kind="points",
                    color=1,
                )
            )
        charts.append(chart(f"band/{key}", f"{name}: {col}", lines, x="time", y=col))
    tables = [
        table("estimates", "Estimates", result.summary.get("estimates")),
        table("fixed", "Fixed values", result.summary.get("fixed") or None),
    ]
    return charts, tables


def overview(lab: FittingLab) -> dict[str, Any]:
    """Steps, their status and gates, and the state of the run."""
    decisions = lab.decisions()
    steps = []
    open_gate: str | None = None
    for i, step in enumerate(STEPS):
        result = lab.results.get(step)
        decision = decisions.get(step)
        steps.append(
            {
                "step": step,
                "number": i,
                "title": TITLES[step],
                "status": step_status(lab.results.get(step)),
                "current": result is not None and lab.is_current(step),
                "duration": None if result is None else result.duration,
                "messages": [] if result is None else list(result.messages),
                "gate": None if decision is None else decision.get("source"),
                "has_options": result is not None and bool(result.options),
            }
        )
        if open_gate is None and result is not None and decision is None:
            open_gate = step
    finished = all(
        step in lab.results and step in decisions and lab.is_current(step)
        for step in STEPS
    )
    return to_json(
        {
            "run_dir": str(lab.store.root),
            "steps": steps,
            "open_gate": open_gate,
            "finished": finished,
            "paused": lab.paused,
            "driven": lab.store.is_driven(),
            "active_time": lab.active_time,
            "time_budget": lab.config.time_budget,
            "config": lab.config.to_dict(),
        }
    )


def step_view(lab: FittingLab, step: str) -> dict[str, Any]:
    """Result, gate and charts of a step."""
    result = lab.results.get(step)
    entry = lab.decisions().get(step)
    out: dict[str, Any] = {
        "step": step,
        "number": STEPS.index(step),  # type: ignore[arg-type]
        "title": TITLES[step],
        "status": step_status(lab.results.get(step)),
        "current": result is not None and lab.is_current(step),
        "result": None if result is None else result.to_dict(),
        "decision": entry,
        "charts": [],
        "tables": [],
    }
    if result is None or result.skipped:
        return to_json(out)
    decision = (entry or {}).get("values", result.proposal)
    if step == "setup":
        charts, tables = _setup(result)
    elif step == "parameters":
        charts, tables = _parameters(result, list(decision.get("parameters", [])))
    elif step == "fit":
        charts, tables = _fit(_with_current(lab, result))
    elif step == "diagnose":
        charts, tables = _diagnose(result)
    elif step == "identifiability":
        charts, tables = _identifiability(result)
    else:
        charts, tables = _uncertainty(lab, result)
    out["charts"] = [c for c in charts if c is not None]
    out["tables"] = [t for t in tables if t is not None]
    return to_json(out)


def _with_current(lab: FittingLab, result: StepResult) -> StepResult:
    """The fit result with the data it was fitted to (for its charts)."""
    setup = lab.results.get("setup")
    if setup is None or "current" not in setup.artifacts:
        return result
    current = cast(Current, setup.artifacts["current"])
    return replace(result, artifacts={**result.artifacts, "current": current})


###############################################################################
# Progress
###############################################################################


def _group(start: str | None) -> str:
    """The fit a start belongs to: its label without trailing numbers.

    `start/3` → `start`, `a/start/0` → `a/start`, `refit-4711-2` → `refit`;
    numbers that are part of a name stay (`hidden1`).
    """
    if start is None:
        return "fit"
    return re.sub(r"([/_-]\d+)+$", "", start) or start


@dataclass
class StepProgress:
    """Progress of one step.

    Attributes
    ----------
    step
        The step
    started
        Wall-clock time the step started
    finished
        Wall-clock time it finished, None while running
    expected
        Planned runtime (low, typical, high) in seconds, if known
    traces
        Best objective over time per fit group: `[[seconds, best], ...]`
    final
        Final objective of every finished start per group (None: failed)
    best_parameters
        Values of the best evaluation so far per group
    n_evaluations
        Evaluations per start

    """

    step: str
    started: float
    finished: float | None = None
    expected: list[float] | None = None
    passed: bool | None = None
    traces: dict[str, list[list[float]]] = field(default_factory=dict)
    final: dict[str, dict[str, float | None]] = field(default_factory=dict)
    best: dict[str, float] = field(default_factory=dict)
    best_parameters: dict[str, dict[str, float]] = field(default_factory=dict)
    n_evaluations: dict[str, int] = field(default_factory=dict)
    version: int = 0

    def feed(self, ev: Mapping[str, Any]) -> None:
        """Add a fit event."""
        start = ev.get("start")
        group = _group(start)
        key = str(start)
        if (n := ev.get("n_evaluations")) is not None:
            self.n_evaluations[key] = max(self.n_evaluations.get(key, 0), int(n))
        kind = ev.get("kind")
        if kind in ("finished", "failed"):
            value = ev.get("value") if kind == "finished" else None
            self.final.setdefault(group, {})[key] = value
        best = ev.get("best")
        if best is None or not math.isfinite(best):
            return
        if best < self.best.get(group, math.inf):
            self.best[group] = float(best)
            trace = self.traces.setdefault(group, [])
            trace.append([float(ev.get("time", time.time())) - self.started, best])
            if len(trace) > _MAX_TRACE:
                del trace[1:-1:2]
            if ev.get("parameters"):
                self.best_parameters[group] = dict(ev["parameters"])
        self.version += 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict, with the waterfall of every group."""
        now = time.time()
        elapsed = (self.finished or now) - self.started
        eta = None
        if self.expected is not None and self.finished is None:
            eta = [
                max(self.expected[0] - elapsed, 0.0),
                max(self.expected[2] - elapsed, 0.0),
            ]
        return to_json(
            {
                "step": self.step,
                "started": self.started,
                "finished": self.finished,
                "elapsed": elapsed,
                "expected": self.expected,
                "eta": eta,
                "passed": self.passed,
                "n_evaluations": sum(self.n_evaluations.values()),
                "n_finished": sum(len(v) for v in self.final.values()),
                "best": self.best,
                "traces": self.traces,
                "waterfalls": {
                    g: sorted(v for v in f.values() if v is not None)
                    for g, f in self.final.items()
                },
                "n_failed": {
                    g: sum(v is None for v in f.values()) for g, f in self.final.items()
                },
                "version": self.version,
            }
        )


class Progress:
    """Live progress of a run, aggregated from its events.

    Feed it the events of `events.jsonl` (all of them, in order); it keeps
    the progress of the latest run of every step and the run's state.
    """

    def __init__(self) -> None:
        self.steps: dict[str, StepProgress] = {}
        self.running: str | None = None
        self.gate: str | None = None
        self.error: dict[str, Any] | None = None
        self.finished = False

    def feed(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Add events."""
        for ev in events:
            step = ev.get("step")
            if ev.get("source") != "lab":
                if isinstance(step, str) and step in self.steps:
                    self.steps[step].feed(ev)
                continue
            kind = ev.get("kind")
            t = float(ev.get("time", time.time()))
            if kind == "step_started" and isinstance(step, str):
                self.steps[step] = StepProgress(
                    step, started=t, expected=ev.get("expected")
                )
                self.running = step
                self.gate = None
                self.error = None
                self.finished = False
            elif kind == "step_finished" and isinstance(step, str):
                if step in self.steps:
                    self.steps[step].finished = t
                    self.steps[step].passed = ev.get("passed")
                if self.running == step:
                    self.running = None
            elif kind == "step_failed":
                self.error = {"step": step, "message": ev.get("message")}
                if self.running == step:
                    self.running = None
            elif kind == "gate_open":
                self.gate = step
            elif kind == "decision" and step == self.gate:
                self.gate = None
            elif kind == "run_finished":
                self.finished = True
                self.gate = None

    def state(self) -> dict[str, Any]:
        """The running step (with ETA), the open gate and errors."""
        running = self.steps.get(self.running) if self.running else None
        return to_json(
            {
                "running": None if running is None else running.to_dict(),
                "gate": self.gate,
                "error": self.error,
                "finished": self.finished,
            }
        )


def live_fit(
    current: Current, parameters: Mapping[str, float]
) -> list[dict[str, Any]] | None:
    """Charts of the data and the model at `parameters`, None if they don't fit."""
    if set(parameters) != set(current.estimates):
        return None
    predictions = simulate_conditions(
        replace(current, estimates=dict(parameters)), dict(parameters)
    )
    return to_json(fit_charts(current, predictions, prefix="live"))
