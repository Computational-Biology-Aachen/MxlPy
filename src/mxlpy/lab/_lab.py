# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""The fitting lab: runs the steps, keeps the run directory and the gates."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from mxlpy.fit.abstract import as_conditions
from mxlpy.lab._core import (
    STEPS,
    LabConfig,
    LabState,
    Observable,
    StepName,
    StepResult,
    parse_duration,
    step_status,
)
from mxlpy.lab.runtime import Estimate, RuntimeModel, fit_evaluations
from mxlpy.lab.steps import RUNNERS
from mxlpy.lab.store import RunLockedError, RunStore
from mxlpy.parallel import _n_workers, content_hash

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from mxlpy import fit
    from mxlpy.lab._core import Model

__all__ = ["FittingLab", "LabEvent"]


def _key(step: StepName | int | str) -> StepName:
    """Step key of a step name, number (0-5) or method name (e.g. `step2_fit`)."""
    if isinstance(step, int) and 0 <= step < len(STEPS):
        return STEPS[step]
    if isinstance(step, str):
        if step in STEPS:
            return step  # type: ignore[return-value]
        methods = {
            "step0_setup": "setup",
            "step1_select_parameters": "parameters",
            "step2_fit": "fit",
            "step3_diagnose": "diagnose",
            "step4_identifiability": "identifiability",
            "step5_uncertainty": "uncertainty",
        }
        if step in methods:
            return methods[step]  # type: ignore[return-value]
    msg = f"Unknown step {step!r}, choose a number 0-5 or one of {STEPS}"
    raise ValueError(msg)


_LOGGER = logging.getLogger(__name__)
_POLL = 0.5


@dataclass(frozen=True)
class LabEvent:
    """Progress of a running step, for `FittingLab.on_event`.

    Attributes
    ----------
    step
        The running step
    kind
        `started`, `progress` (a fit event), `finished` or `cached` (an
        up-to-date result was loaded)
    best_loss
        Lowest objective of the step so far, None before the first
    n_evaluations
        Evaluations of the step so far
    elapsed
        Seconds since the step started
    eta
        Estimated remaining seconds (low, high), None if unknown
    event
        The raw event (a `FitEvent` as dict for `progress`)

    """

    step: str
    kind: str
    best_loss: float | None = None
    n_evaluations: int = 0
    elapsed: float = 0.0
    eta: tuple[float, float] | None = None
    event: dict[str, Any] = field(default_factory=dict)


class _Monitor:
    """Tails `events.jsonl` while a step runs and tells the listeners.

    Every finished fit refines the runtime model with its cost per
    evaluation (including derivatives); the ETA is the step's planned runtime
    rescaled by how far the observed cost is off the planned one.
    """

    def __init__(
        self,
        lab: FittingLab,
        step: str,
        expected: Estimate | None,
        n_parameters: int,
    ) -> None:
        self.lab = lab
        self.step = step
        self.expected = expected
        self.n_parameters = n_parameters
        self.planned = lab.runtime.per_evaluation(n_parameters)[0]
        self.best: float | None = None
        self._evaluations: dict[str | None, int] = {}
        self.t0 = time.perf_counter()
        self._stop = threading.Event()
        self._offset = (
            lab.store.events_file.stat().st_size
            if lab.store.events_file.exists()
            else 0
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @property
    def n(self) -> int:
        """Evaluations of all fits of the step so far."""
        return sum(self._evaluations.values())

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._poll()

    def eta(self) -> tuple[float, float] | None:
        """Remaining seconds (optimistic, pessimistic), None without a plan."""
        if self.expected is None:
            return None
        current = self.lab.runtime.per_evaluation(self.n_parameters)[0]
        ratio = current / self.planned if self.planned > 0 and current > 0 else 1.0
        if not math.isfinite(ratio):
            ratio = 1.0
        elapsed = time.perf_counter() - self.t0
        return (
            max(self.expected.low * ratio - elapsed, 0.0),
            max(self.expected.high * ratio - elapsed, 0.0),
        )

    def _poll(self) -> None:
        events, self._offset = self.lab.store.read_events(self._offset)
        for ev in events:
            if ev.get("step") != self.step or ev.get("source") == "lab":
                continue
            start = ev.get("start")
            if (n := ev.get("n_evaluations")) is not None:
                self._evaluations[start] = max(self._evaluations.get(start, 0), int(n))
            if ev.get("kind") == "finished" and n and (d := ev.get("duration")):
                self.lab.runtime.observe(float(d) / int(n), self.n_parameters)
            if (b := ev.get("best")) is not None and (
                self.best is None or b < self.best
            ):
                self.best = float(b)
            self.lab._notify(  # noqa: SLF001
                LabEvent(
                    step=self.step,
                    kind="progress",
                    best_loss=self.best,
                    n_evaluations=self.n,
                    elapsed=time.perf_counter() - self.t0,
                    eta=self.eta(),
                    event=ev,
                )
            )

    def _loop(self) -> None:
        while not self._stop.wait(_POLL):
            try:
                self._poll()
            except Exception:
                _LOGGER.debug("Reading progress events failed", exc_info=True)


class FittingLab:
    """Guided, checkpointed fitting of a model to one or several data sets.

    Six steps, each ending at a review gate with a result and a proposal:

    0. `step0_setup`: map data columns to model outputs (`observables`),
       estimate the measurement noise, pick an integrator, time the model
    1. `step1_select_parameters`: rank the candidate parameters by their
       sensitivities, drop collinear ones, limit their number by the
       sensitivity rank and the time budget
    2. `step2_fit`: multi-start fits, χ² goodness of fit, residual
       whiteness, multi-start convergence; with several data sets
       condition-specific parameters if every data set fits on its own
    3. `step3_diagnose` (if step 2 fails): hidden states until the residuals
       are white, then structural variants of the flagged variables
    4. `step4_identifiability`: Fisher information, pulling test and
       profile likelihoods; proposes parameters to fix
    5. `step5_uncertainty`: prediction bands of the observables, optional
       posterior sampling

    Calling a step runs the steps before it as needed and accepts their
    proposals, unless `decide` recorded another decision. Results are
    cached in `run_dir` with a hash of everything they depend on, so an
    interrupted or resumed run only recomputes what changed.

    Examples
    --------
        >>> run = lab.FittingLab(model, {"wt": df_wt, "ko": df_ko}, "runs/glyco",
        ...                      time_budget="2h")
        >>> run.step0_setup()
        >>> run.step1_select_parameters()
        >>> run.decide("parameters", parameters=["k1", "k2", "Km"])
        >>> run.step2_fit()
        >>> lab.FittingLab.resume("runs/glyco").run(auto=True)
        >>> run.report()

    Parameters
    ----------
    model
        The model to fit (kinetic or ODE builder)
    data
        Time courses: one DataFrame, a `fit.Condition`, or a mapping of names
        to either
    run_dir
        Directory of the run (created if needed)
    parameters
        Candidate parameters of step 1, all parameters by default
    max_parameters
        Upper limit of the number of estimated parameters
    time_budget
        Planned duration of the whole run, e.g. `"2h"`, `"30min"` or seconds
    observables
        How data columns that aren't model outputs are computed:
        `{column: Observable(...)}` or `{column: target}`
    sigma
        Measurement noise per data set, estimated where missing
    bounds
        Bounds of estimated values
    variants
        Structural alternatives for step 3: `{name: fn(model) -> model}`
    runner
        How the model is simulated, chosen in step 0 by default
    **settings
        Further `LabConfig` settings, e.g. `n_starts=20`, `mcmc=True`

    """

    def __init__(
        self,
        model: Model,
        data: fit.Data,
        run_dir: str | os.PathLike[str],
        *,
        parameters: list[str] | None = None,
        max_parameters: int | None = None,
        time_budget: str | float | None = None,
        observables: Mapping[str, Observable | str] | None = None,
        sigma: Mapping[str, Any] | None = None,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        variants: Mapping[str, Callable[[Any], Any]] | None = None,
        runner: Any | None = None,
        **settings: Any,
    ) -> None:
        conditions = as_conditions(data)
        if sigma and (unknown := sorted(set(sigma) - set(conditions))):
            msg = f"sigma of unknown data sets {unknown}"
            raise KeyError(msg)
        config = LabConfig(
            parameters=None if parameters is None else list(parameters),
            max_parameters=max_parameters,
            time_budget=parse_duration(time_budget),
            observables={
                k: Observable(v) if isinstance(v, str) else v
                for k, v in (observables or {}).items()
            },
            sigma=dict(sigma or {}),
            bounds={k: tuple(v) for k, v in (bounds or {}).items()},  # type: ignore[misc]
            **settings,
        )
        store = RunStore(run_dir)
        store.root.mkdir(parents=True, exist_ok=True)
        store.write_json(store.config_file, config.to_dict())
        store.write_pickle(
            store.inputs_file,
            {
                "model": model,
                "conditions": conditions,
                "runner": runner,
                "variants": dict(variants or {}),
            },
        )
        self._setup(
            store,
            model=model,
            conditions=conditions,
            config=config,
            runner=runner,
            variants=dict(variants or {}),
        )

    @classmethod
    def resume(cls, run_dir: str | os.PathLike[str]) -> FittingLab:
        """Continue a run from its directory."""
        store = RunStore(run_dir)
        if not store.inputs_file.exists():
            msg = f"No lab run in {store.root}"
            raise FileNotFoundError(msg)
        inputs = store.read_pickle(store.inputs_file)
        config = LabConfig.from_dict(store.read_json(store.config_file, {}))
        lab = cls.__new__(cls)
        lab._setup(  # noqa: SLF001
            store,
            model=inputs["model"],
            conditions=inputs["conditions"],
            config=config,
            runner=inputs["runner"],
            variants=inputs["variants"],
        )
        return lab

    def _setup(
        self,
        store: RunStore,
        *,
        model: Model,
        conditions: dict[str, fit.Condition],
        config: LabConfig,
        runner: Any,
        variants: dict[str, Callable[[Any], Any]],
    ) -> None:
        self.store = store
        self.model = model
        self.conditions = conditions
        self.config = config
        self.runner = runner
        self.variants = variants
        self._listeners: list[Callable[[LabEvent], None]] = []
        self._driver = threading.RLock()
        self._driving_depth = 0
        self._lock_stack: contextlib.ExitStack | None = None
        self._mtimes: dict[str, float] = {}
        state = store.read_json(store.runtime_file, {}) or {}
        self.runtime = RuntimeModel.from_dict(state.get("model", {}))
        self.runtime.workers = (
            _n_workers(config.max_workers, 10**6) if config.parallel else 1
        )
        self._save_runtime()
        self._inputs_hash = content_hash(
            model,
            conditions,
            repr(runner),
            sorted(variants),
            [content_hash(v) for _, v in sorted(variants.items())],
        )
        self.results: dict[str, StepResult] = {}
        self._refresh()

    def _refresh(self) -> list[str]:
        """Load step results another process wrote since; returns their steps."""
        changed = []
        for step in STEPS:
            file = self.store.step_dir(step) / "result.pkl"
            try:
                mtime = file.stat().st_mtime
            except FileNotFoundError:
                continue
            if self._mtimes.get(step) == mtime:
                continue
            try:
                self.results[step] = self.store.read_pickle(file)
            except Exception:  # noqa: BLE001
                _LOGGER.warning("Couldn't load the result of %s, rerunning", step)
                continue
            self._mtimes[step] = mtime
            changed.append(step)
        return changed

    # Events ##################################################################

    def on_event(self, fn: Callable[[LabEvent], None]) -> None:
        """Call `fn` with every `LabEvent` (progress of running steps)."""
        self._listeners.append(fn)

    def _notify(self, event: LabEvent) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:
                _LOGGER.warning("Event listener failed", exc_info=True)

    def _save_runtime(self) -> None:
        self.store.write_json(
            self.store.runtime_file,
            {"model": self.runtime.to_dict()},
        )

    # Decisions ###############################################################

    def decisions(self) -> dict[str, dict[str, Any]]:
        """Valid decisions of all gates: values, source (user, auto or implicit), basis."""
        return {k: v for k, v in self.store.decisions().items() if self._valid(k, v)}

    def _valid(self, step: str, entry: dict[str, Any]) -> bool:
        """Whether a decision applies to the step's current result.

        Accepted proposals (implicit, auto) only hold for the result they were
        made on; a user's decision holds until changed.
        """
        if entry.get("source") == "user":
            return True
        result = self.results.get(step)
        return result is not None and entry.get("basis") == result.input_hash

    def decision(self, step: StepName | int | str) -> dict[str, Any] | None:
        """The decided values of a step's gate, None if undecided."""
        entry = self.decisions().get(_key(step))
        return None if entry is None else dict(entry.get("values", {}))

    def decide(self, step: StepName | int | str, **values: Any) -> None:
        """Override a step's proposal (keys as in its `proposal`).

        Steps after it become outdated and are recomputed when called.

        Examples
        --------
            >>> run.decide("parameters", parameters=["k1", "Km"])
            >>> run.decide("identifiability", fix=[])

        """
        step = _key(step)
        result = self.results.get(step)
        proposal = {} if result is None else result.proposal
        unknown = sorted(set(values) - set(proposal)) if proposal else []
        if unknown:
            msg = f"Step {step!r} decides {sorted(proposal)}, not {unknown}"
            raise ValueError(msg)
        options = {} if result is None else result.options
        for k, v in values.items():
            if k in options:
                options[k].check(k, v)
        self._record(step, {**proposal, **values}, "user")

    def reopen(self, step: StepName | int | str) -> None:
        """Forget the decision of a step's gate, so `run` stops there again."""
        step = _key(step)
        self.store.drop_decision(step)
        self.store.append_event(
            {"source": "lab", "kind": "decision", "step": step, "by": None}
        )

    def _record(self, step: str, values: dict[str, Any], source: str) -> None:
        result = self.results.get(step)
        self.store.write_decision(
            step,
            {
                "values": values,
                "source": source,
                "basis": None if result is None else result.input_hash,
            },
        )
        self.store.append_event(
            {"source": "lab", "kind": "decision", "step": step, "by": source}
        )

    # Running #################################################################

    def _state(self) -> LabState:
        decisions = {k: dict(v.get("values", {})) for k, v in self.decisions().items()}
        return LabState(
            model=self.model,
            conditions=self.conditions,
            config=self.config,
            runner=self.runner,
            variants=self.variants,
            store=self.store,
            runtime=self.runtime,
            results=dict(self.results),
            decisions=decisions,
        )

    def _hash(self, step: StepName) -> str:
        decisions = self.decisions()
        upstream = [
            (
                s,
                self.results[s].input_hash if s in self.results else None,
                decisions.get(s, {}).get("values"),
            )
            for s in STEPS[: STEPS.index(step)]
        ]
        return content_hash(
            step, self._inputs_hash, self.config.settings_of(step), upstream
        )

    def is_current(self, step: StepName | int | str) -> bool:
        """Whether the stored result of a step is up to date."""
        step = _key(step)
        res = self.results.get(step)
        return res is not None and res.input_hash == self._hash(step)

    def _needs_diagnosis(self) -> bool:
        fit_result = self.results.get("fit")
        decision = self.decision("fit")
        if decision is not None and "next" in decision:
            return decision["next"] == "diagnose"
        return fit_result is not None and fit_result.passed is False

    def _prepare(self, step: StepName, source: str) -> None:
        """Bring the steps before `step` up to date and decide their gates."""
        for s in STEPS[: STEPS.index(step)]:
            if s == "diagnose" and not self._needs_diagnosis():
                # An explicitly run, up-to-date diagnosis is kept
                if not self.is_current(s):
                    self._skip(s)
            elif not self.is_current(s):
                self._execute(s)
            if self.decision(s) is None:
                self._record(s, dict(self.results[s].proposal), source)

    def _skip(self, step: StepName) -> StepResult:
        result = StepResult(step=step, skipped=True, messages=["Not needed"])
        result.input_hash = self._hash(step)
        self._save(result)
        self.store.append_event({"source": "lab", "kind": "step_skipped", "step": step})
        return result

    def _save(self, result: StepResult) -> None:
        self.results[result.step] = result
        directory = self.store.step_dir(result.step)
        self.store.write_pickle(directory / "result.pkl", result)
        self.store.write_json(directory / "result.json", result.to_dict())
        self._mtimes[result.step] = (directory / "result.pkl").stat().st_mtime

    def _expected(self, step: StepName) -> tuple[Estimate | None, int]:
        """Expected runtime of a step and the number of estimated parameters."""
        config, runtime = self.config, self.runtime
        params = self.results.get("parameters")
        if step == "parameters":
            n = (
                len(self.results["setup"].summary.get("candidates", []))
                if "setup" in self.results
                else 1
            )
            samples = config.n_samples + 1
            return runtime.estimate(
                (samples / 2, samples / 2, samples), n, parallel_tasks=samples
            ), n
        if params is None:
            return None, len(config.parameters or [])
        decision = self.decision("parameters")
        chosen = (decision or params.proposal).get("parameters", [])
        p = len(chosen)
        if step == "diagnose":
            # hidden-state trainings (~30 weights) and the variants' fits
            lo, typ, hi = fit_evaluations(p + 30)
            hidden = runtime.estimate(
                (lo, typ * config.max_hidden, hi * config.max_hidden), p + 30
            )
            n_fits = (3 + len(self.variants)) * max(config.n_starts // 2, 1)
            lo, typ, hi = fit_evaluations(p + 2)
            variants = runtime.estimate(
                (n_fits * lo, n_fits * typ, n_fits * hi), p + 2, parallel_tasks=n_fits
            )
            return hidden + variants, p
        plan = params.summary.get("runtime", {}).get(str(p), {})
        if step in plan:
            r = plan[step]
            return Estimate(r["low"], r["typical"], r["high"]), p
        return None, p

    def _execute(self, step: StepName) -> StepResult:
        input_hash = self._hash(step)
        expected, n_parameters = self._expected(step)
        self.store.append_event(
            {
                "source": "lab",
                "kind": "step_started",
                "step": step,
                "expected": None
                if expected is None
                else [expected.low, expected.typical, expected.high],
            }
        )
        self._notify(
            LabEvent(
                step=step,
                kind="started",
                eta=None if expected is None else (expected.low, expected.high),
            )
        )
        t0 = time.perf_counter()
        with _Monitor(self, step, expected, n_parameters) as monitor:
            try:
                result = RUNNERS[step](self._state())
            except Exception as e:
                self.store.append_event(
                    {
                        "source": "lab",
                        "kind": "step_failed",
                        "step": step,
                        "message": f"{type(e).__name__}: {e}",
                    }
                )
                raise
        result.input_hash = input_hash
        result.duration = time.perf_counter() - t0
        self._save(result)
        self._save_runtime()
        self.store.append_event(
            {
                "source": "lab",
                "kind": "step_finished",
                "step": step,
                "passed": result.passed,
                "duration": result.duration,
            }
        )
        self._notify(
            LabEvent(
                step=step,
                kind="finished",
                best_loss=monitor.best,
                n_evaluations=monitor.n,
                elapsed=result.duration,
            )
        )
        return result

    @contextlib.contextmanager
    def _driving(self) -> Iterator[None]:
        """Run steps: only one thread of one process at a time (`RunLockedError`)."""
        if not self._driver.acquire(blocking=False):
            msg = "Another thread is running the steps of this lab"
            raise RunLockedError(msg)
        try:
            if self._driving_depth == 0:
                stack = contextlib.ExitStack()
                stack.enter_context(self.store.driving())
                self._lock_stack = stack
            self._driving_depth += 1
            try:
                yield
            finally:
                self._driving_depth -= 1
                if self._driving_depth == 0 and self._lock_stack is not None:
                    self._lock_stack.close()
                    self._lock_stack = None
        finally:
            self._driver.release()

    def _step(self, step: StepName, *, source: str = "implicit") -> StepResult:
        with self._driving():
            self._prepare(step, source)
            if self.is_current(step) and not self.results[step].skipped:
                self._notify(LabEvent(step=step, kind="cached"))
                return self.results[step]
            return self._execute(step)

    def step0_setup(self) -> StepResult:
        """Observables, noise, integrator and timing."""
        return self._step("setup")

    def step1_select_parameters(self) -> StepResult:
        """Rank parameters by sensitivity and propose which to estimate."""
        return self._step("parameters")

    def step2_fit(self) -> StepResult:
        """Multi-start fits and their checks."""
        return self._step("fit")

    def step3_diagnose(self) -> StepResult:
        """Hidden states and structural variants (usually only if step 2 failed)."""
        return self._step("diagnose")

    def step4_identifiability(self) -> StepResult:
        """Fisher information, pulling test and profile likelihoods."""
        return self._step("identifiability")

    def step5_uncertainty(self) -> StepResult:
        """Prediction bands and the optional posterior."""
        return self._step("uncertainty")

    @property
    def paused(self) -> bool:
        """Whether `run` pauses before its next step."""
        return self.store.paused

    def pause(self) -> None:
        """Pause `run` once the running step finished (from any process)."""
        self.store.set_paused(True)
        self.store.append_event({"source": "lab", "kind": "paused"})

    def unpause(self) -> None:
        """Let a paused `run` go on."""
        self.store.set_paused(False)
        self.store.append_event({"source": "lab", "kind": "unpaused"})

    def _advance(
        self, *, auto: bool, stop: threading.Event | None = None
    ) -> tuple[StepResult | None, str | None]:
        """Run the steps up to the first open gate or pause.

        Returns the last result and why it stopped: the step of an open gate,
        `"paused"`, `"stopped"` (by `stop`), or None at the end of the pipeline.
        """
        last: StepResult | None = None
        for step in STEPS:
            if step == "diagnose" and not self._needs_diagnosis():
                if not self.is_current(step):
                    self._skip(step)
                if self.decision(step) is None:
                    self._record(step, {}, "auto" if auto else "implicit")
                continue
            if not self.is_current(step):
                if stop is not None and stop.is_set():
                    return last, "stopped"
                if self.paused:
                    return last, "paused"
            last = self._step(step, source="auto")
            if self.decision(step) is None:
                if not auto:
                    return last, step
                self._record(step, dict(last.proposal), "auto")
        return last, None

    def run(
        self,
        *,
        auto: bool = False,
        wait: bool = False,
        stop: threading.Event | None = None,
    ) -> StepResult | None:
        """Run the steps in order.

        With `auto`, every proposal is accepted and the run goes through to
        the end. Otherwise it stops at the first gate without a decision and
        returns its result; call `decide` (or a later step, accepting the
        proposal) to go on. With `wait`, it waits there instead until the
        gate is decided, e.g. from the web UI or another process, and goes
        on to the end. A `pause` (from any process) stops it before its next
        step; with `wait` it goes on after `unpause`.

        Parameters
        ----------
        auto
            Accept every proposal
        wait
            Wait at open gates and pauses instead of returning
        stop
            Event that ends the run before its next step or while waiting

        Raises
        ------
        RunLockedError
            If another process or thread is running the steps

        """
        with self._driving():
            announced: str | None = None
            while True:
                # Taken before advancing: decisions made while it stops at a
                # gate count as changes
                snapshot = (self.store.decisions(), self.paused)
                last, stopped_at = self._advance(auto=auto, stop=stop)
                if stopped_at is None:
                    self.store.append_event({"source": "lab", "kind": "run_finished"})
                    return last
                if stopped_at == "stopped":
                    return last
                if stopped_at == "paused" and not wait:
                    warnings.warn(
                        "The run is paused, call `unpause()` to go on",
                        stacklevel=2,
                    )
                    return last
                if stopped_at not in ("paused", announced):
                    self.store.append_event(
                        {"source": "lab", "kind": "gate_open", "step": stopped_at}
                    )
                    announced = stopped_at
                if not wait:
                    return last
                while snapshot == (self.store.decisions(), self.paused):
                    if stop is not None and stop.wait(_POLL):
                        return last
                    if stop is None:
                        time.sleep(_POLL)

    def __getitem__(self, step: StepName | int | str) -> StepResult:
        """The stored result of a step (by name or number)."""
        return self.results[_key(step)]

    @property
    def active_time(self) -> float:
        """Seconds the current results took to compute (counts against the budget)."""
        return sum(r.duration for r in list(self.results.values()))

    def report(self, *, html: bool = True) -> Path:
        """Write `report.html` (and always `report.json`), returns the HTML path."""
        from mxlpy.lab.report import write_report  # noqa: PLC0415 (matplotlib)

        return write_report(self, html=html)

    def __repr__(self) -> str:
        """Steps and their status."""
        lines = [f"FittingLab({str(self.store.root)!r})"]
        decisions = self.store.decisions()
        for step in STEPS:
            res = self.results.get(step)
            if res is None:
                status = "-"
            else:
                status = step_status(res)
                if not res.skipped and not self.is_current(step):
                    status += " (outdated)"
            gate = decisions.get(step, {}).get("source", "open")
            lines.append(f"  {step:16s} {status:20s} gate: {gate}")
        return "\n".join(lines)
