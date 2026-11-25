# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Types and helpers shared by the steps of the fitting lab."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit._problem import _Problem, setup_problem
from mxlpy.fit.callbacks import JsonlCallback
from mxlpy.fit.residuals import predicted
from mxlpy.simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.fit._gradient import Transform
    from mxlpy.lab.runtime import RuntimeModel
    from mxlpy.lab.store import RunStore

__all__ = [
    "STEPS",
    "Current",
    "LabConfig",
    "LabState",
    "Observable",
    "Option",
    "StepName",
    "StepResult",
    "TITLES",
    "parse_duration",
    "step_status",
]

type Model = KineticModelBuilder | OdeModelBuilder

type StepName = Literal[
    "setup", "parameters", "fit", "diagnose", "identifiability", "uncertainty"
]
STEPS: tuple[StepName, ...] = (
    "setup",
    "parameters",
    "fit",
    "diagnose",
    "identifiability",
    "uncertainty",
)
TITLES: dict[str, str] = {
    "setup": "Setup",
    "parameters": "Parameter selection",
    "fit": "Fit",
    "diagnose": "Missing dynamics",
    "identifiability": "Identifiability",
    "uncertainty": "Uncertainty",
}


###############################################################################
# Settings
###############################################################################


@dataclass(frozen=True)
class Observable:
    """How a data column is computed from the model.

    Parameters
    ----------
    target
        Model output the column measures: a variable, derived value or flux
    scale
        Whether the column is measured in arbitrary units, `scale · target`.
        Adds the parameter `scale_{column}`, a candidate of step 1
    offset
        Whether the column has a background, `target + offset`. Adds the
        parameter `offset_{column}`

    Examples
    --------
        >>> lab.Observable("ERK_p", scale=True)

    """

    target: str
    scale: bool = False
    offset: bool = False


_DURATION = re.compile(r"^\s*([\d.]+)\s*(s|sec|min|m|h|hr|d)?\s*$", re.IGNORECASE)
_UNITS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400}


def parse_duration(value: str | float | None) -> float | None:
    """Seconds of a duration like `90`, `"90s"`, `"30min"` or `"2h"`."""
    if value is None or isinstance(value, int | float):
        return None if value is None else float(value)
    if (m := _DURATION.match(value)) is None:
        msg = f"Can't read the duration {value!r}, use e.g. '90s', '30min' or '2h'"
        raise ValueError(msg)
    return float(m.group(1)) * _UNITS[(m.group(2) or "s").lower()]


_STEP_SETTINGS: dict[str, tuple[str, ...]] = {
    "setup": ("parameters", "observables", "sigma"),
    "parameters": (
        "max_parameters",
        "time_budget",
        "bounds",
        "spread",
        "n_samples",
        "collinearity",
        "rank_rtol",
        "seed",
        # runtime estimates of the later steps
        "n_starts",
        "n_pull_fits",
        "prediction_times",
    ),
    "fit": ("bounds", "spread", "n_starts", "significance", "converged", "seed"),
    "diagnose": (
        "bounds",
        "spread",
        "n_starts",
        "significance",
        "max_hidden",
        "hidden_layers",
        "structure_search",
        "delay_stages",
        "spline_knots",
        "seed",
    ),
    "identifiability": ("bounds", "significance", "n_pull_fits", "profile", "seed"),
    "uncertainty": (
        "bounds",
        "significance",
        "prediction_times",
        "mcmc",
        "mcmc_samples",
        "seed",
    ),
}


@dataclass
class LabConfig:
    """Settings of a lab run (stored in `config.json`).

    Attributes
    ----------
    parameters
        Candidate parameters of step 1, all parameters (and scales and
        offsets of observables) by default
    max_parameters
        Upper limit of the number of estimated parameters
    time_budget
        Seconds the whole run should take, which limits the number of
        parameters and starts; None for no limit
    observables
        How data columns that aren't model outputs are computed
    sigma
        Measurement noise per data set (scalar or per observable), estimated
        from the data where missing (and not given by a `fit.Condition`)
    bounds
        Bounds of estimated values
    spread
        Factor around the nominal values that samples and starts are drawn
        from (log-uniformly)
    n_samples
        Parameter samples of the sensitivity analysis of step 1
    collinearity
        Minimal sine of the angle between a parameter's sensitivity vector and
        those of the parameters already chosen in step 1
    rank_rtol
        Singular values of the scaled sensitivities below this fraction of the
        largest don't count towards the rank estimate
    n_starts
        Starts of the multi-start fits of step 2
    significance
        Significance level of all tests
    converged
        Objective difference to the best start below which a start counts as
        converged (0.5: Δχ² < 1)
    max_hidden
        Largest number of hidden states tried in step 3
    hidden_layers
        Hidden layer widths of the neural network of the hidden states
    structure_search
        Whether step 3 compares structural variants
    delay_stages
        Intermediate states of the generic delay variant
    spline_knots
        Knots of the generic unknown-input variant
    n_pull_fits
        Fits per round of the pulling test of step 4
    profile
        Parameters profiled in step 4: `all` or only `flagged` ones
    prediction_times
        Time points of the prediction bands of step 5
    mcmc
        Whether step 5 samples the posterior (single data set only)
    mcmc_samples
        Posterior samples per chain
    seed
        Seed of all random numbers
    parallel
        Whether to run the analyses in parallel processes
    max_workers
        Maximal number of worker processes

    """

    parameters: list[str] | None = None
    max_parameters: int | None = None
    time_budget: float | None = None
    observables: dict[str, Observable] = field(default_factory=dict)
    sigma: dict[str, Any] = field(default_factory=dict)
    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    spread: float = 10.0
    n_samples: int = 20
    collinearity: float = 0.1
    rank_rtol: float = 1e-3
    n_starts: int = 10
    significance: float = 0.05
    converged: float = 0.5
    max_hidden: int = 3
    hidden_layers: tuple[int, ...] = (4,)
    structure_search: bool = True
    delay_stages: int = 2
    spline_knots: int = 6
    n_pull_fits: int = 5
    profile: Literal["all", "flagged"] = "all"
    prediction_times: int = 20
    mcmc: bool = False
    mcmc_samples: int = 1000
    seed: int = 0
    parallel: bool = True
    max_workers: int | None = None

    def settings_of(self, step: str) -> dict[str, Any]:
        """The settings a step's result depends on (for its input hash)."""
        values = self.to_dict()
        return {k: values[k] for k in _STEP_SETTINGS[step]}

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        out = asdict(self)
        out["observables"] = {k: asdict(v) for k, v in self.observables.items()}
        out["hidden_layers"] = list(self.hidden_layers)
        out["bounds"] = {k: list(v) for k, v in self.bounds.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabConfig:
        """Restore a config written by `to_dict`."""
        values = dict(data)
        values["observables"] = {
            k: Observable(**v) for k, v in values.get("observables", {}).items()
        }
        values["hidden_layers"] = tuple(values.get("hidden_layers", (4,)))
        values["bounds"] = {k: tuple(v) for k, v in values.get("bounds", {}).items()}
        return cls(**values)


###############################################################################
# Results and state
###############################################################################

type OptionKind = Literal["subset", "choice", "flag", "mapping"]


@dataclass(frozen=True)
class Option:
    """What a key of a step's proposal can be set to (see `FittingLab.decide`).

    Steps declare their options in `StepResult.options`; `decide` checks
    values against them and the web UI builds its gate forms from them.

    Parameters
    ----------
    kind
        `subset` (a list of `choices`), `choice` (one of `choices`), `flag`
        (a bool) or `mapping` (a JSON object, checked by the step)
    choices
        Allowed values of `subset` and `choice`
    description
        What the key decides

    Examples
    --------
        >>> lab.Option("subset", ("k1", "k2", "Km"), "Estimated parameters")

    """

    kind: OptionKind
    choices: tuple[Any, ...] = ()
    description: str = ""

    def check(self, name: str, value: Any) -> None:
        """Raise if `value` isn't allowed."""
        if self.kind == "subset":
            if isinstance(value, str) or not isinstance(value, list | tuple):
                msg = f"{name} takes a list of {list(self.choices)}"
                raise TypeError(msg)
            if unknown := [v for v in value if v not in self.choices]:
                msg = f"{name}: {unknown} not in {list(self.choices)}"
                raise ValueError(msg)
            if len(set(value)) != len(value):
                msg = f"{name}: duplicate values in {list(value)}"
                raise ValueError(msg)
        elif self.kind == "choice":
            if value not in self.choices:
                msg = f"{name} is one of {list(self.choices)}, not {value!r}"
                raise ValueError(msg)
        elif self.kind == "flag":
            if not isinstance(value, bool):
                msg = f"{name} takes true or false, not {value!r}"
                raise TypeError(msg)
        elif not isinstance(value, dict):
            msg = f"{name} takes a mapping, not {value!r}"
            raise TypeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        return {
            "kind": self.kind,
            "choices": list(self.choices),
            "description": self.description,
        }


@dataclass
class StepResult:
    """Result of one step of the lab.

    Attributes
    ----------
    step
        Name of the step
    passed
        Whether the step's checks passed (None if it has none)
    summary
        Metrics and tables of the step (JSON-compatible, shown in the
        report and the web UI)
    proposal
        The proposed decision of the step's gate, see `FittingLab.decide`
    options
        What each key of the proposal can be set to
    messages
        Findings in words
    artifacts
        Python results (fits, profiles, models); pickled, not in JSON
    input_hash
        Hash of everything the step depends on
    duration
        Seconds the step took
    skipped
        Whether the step wasn't needed (e.g. diagnosing a model that fits)

    """

    step: StepName
    passed: bool | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Option] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    input_hash: str = ""
    duration: float = 0.0
    skipped: bool = False

    def __repr__(self) -> str:
        """Short representation."""
        lines = [
            f"StepResult({self.step!r}, {step_status(self)}, {self.duration:.1f} s)"
        ]
        lines.extend(f"  - {m}" for m in self.messages)
        if self.proposal:
            lines.append(f"  proposal: {self.proposal}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible part (without artifacts)."""
        return {
            "step": self.step,
            "passed": self.passed,
            "summary": self.summary,
            "proposal": self.proposal,
            "options": {k: v.to_dict() for k, v in self.options.items()},
            "messages": self.messages,
            "input_hash": self.input_hash,
            "duration": self.duration,
            "skipped": self.skipped,
        }


def step_status(result: StepResult | None) -> str:
    """`pending` (no result), `skipped`, `passed`, `failed` or `done` (no checks)."""
    if result is None:
        return "pending"
    if result.skipped:
        return "skipped"
    return {True: "passed", False: "failed", None: "done"}[result.passed]


@dataclass
class Current:
    """The model, data and estimates handed from one step to the next.

    Attributes
    ----------
    model
        The model (with observables)
    conditions
        The data sets, with their noise
    estimates
        Estimated names and their current values
    runner
        How the model is simulated
    fixed
        Values fixed after the identifiability analysis
    linear
        Estimates that are fitted linearly and without default bounds
        (offsets, which can be negative)

    """

    model: Model
    conditions: dict[str, fit.Condition]
    estimates: dict[str, float]
    runner: Any
    fixed: dict[str, float] = field(default_factory=dict)
    linear: set[str] = field(default_factory=set)

    def copy(self) -> Current:
        """A copy whose model can be modified."""
        return replace(
            self,
            model=deepcopy(self.model),
            conditions=dict(self.conditions),
            estimates=dict(self.estimates),
            fixed=dict(self.fixed),
            linear=set(self.linear),
        )

    def transform(self, names: list[str] | None = None) -> Transform:
        """Log transform for positive values, linear otherwise."""
        names = list(self.estimates) if names is None else names
        return {
            k: "log"
            if self.estimates.get(k, 0.0) > 0 and k not in self.linear
            else "linear"
            for k in names
        }

    def bounds(
        self, config: LabConfig, names: list[str] | None = None
    ) -> dict[str, Any]:
        """Bounds of the estimates: the config's, unbounded for `linear` ones."""
        names = list(self.estimates) if names is None else names
        out: dict[str, Any] = {k: (None, None) for k in names if k in self.linear}
        return out | {k: v for k, v in config.bounds.items() if k in names}

    def problem(
        self,
        config: LabConfig,
        names: list[str] | None = None,
        *,
        callback: Any = None,
        minimizer: Any = None,
    ) -> _Problem:
        """The fit of `names` (all estimates by default) to all data sets."""
        names = list(self.estimates) if names is None else names
        start = {k: self.estimates[k] for k in names}
        return setup_problem(
            self.model,
            self.conditions,
            start,
            loss_fn=None,
            minimizer=fit.LeastSquaresMinimizer() if minimizer is None else minimizer,
            y0=None,
            protocol=None,
            runner=self.runner,
            bounds=self.bounds(config, names),
            prior=None,
            penalty=None,
            transform=self.transform(names),
            postprocess_fn=None,
            standard_scale=False,
            callback=callback,
        )

    @property
    def n_data(self) -> int:
        """Number of finite data points."""
        return sum(
            int(np.isfinite(c.data.to_numpy(dtype=float)).sum())
            for c in self.conditions.values()
        )


@dataclass
class LabState:
    """Everything a step function gets.

    Attributes
    ----------
    model
        The user's model
    conditions
        The user's data sets
    config
        Settings of the run
    runner
        How the model is simulated, None for the lab's choice
    variants
        Structural variants of step 3 by name, `fn(model) -> model`
    store
        Files of the run
    runtime
        Runtime estimates
    results
        Results of the steps run so far
    decisions
        Accepted decisions of the steps run so far

    """

    model: Model
    conditions: dict[str, fit.Condition]
    config: LabConfig
    runner: Any
    variants: dict[str, Callable[[Any], Any]]
    store: RunStore
    runtime: RuntimeModel
    results: dict[str, StepResult]
    decisions: dict[str, dict[str, Any]]

    def callback(self, step: str) -> JsonlCallback:
        """Progress callback of a step, writing to `events.jsonl`."""
        return JsonlCallback(self.store.events_file, fields={"step": step})

    def current(self, step: StepName) -> Current:
        """The model, data and estimates step `step` starts from."""
        from mxlpy.lab.steps import apply  # noqa: PLC0415 (cycle)

        out: Current | None = None
        for s in STEPS[: STEPS.index(step)]:
            res = self.results.get(s)
            if res is None:
                continue
            out = apply(s, out, res, self.decisions.get(s, res.proposal), self)
        if out is None:
            msg = "Run the setup step first"
            raise RuntimeError(msg)
        return out


###############################################################################
# Helpers
###############################################################################


def simulate_conditions(
    current: Current, values: Mapping[str, float] | None = None
) -> dict[str, pd.DataFrame | None]:
    """Model outputs at the data time points per data set, None if a simulation fails.

    `values` are estimates (default: the current ones). Condition-specific
    (`local`) names are resolved per data set.
    """
    values = dict(current.estimates if values is None else values)
    values |= current.fixed
    out: dict[str, pd.DataFrame | None] = {}
    for name, cond in current.conditions.items():
        model = condition_model(current.model, cond, values)
        data = cast(pd.DataFrame, cond.data)
        times = np.unique(np.asarray(data.index, dtype=float))
        integrator = getattr(current.runner, "integrator", None)
        sim = Simulator(model, integrator=integrator)
        if cond.protocol is not None:
            sim.simulate_protocol_time_course(cond.protocol, times)
        else:
            sim.simulate_time_course(times)
        res = sim.get_result().value
        if isinstance(res, Exception):
            out[name] = None
            continue
        combined = predicted(res, list(data.columns))
        out[name] = combined.reindex(times).loc[:, list(data.columns)]
    return out


def condition_model(
    model: Model, cond: fit.Condition, values: Mapping[str, float]
) -> Model:
    """A copy of `model` with `values` and the settings of a data set."""
    out = deepcopy(model)
    variables = (
        set()
        if isinstance(out, SteadyStateModelBuilder)
        else set(out.get_variable_names())
    )
    addressable = set(out.get_addressable_names())
    local = dict(cond.local or {})
    resolved = {k: values[v] for k, v in local.items() if v in values}
    pars = {k: v for k, v in values.items() if k in addressable} | resolved
    out.update_addressable(pars)
    if cond.parameters:
        out.update_addressable(dict(cond.parameters))
    if cond.y0:
        out.update_variables(dict(cond.y0))
    if init := {k: v for k, v in values.items() if k in variables}:
        out.update_variables(init)
    return out


def residuals_of(
    current: Current, predictions: Mapping[str, pd.DataFrame | None]
) -> dict[str, pd.DataFrame]:
    """Data minus predictions per data set (replicates compared to their time point)."""
    out = {}
    for name, cond in current.conditions.items():
        pred = predictions.get(name)
        if pred is None:
            continue
        data = cast(pd.DataFrame, cond.data)
        aligned = pred.reindex(np.asarray(data.index, dtype=float))
        aligned.index = data.index
        out[name] = data - aligned
    return out


def nrmse(current: Current, residuals: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """RMSE per data set and observable, normalised by the data's range."""
    rows = {}
    for name, res in residuals.items():
        data = cast(pd.DataFrame, current.conditions[name].data)
        for col in res.columns:
            r = res[col].to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            values = data[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            span = float(values.max() - values.min()) if values.size else math.nan
            rmse = math.sqrt(float(np.mean(r**2))) if r.size else math.nan
            rows[(name, col)] = {
                "rmse": rmse,
                "nrmse": rmse / span if span > 0 else math.nan,
            }
    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df.index = pd.MultiIndex.from_tuples(df.index, names=["data", "observable"])
    return df


def start_values(
    estimates: Mapping[str, float],
    n: int,
    spread: float,
    bounds: Mapping[str, tuple[float | None, float | None]],
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """The estimates, then `n - 1` log-uniform samples around them within bounds.

    Values that aren't positive are kept, as they can't be scaled.
    """
    starts = [dict(estimates)]
    log_spread = math.log10(max(spread, 1.0))
    for _ in range(max(n - 1, 0)):
        start = {}
        for k, v in estimates.items():
            if v > 0:
                v = v * 10 ** rng.uniform(-log_spread, log_spread)  # noqa: PLW2901
            lo, hi = bounds.get(k, (None, None))
            if lo is not None:
                v = max(v, lo)  # noqa: PLW2901
            if hi is not None:
                v = min(v, hi)  # noqa: PLW2901
            start[k] = float(v)
        starts.append(start)
    return starts


def safe_name(text: str) -> str:
    """An identifier from arbitrary text (for generated parameter names)."""
    return re.sub(r"\W+", "_", text).strip("_") or "x"
