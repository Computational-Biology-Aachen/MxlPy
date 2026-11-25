# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 1: rank parameters by their sensitivities and drop collinear ones."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from mxlpy.lab._core import Current, LabState, StepResult, start_values
from mxlpy.lab.runtime import Estimate, RuntimeModel, fit_evaluations
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from mxlpy.fit._problem import _Problem
    from mxlpy.lab._core import LabConfig
    from mxlpy.types import Array

__all__ = ["apply", "estimate_runtime", "run"]

# Below this fraction of the largest mean sensitivity a parameter has no effect
_INSENSITIVE = 1e-6


def _scaled_jacobian(task: tuple[_Problem, dict[str, float]]) -> Array | None:
    """Noise-weighted residual sensitivities w.r.t. log values (where positive)."""
    problem, values = task
    try:
        jac, _ = problem.loss_jacobian(values)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(jac).all():
        return None
    scale = np.array([v if v > 0 else 1.0 for v in values.values()])
    return jac * scale[None, :]


def _sine_to_span(column: Array, basis: Array) -> float:
    """Sine of the angle between `column` and the span of `basis`' columns."""
    norm = float(np.linalg.norm(column))
    if norm == 0:
        return 0.0
    if basis.shape[1] == 0:
        return 1.0
    coef, *_ = np.linalg.lstsq(basis, column, rcond=None)
    return float(np.linalg.norm(column - basis @ coef)) / norm


def estimate_runtime(
    runtime: RuntimeModel, config: LabConfig, n_parameters: int, n_bands: int
) -> dict[str, Estimate]:
    """Runtime of steps 2, 4 and 5 with `n_parameters` estimated."""
    lo, typ, hi = fit_evaluations(n_parameters)
    fit_step = runtime.estimate(
        (config.n_starts * lo, config.n_starts * typ, config.n_starts * hi),
        n_parameters,
        parallel_tasks=config.n_starts,
    )
    # Refits along profiles start close to the optimum: a few evaluations each
    steps = 2 * n_parameters * 15
    identifiability = runtime.estimate(
        (steps * 2, steps * 4, steps * 10),
        n_parameters,
        parallel_tasks=2 * n_parameters,
    ) + runtime.estimate(
        (
            config.n_pull_fits * lo,
            config.n_pull_fits * typ,
            config.n_pull_fits * hi * n_parameters,
        ),
        n_parameters,
        parallel_tasks=config.n_pull_fits,
    )
    # Both sides of every time point of every band, ~10 refits each
    sides = 2 * config.prediction_times * n_bands
    uncertainty = runtime.estimate(
        (sides * 20, sides * 50, sides * 150), n_parameters, parallel_tasks=sides
    )
    return {
        "fit": fit_step,
        "identifiability": identifiability,
        "uncertainty": uncertainty,
    }


def run(state: LabState) -> StepResult:
    """Rank the candidates and propose the parameters to estimate."""
    config = state.config
    current = state.current("parameters")
    names = list(current.estimates)
    problem = current.problem(config, names)
    rng = np.random.default_rng(config.seed)
    samples = start_values(
        current.estimates, config.n_samples + 1, config.spread, config.bounds, rng
    )
    tasks = [(i, (problem, s)) for i, s in enumerate(samples)]
    parallel = (
        config.parallel and state.runtime.sensitivity * len(samples) > 5.0  # noqa: PLR2004
    )
    results = dict(
        parallelise(
            _scaled_jacobian,
            tasks,
            parallel=parallel,
            max_workers=config.max_workers,
            disable_tqdm=True,
        )
    )
    jacobians = [j for i in range(len(samples)) if (j := results.get(i)) is not None]
    if not jacobians:
        msg = "The sensitivities failed at every sample"
        raise ValueError(msg)
    nominal = results.get(0)
    n_rows = jacobians[0].shape[0]
    # RMS sensitivity per parameter and sample, averaged over samples
    importance = np.mean(
        [np.linalg.norm(j, axis=0) / math.sqrt(max(n_rows, 1)) for j in jacobians],
        axis=0,
    )
    order = list(np.argsort(-importance))

    chosen: list[int] = []
    status: dict[str, str] = {}
    sines: dict[str, float] = {}
    top = float(importance.max()) if importance.size else 0.0
    for idx in order:
        name = names[idx]
        if importance[idx] <= _INSENSITIVE * top:
            status[name] = "insensitive"
            continue
        values = [_sine_to_span(j[:, idx], j[:, chosen]) for j in jacobians]
        sine = float(np.median(values))
        sines[name] = sine
        if sine < config.collinearity:
            # the chosen parameter most aligned with it
            j = jacobians[0]
            cos = [
                abs(float(j[:, idx] @ j[:, c]))
                / max(
                    float(np.linalg.norm(j[:, idx]) * np.linalg.norm(j[:, c])), 1e-300
                )
                for c in chosen
            ]
            status[name] = f"collinear with {names[chosen[int(np.argmax(cos))]]}"
            continue
        chosen.append(int(idx))
        status[name] = "selected"
    selected = [names[i] for i in chosen]

    # rank estimate from the sensitivity spectrum at the nominal values
    ref = nominal if nominal is not None else jacobians[0]
    singular = np.linalg.svd(ref, compute_uv=False)
    rank = (
        int(np.sum(singular > config.rank_rtol * singular[0])) if singular.size else 0
    )

    n_bands = sum(len(c.data.columns) for c in current.conditions.values())
    runtimes = {
        p: estimate_runtime(state.runtime, config, p, n_bands)
        for p in range(1, len(selected) + 1)
    }
    totals = {
        p: e["fit"] + e["identifiability"] + e["uncertainty"]
        for p, e in runtimes.items()
    }
    by_budget = len(selected)
    if config.time_budget is not None:
        # Computing time of the steps so far, not wall-clock time (pauses
        # between sessions don't count)
        spent = sum(r.duration for k, r in state.results.items() if k == "setup")
        remaining = config.time_budget - spent
        fitting = [p for p, t in totals.items() if t.typical <= remaining]
        by_budget = max(fitting) if fitting else 1
    limits = {
        "selected": len(selected),
        "rank": rank,
        "time_budget": by_budget,
    }
    if config.max_parameters is not None:
        limits["max_parameters"] = config.max_parameters
    max_parameters = max(min(limits.values()), 1) if selected else 0
    proposal = selected[:max_parameters]
    for name in selected[max_parameters:]:
        status[name] = "beyond limit"

    table = {
        name: {
            "importance": float(importance[i]),
            "sine": sines.get(name),
            "status": status[name],
        }
        for i, name in sorted(enumerate(names), key=lambda x: -importance[x[0]])
    }
    limiting = min(limits, key=lambda k: limits[k])
    messages = [
        f"{len(selected)} of {len(names)} candidates are sensitive and not collinear",
        f"Rank estimate of the sensitivities: {rank}",
        f"Proposed {max_parameters} parameters (limited by {limiting}): "
        + ", ".join(proposal),
    ]
    if proposal:
        messages.append(f"Estimated runtime of steps 2-5: {totals[len(proposal)]}")
    return StepResult(
        step="parameters",
        summary={
            "parameters": table,
            "rank": rank,
            "limits": limits,
            "n_samples": len(jacobians),
            "runtime": {
                str(p): {k: v.to_dict() for k, v in e.items()}
                | {"total": totals[p].to_dict()}
                for p, e in runtimes.items()
            },
        },
        proposal={"parameters": proposal},
        messages=messages,
        artifacts={"importance": dict(zip(names, importance.tolist(), strict=True))},
    )


def apply(
    current: Current | None,
    result: StepResult,
    decision: dict[str, Any],
    state: LabState,
) -> Current:
    """Estimate the chosen parameters, keep the others at their values."""
    del result, state
    assert current is not None  # noqa: S101
    out = current.copy()
    chosen = list(decision.get("parameters", []))
    values = out.model.get_parameter_values() | out.model.get_initial_conditions()
    renamed = {
        v: k for c in out.conditions.values() for k, v in (c.local or {}).items()
    }
    values |= {v: values[k] for v, k in renamed.items() if k in values}
    if unknown := sorted(set(chosen) - set(values)):
        msg = f"Unknown parameters chosen: {unknown}"
        raise ValueError(msg)
    out.estimates = {k: float(out.estimates.get(k, values[k])) for k in chosen}
    return out
