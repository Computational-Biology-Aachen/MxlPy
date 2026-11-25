# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 2: multi-start fits, goodness of fit and residual checks."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy import fit, select
from mxlpy.fit.callbacks import tagged
from mxlpy.fit.diagnostics import ResidualCheck, check_residuals, chi2_test
from mxlpy.lab._core import (
    Current,
    LabState,
    Option,
    StepResult,
    condition_model,
    nrmse,
    residuals_of,
    safe_name,
    simulate_conditions,
    start_values,
)
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from mxlpy.fit._problem import Refit, _Problem
    from mxlpy.lab._core import LabConfig

__all__ = ["Evaluation", "apply", "evaluate", "multistart", "run"]


def _fit_start(task: tuple[_Problem, dict[str, float], str]) -> Refit | None:
    problem, start, label = task
    if problem.callback is not None:
        problem = replace(problem, callback=tagged(problem.callback, label))
    res = problem.fit_result(deepcopy(problem.model), {}, start)
    if res is None or not math.isfinite(res.objective):
        return None
    return res


def multistart(
    current: Current,
    config: LabConfig,
    *,
    callback: Any,
    n_starts: int | None = None,
    label: str = "start",
) -> tuple[dict[str, float], float, list[float]]:
    """Best estimates, their objective and the objective of every start (inf: failed)."""
    problem = current.problem(config, callback=callback)
    rng = np.random.default_rng(config.seed)
    starts = start_values(
        current.estimates,
        config.n_starts if n_starts is None else n_starts,
        config.spread,
        config.bounds,
        rng,
    )
    tasks = [(i, (problem, s, f"{label}/{i}")) for i, s in enumerate(starts)]
    results = dict(
        parallelise(
            _fit_start,
            tasks,
            parallel=config.parallel and len(tasks) > 1,
            max_workers=config.max_workers,
            tqdm_desc="fits",
            disable_tqdm=True,
        )
    )
    outcomes = [results.get(i) for i in range(len(starts))]
    objectives = [math.inf if r is None else r.objective for r in outcomes]
    done = [r for r in outcomes if r is not None]
    if not done:
        msg = "All starts failed"
        raise ValueError(msg)
    best = min(done, key=lambda r: r.objective)
    estimates = {k: float(best.parameters[k]) for k in current.estimates}
    return estimates, float(best.objective), objectives


class Evaluation:
    """Goodness of fit and residual checks of estimates."""

    def __init__(
        self, current: Current, config: LabConfig, estimates: dict[str, float]
    ) -> None:
        values = replace(current, estimates=estimates)
        self.predictions = simulate_conditions(values)
        self.residuals = residuals_of(current, self.predictions)
        nll = 0.0
        for name, cond in current.conditions.items():
            res = self.residuals.get(name)
            if res is None:
                nll = math.inf
                continue
            data = cast(pd.DataFrame, cond.data)
            nll += cond.loss(None)(data - res, data)
        self.nll = float(nll)
        self.gof = chi2_test(self.nll, current.n_data, len(estimates))
        self.check: ResidualCheck = check_residuals(self.residuals, config.significance)
        self.nrmse = nrmse(current, self.residuals)
        self.significance = config.significance

    @property
    def passed(self) -> bool:
        """χ² test and residual checks passed."""
        return self.gof.passed(self.significance) and self.check.passed

    def summary(self) -> dict[str, Any]:
        """JSON-compatible metrics."""
        return {
            "nll": self.nll,
            "chi2": self.gof.chi2,
            "dof": self.gof.dof,
            "chi2_p": self.gof.p_value,
            "residuals": self.check.table.reset_index().to_dict(orient="records"),
            "flagged": [list(i) for i in self.check.flagged.index],
            "nrmse": self.nrmse.reset_index().to_dict(orient="records"),
            "passed": self.passed,
        }


def evaluate(
    current: Current, config: LabConfig, estimates: dict[str, float]
) -> Evaluation:
    """Checks of the estimates."""
    return Evaluation(current, config, estimates)


def _single(current: Current, name: str) -> Current:
    """The fit of one data set."""
    return replace(current, conditions={name: current.conditions[name]})


def _local_conditions(
    current: Current,
    result: select.ConditionSpecific,
) -> tuple[dict[str, fit.Condition], dict[str, float]]:
    """Conditions with the selected condition-specific estimates, and their values.

    The values are the start of the refit: the shared parameters at the
    reference condition's value, the condition-specific names at their
    group's.
    """
    best = result.best
    conditions = dict(current.conditions)
    estimates: dict[str, float] = {
        str(k): float(best.parameters.loc[result.reference, k])
        for k in best.parameters.columns
        if k in current.estimates
    }
    for k, groups in best.structure.items():
        for group in groups[1:]:
            local_name = f"{k}__{safe_name('_'.join(group))}"
            estimates[local_name] = float(best.parameters.loc[group[0], k])
            for c in group:
                cond = conditions[c]
                conditions[c] = replace(
                    cond, local={**(cond.local or {}), k: local_name}
                )
    return conditions, estimates


def run(state: LabState) -> StepResult:
    """Fit from several starts and check the fit."""
    config = state.config
    current = state.current("fit")
    callback = state.callback("fit")
    estimates, objective, objectives = multistart(current, config, callback=callback)
    finite = np.array([o for o in objectives if math.isfinite(o)])
    converged = (
        float(np.mean(finite - objective < config.converged)) if finite.size else 0.0
    )
    n_failed = sum(not math.isfinite(o) for o in objectives)
    ev = evaluate(current, config, estimates)
    messages = [
        (
            f"Best objective {objective:.4g} (χ² = {ev.gof.chi2:.4g}, "
            f"dof {ev.gof.dof}, p = {ev.gof.p_value:.3g})"
        ),
        (
            f"{converged:.0%} of {len(objectives) - n_failed} finished starts "
            f"converged to the best (Δχ² < {2 * config.converged:g}), "
            f"{n_failed} failed"
        ),
    ]
    if converged < 0.2 and len(objectives) > 1:  # noqa: PLR2004
        messages.append(
            "Few starts reach the best fit: the problem is hard to optimise; "
            "more starts may find a better one"
        )
    if ev.gof.p_value > 0.99:  # noqa: PLR2004
        messages.append(
            "The χ² is much smaller than expected: the noise (sigma) may be "
            "overestimated, which widens all intervals"
        )
    if not ev.check.passed:
        flagged = ", ".join(f"{d}:{o}" for d, o in ev.check.flagged.index)
        messages.append(f"Structured residuals (missing dynamics?): {flagged}")
    summary: dict[str, Any] = {
        "estimates": estimates,
        "objectives": objectives,
        "converged": converged,
        "n_failed": n_failed,
        "joint": ev.summary(),
    }
    artifacts: dict[str, Any] = {"estimates": estimates, "evaluation": ev}
    proposal: dict[str, Any] = {"local": {}}
    passed = ev.passed

    if not passed and len(current.conditions) > 1:
        # Does every data set fit on its own? Then parameters differ between them
        separate = {}
        for name in current.conditions:
            single = _single(current, name)
            try:
                est, _, _ = multistart(
                    replace(single, estimates=estimates),
                    config,
                    callback=callback,
                    n_starts=max(config.n_starts // 2, 1),
                    label=f"{name}/start",
                )
            except ValueError:
                separate[name] = None
                continue
            separate[name] = evaluate(single, config, est)
        summary["separate"] = {
            k: (None if v is None else v.summary()) for k, v in separate.items()
        }
        if all(v is not None and v.passed for v in separate.values()):
            messages.append(
                "Every data set fits on its own: looking for condition-specific parameters"
            )
            settings = {
                name: fit.FitSettings(
                    model=condition_model(current.model, cond, {}),
                    data=cond.data,
                    y0=None,
                    loss_fn=cond.loss(None),
                    protocol=cond.protocol,
                    runner=current.runner,
                )
                for name, cond in current.conditions.items()
            }
            specific = select.condition_specific(
                settings,
                estimates,
                list(estimates),
                loss_fn=fit.GaussianNll(1.0),
                transform=current.transform(),
                bounds=current.bounds(config),
                seed=config.seed,
                parallel=config.parallel,
                max_workers=config.max_workers,
            )
            artifacts["condition_specific"] = specific
            conditions, local_values = _local_conditions(current, specific)
            if any(c.local for c in conditions.values()):
                local_current = replace(
                    current,
                    conditions=conditions,
                    estimates=estimates | local_values,
                )
                local_est, _, _ = multistart(
                    local_current, config, callback=callback, n_starts=1, label="local"
                )
                local_ev = evaluate(local_current, config, local_est)
                summary["local"] = {
                    "structure": specific.best_id,
                    "estimates": local_est,
                    "evaluation": local_ev.summary(),
                }
                proposal["local"] = {
                    k: dict(c.local or {}) for k, c in conditions.items() if c.local
                }
                artifacts["local"] = (local_current, local_est, local_ev)
                messages.append(
                    f"Condition-specific parameters: {specific.best_id} "
                    f"({'passes' if local_ev.passed else 'still fails'} the checks)"
                )
                passed = local_ev.passed
    proposal["next"] = "identifiability" if passed else "diagnose"
    return StepResult(
        step="fit",
        passed=passed,
        summary=summary,
        proposal=proposal,
        options={
            "next": Option(
                "choice",
                ("identifiability", "diagnose"),
                "Go on with the identifiability analysis or diagnose missing dynamics",
            ),
            "local": Option(
                "mapping",
                description=(
                    "Condition-specific estimates: {data set: {parameter: new name}}"
                ),
            ),
        },
        messages=messages,
        artifacts=artifacts,
    )


def apply(
    current: Current | None,
    result: StepResult,
    decision: dict[str, Any],
    state: LabState,
) -> Current:
    """The fitted estimates, condition-specific if decided so."""
    del state
    assert current is not None  # noqa: S101
    out = current.copy()
    local = decision.get("local") or {}
    stored = result.artifacts.get("local")
    if local and stored is not None and local == result.proposal.get("local"):
        local_current, local_est, _ = stored
        out.conditions = dict(local_current.conditions)
        out.estimates = dict(local_est)
        return out
    if local:
        # A structure other than the proposal: start from the shared estimates
        estimates = dict(cast(dict[str, float], result.artifacts["estimates"]))
        for cname, mapping in local.items():
            cond = out.conditions[cname]
            out.conditions[cname] = replace(cond, local=dict(mapping))
            for k, v in mapping.items():
                estimates.setdefault(
                    v, estimates.get(k, out.model.get_parameter_values().get(k, 1.0))
                )
        out.estimates = estimates
        return out
    out.estimates = dict(cast(dict[str, float], result.artifacts["estimates"]))
    return out


def predictions_frame(ev: Evaluation) -> pd.DataFrame:
    """Predictions of all data sets stacked (for reports)."""
    frames = {k: v for k, v in ev.predictions.items() if v is not None}
    return pd.concat(frames, names=["data", "time"]) if frames else pd.DataFrame()
