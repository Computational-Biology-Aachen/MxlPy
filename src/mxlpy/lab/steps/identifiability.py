# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 4: Fisher information, pulling test and profile likelihoods."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from mxlpy import identify
from mxlpy.lab._core import Current, LabState, StepResult

if TYPE_CHECKING:
    import pandas as pd

    from mxlpy.lab._core import LabConfig

__all__ = ["apply", "run"]

_LOGGER = logging.getLogger(__name__)


def _common(current: Current, config: LabConfig, callback: Any) -> dict[str, Any]:
    return {
        "runner": current.runner,
        "bounds": current.bounds(config),
        "transform": current.transform(),
        "parallel": config.parallel,
        "max_workers": config.max_workers,
        "callback": callback,
    }


def run(state: LabState) -> StepResult:
    """Escalating identifiability analysis of the estimates."""
    config = state.config
    current = state.current("identifiability")
    callback = state.callback("identifiability")
    names = list(current.estimates)
    messages: list[str] = []
    summary: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}

    fim = identify.fisher_information(
        current.model,
        current.conditions,
        current.estimates,
        runner=current.runner,
        transform=current.transform(),
    )
    null = fim.non_identifiable()
    artifacts["fisher"] = fim
    summary["fisher"] = {
        "rank": fim.rank,
        "std_errors": fim.std_errors,
        "wald_ci": fim.wald_ci(),
        "null_directions": null,
    }
    messages.append(f"Fisher information: rank {fim.rank} of {len(names)}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test = identify.identifiability_test(
            current.model,
            current.conditions,
            current.estimates,
            n_fits=config.n_pull_fits,
            recursive=True,
            alpha=1 - config.significance,
            seed=config.seed,
            **_common(current, config, callback),
        )
    inconclusive = [str(w.message) for w in caught]
    artifacts["pulling"] = test
    summary["pulling"] = {
        "non_identifiable": test.non_identifiable,
        "displacement": test.displacement,
        "warnings": inconclusive,
    }
    if test.non_identifiable:
        messages.append(
            f"Pulling test: non-identifiable {', '.join(test.non_identifiable)}"
        )
    else:
        messages.append("Pulling test: all estimates identifiable")

    flagged = list(dict.fromkeys([*test.non_identifiable, *_null_names(null, names)]))
    profiled = names if config.profile == "all" else flagged
    fix: list[str] = []
    if profiled:
        profiles = identify.profile_likelihood(
            current.model,
            current.conditions,
            current.estimates,
            names=profiled,
            alpha=1 - config.significance,
            **_common(current, config, callback),
        )
        artifacts["profiles"] = profiles
        ci = profiles.ci
        summary["profiles"] = {"ci": ci, "better_optimum": profiles.better_optimum}
        non_identifiable = [
            k for k in profiled if profiles[k].identifiability != "identifiable"
        ]
        if profiles.better_optimum:
            messages.append(
                "The profiles found a better optimum: the fit stopped in a local one"
            )
        if non_identifiable:
            related = identify.related_parameters(profiles)
            artifacts["related"] = related
            summary["related"] = [str(r) for r in related.relations]
            messages.append(
                "Profiles: not identifiable "
                + ", ".join(
                    f"{k} ({profiles[k].identifiability})" for k in non_identifiable
                )
            )
            messages.extend(f"Identifiable combination: {r}" for r in related.relations)
            # Every related group keeps its most sensitive member; the others,
            # and non-identifiable estimates in no group, are fixed
            importance = cast(
                dict[str, float],
                state.results["parameters"].artifacts.get("importance", {}),
            )
            grouped: set[str] = set()
            for group in related.groups:
                members = [k for k in group if k in current.estimates]
                grouped |= set(members)
                ranked = sorted(members, key=lambda k: -importance.get(k, 0.0))
                fix.extend(ranked[1:])
            fix.extend(k for k in non_identifiable if k not in grouped)
            fix = list(dict.fromkeys(fix))
        else:
            messages.append("Profiles: all profiled estimates identifiable")
    passed = not fix
    if fix:
        messages.append(f"Proposed: fix {', '.join(fix)} at their values and refit")
    return StepResult(
        step="identifiability",
        passed=passed,
        summary=summary,
        proposal={"fix": fix, "refit": bool(fix)},
        messages=messages,
        artifacts=artifacts,
    )


def _null_names(null: pd.DataFrame, names: list[str]) -> list[str]:
    """Estimates with large weights in the null directions of the Fisher information."""
    if null.empty:
        return []
    weights = null.abs().max(axis=0)
    return [str(k) for k, v in weights.items() if v > 0.3 and k in names]  # noqa: PLR2004


def apply(
    current: Current | None,
    result: StepResult,
    decision: dict[str, Any],
    state: LabState,
) -> Current:
    """Fix the decided estimates, and refit the others."""
    assert current is not None  # noqa: S101
    fix = [k for k in decision.get("fix", []) if k in current.estimates]
    if not fix:
        return current
    out = current.copy()
    fixed = {k: out.estimates.pop(k) for k in fix}
    out.fixed |= fixed
    # Fixed values go into the model, or the conditions for local names
    addressable = set(out.model.get_addressable_names())
    out.model.update_addressable({k: v for k, v in fixed.items() if k in addressable})
    if variables := {
        k: v for k, v in fixed.items() if k in out.model.get_variable_names()
    }:
        out.model.update_variables(variables)
    for name, cond in out.conditions.items():
        if cond.local and (gone := {k: v for k, v in cond.local.items() if v in fixed}):
            out.conditions[name] = replace(
                cond,
                local={k: v for k, v in cond.local.items() if k not in gone} or None,
                parameters={
                    **(cond.parameters or {}),
                    **{k: fixed[v] for k, v in gone.items()},
                },
            )
    if decision.get("refit", True) and out.estimates:
        # Cached on the result, as every later step replays the gates
        refits = result.artifacts.setdefault("refits", {})
        key = tuple(sorted(fix))
        if key not in refits:
            problem = out.problem(state.config)
            res = problem.fit_result(deepcopy(problem.model), {}, dict(out.estimates))
            if res is None:
                _LOGGER.warning(
                    "Refitting with %s fixed failed, keeping the estimates", fix
                )
                refits[key] = dict(out.estimates)
            else:
                refits[key] = {k: float(res.parameters[k]) for k in out.estimates}
        out.estimates = dict(refits[key])
    return out
