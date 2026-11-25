# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 3: diagnose missing dynamics with hidden states and structural variants."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mxlpy import fit, fns, select
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.fit.callbacks import tagged
from mxlpy.fit.diagnostics import check_residuals
from mxlpy.lab._core import Current, LabState, Option, StepResult, safe_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.lab._core import LabConfig, Model

__all__ = ["apply", "generic_variants", "run"]

_LOGGER = logging.getLogger(__name__)
_BLOCK = "lab"
# Corrections below this fraction of the largest don't flag a target
_FLAG = 0.1
# Information criterion difference a variant needs to be proposed
_DELTA = 2.0


def _negative_mass_action(x: float, k: float) -> float:
    """`-k x`, a first-order loss added as a term."""
    return -k * x


def _time_span(current: Current) -> tuple[float, float]:
    times = np.concatenate(
        [np.asarray(c.data.index, dtype=float) for c in current.conditions.values()]
    )
    return float(times.min()), float(times.max())


def _degradation(variable: str, rate: float) -> Callable[[Model], Model]:
    def edit(model: Model) -> Model:
        k = f"kdeg_{safe_name(variable)}"
        model.add_parameter(k, rate)
        model.add_term(
            f"deg_{safe_name(variable)}",
            _negative_mass_action,
            args=[variable, k],
            target=variable,
            mechanism="additive",
        )
        return model

    return edit


def _spline_input(variable: str, knots: list[float]) -> Callable[[Model], Model]:
    def edit(model: Model) -> Model:
        name = f"input_{safe_name(variable)}"
        model.add_spline_input(name, knots, initial=1e-3, positive=True)
        model.add_term(
            f"in_{safe_name(variable)}",
            fns.constant,
            args=[name],
            target=variable,
            mechanism="additive",
        )
        return model

    return edit


def _delay(variable: str, stages: int, rate: float) -> Callable[[Model], Model] | None:
    """Route the production of `variable` through a linear chain (kinetic models)."""

    def edit(model: Model) -> Model:
        model = cast(KineticModelBuilder, model)
        k = f"kdelay_{safe_name(variable)}"
        model.add_parameter(k, rate)
        chain = [f"{safe_name(variable)}_delay{i}" for i in range(1, stages + 1)]
        model.add_variables(dict.fromkeys(chain, 0.0))
        for name, rxn in model.get_raw_reactions().items():
            stoich = dict(rxn.stoichiometry)
            value = stoich.get(variable)
            if isinstance(value, int | float) and value > 0:
                stoich[chain[0]] = stoich.pop(variable)
                model.update_reaction(name, stoichiometry=stoich)
        for i, src in enumerate(chain):
            dst = chain[i + 1] if i + 1 < len(chain) else variable
            model.add_reaction(
                f"{src}_flux",
                fns.mass_action_1s,
                args=[src, k],
                stoichiometry={src: -1, dst: 1},
            )
        return model

    return edit


def generic_variants(
    current: Current, config: LabConfig, variables: list[str]
) -> dict[str, tuple[Callable[[Model], Model], dict[str, float]]]:
    """The generic structural edits of flagged variables and the starts of their parameters."""
    t0, t1 = _time_span(current)
    span = max(t1 - t0, 1e-12)
    out: dict[str, tuple[Callable[[Model], Model], dict[str, float]]] = {}
    model = current.model
    for v in variables:
        tag = safe_name(v)
        out[f"degradation[{v}]"] = (
            _degradation(v, 1 / span),
            {f"kdeg_{tag}": 1 / span},
        )
        knots = [float(i) for i in np.linspace(t0, t1, max(config.spline_knots, 2))]
        names = [f"input_{tag}_{i}" for i in range(len(knots))]
        out[f"input[{v}]"] = (_spline_input(v, knots), dict.fromkeys(names, 1e-3))
        if isinstance(model, KineticModelBuilder) and any(
            isinstance(s := r.stoichiometry.get(v), int | float) and s > 0
            for r in model.get_raw_reactions(as_copy=False).values()
        ):
            rate = config.delay_stages / (0.2 * span)
            edit = _delay(v, config.delay_stages, rate)
            if edit is not None:
                out[f"delay[{v}]"] = (edit, {f"kdelay_{tag}": rate})
    return out


def _augmented_conditions(current: Current) -> dict[str, fit.Condition]:
    """Conditions with condition-specific estimates fixed at their values."""
    out = {}
    for name, cond in current.conditions.items():
        if not cond.local:
            out[name] = cond
            continue
        fixed = {k: current.estimates[v] for k, v in cond.local.items()}
        out[name] = replace(
            cond, local=None, parameters={**(cond.parameters or {}), **fixed}
        )
    return out


def _targets(current: Current, state: LabState) -> list[str]:
    """Model names the hidden states correct: targets of the flagged observables."""
    model = current.model
    allowed = set(model.get_variable_names())
    if isinstance(model, KineticModelBuilder):
        allowed |= set(model.get_reaction_names())
    setup = state.results["setup"].summary
    targets_of = cast(dict[str, str], setup.get("targets", {}))
    fit_result = state.results.get("fit")
    flagged: list[str] = []
    if fit_result is not None:
        flagged = [o for _, o in fit_result.summary["joint"].get("flagged", [])]
    picked = [targets_of.get(o, o) for o in flagged]
    picked = [t for t in dict.fromkeys(picked) if t in allowed]
    if not picked:
        picked = [t for t in dict.fromkeys(targets_of.values()) if t in allowed]
    if not picked:
        picked = list(model.get_variable_names())
    return picked


def run(state: LabState) -> StepResult:
    """Hidden states until the residuals are white, then structural variants."""
    config = state.config
    current = state.current("diagnose")
    callback = state.callback("diagnose")
    conditions = _augmented_conditions(current)
    model_names = set(current.model.get_addressable_names()) | set(
        current.model.get_variable_names()
    )
    p0 = {k: v for k, v in current.estimates.items() if k in model_names}
    targets = _targets(current, state)
    messages = [f"Hidden states correct {', '.join(targets)}"]

    hidden: dict[int, dict[str, Any]] = {}
    augmented: dict[int, fit.AugmentedFit] = {}
    chosen: int | None = None
    for n in range(1, config.max_hidden + 1):
        model = deepcopy(current.model)
        model.add_hidden_states(
            _BLOCK,
            n_hidden=n,
            targets=targets,
            mechanism="additive",
            layers=config.hidden_layers,
            seed=config.seed,
        )
        res = fit.augmented_time_course(
            model,
            data=conditions,
            p0=p0,
            runner=current.runner,
            callback=tagged(callback, f"hidden{n}"),
        ).value
        if isinstance(res, Exception):
            hidden[n] = {"failed": fit.callbacks.failure_message(res)}
            messages.append(
                f"Training {n} hidden state(s) failed: {hidden[n]['failed']}"
            )
            break
        check = check_residuals(res.residuals, config.significance)
        augmented[n] = res
        hidden[n] = {
            "loss": res.loss,
            "white": check.passed,
            "flagged": [list(i) for i in check.flagged.index],
            "n_weights": len(model.get_weight_names()),
        }
        if check.passed:
            chosen = n
            break
    if chosen is None and augmented:
        chosen = min(augmented, key=lambda k: augmented[k].loss)
        messages.append(
            f"Residuals stay structured with up to {config.max_hidden} hidden state(s)"
        )
    elif chosen is not None:
        messages.append(f"{chosen} hidden state(s) make the residuals white")

    correction_summary: dict[str, float] = {}
    flagged_vars: list[str] = []
    if chosen is not None:
        res = augmented[chosen]
        for col in res.correction.columns:
            rms = float(
                np.sqrt(
                    np.nanmean(np.square(res.correction[col].to_numpy(dtype=float)))
                )
            )
            scale = res.scales.get(col, 1.0) or 1.0
            correction_summary[col] = rms / scale
        top = max(correction_summary.values(), default=0.0)
        flagged_vars = [
            col.removeprefix(f"{_BLOCK}_")
            for col, v in correction_summary.items()
            if top > 0 and v >= _FLAG * top
        ]
        flagged_vars = [
            v for v in flagged_vars if v in current.model.get_variable_names()
        ]
        if flagged_vars:
            messages.append(
                f"The mechanism is off mostly for {', '.join(flagged_vars)}"
            )

    summary: dict[str, Any] = {
        "targets": targets,
        "hidden": {str(k): v for k, v in hidden.items()},
        "chosen_hidden": chosen,
        "correction": correction_summary,
        "flagged": flagged_vars,
    }
    artifacts: dict[str, Any] = {"augmented": augmented}
    proposal: dict[str, Any] = {"model": "original"}

    if config.structure_search:
        edits: dict[str, tuple[Callable[[Model], Model], dict[str, float]]] = {
            name: (fn, {}) for name, fn in state.variants.items()
        }
        edits |= generic_variants(current, config, flagged_vars)
        models: dict[str, Model] = {"original": deepcopy(current.model)}
        starts = dict(current.estimates)
        for name, (edit, new) in edits.items():
            try:
                models[name] = edit(deepcopy(current.model))
            except Exception as e:  # noqa: BLE001
                messages.append(f"Variant {name} couldn't be built: {e}")
                continue
            starts |= new
        if len(models) > 1:
            known = set().union(
                *(
                    set(m.get_addressable_names()) | set(m.get_variable_names())
                    for m in models.values()
                )
            )
            local = {
                v for c in current.conditions.values() for v in (c.local or {}).values()
            }
            p0_all = {k: v for k, v in starts.items() if k in known or k in local}
            selection = select.compare(
                models,
                current.conditions,
                p0_all,
                criterion="aicc",
                n_starts=max(config.n_starts // 2, 1),
                runner=current.runner,
                transform={
                    k: "log" if v > 0 and k not in current.linear else "linear"
                    for k, v in p0_all.items()
                },
                bounds=current.bounds(config, list(p0_all)),
                seed=config.seed,
                parallel=config.parallel,
                max_workers=config.max_workers,
                callback=callback,
            )
            table = selection.table
            artifacts["selection"] = selection
            artifacts["models"] = models
            summary["variants"] = table.reset_index().to_dict(orient="records")
            original = selection["original"].criterion("aicc")
            best = selection.best_id
            if (
                best != "original"
                and original - selection[best].criterion("aicc") > _DELTA
            ):
                proposal["model"] = best
                messages.append(f"Best structural variant: {best}")
            else:
                messages.append("No structural variant improves the model clearly")
            if chosen is not None:
                messages.append(
                    f"Hidden states reach a loss of {augmented[chosen].loss:.4g}, "
                    f"{best} {selection[best].loss:.4g}"
                )
    return StepResult(
        step="diagnose",
        passed=bool(chosen is not None and hidden[chosen].get("white")),
        summary=summary,
        proposal=proposal,
        options={
            "model": Option(
                "choice",
                tuple(artifacts.get("models", {"original": None})),
                "Model the later steps continue with",
            )
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
    """The chosen model and its estimates."""
    del state
    assert current is not None  # noqa: S101
    name = decision.get("model", "original")
    if result.skipped or name == "original":
        return current
    models = result.artifacts.get("models", {})
    if name not in models:
        msg = f"Unknown model {name!r}, choose from {sorted(models)}"
        raise ValueError(msg)
    selection = cast(select.Selection, result.artifacts["selection"])
    candidate = selection[name]
    out = current.copy()
    out.model = deepcopy(models[name])
    out.estimates = {k: float(v) for k, v in candidate.best_pars.items()}
    return out
