# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 0: observables, noise, integrator and timing."""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy import fns, mca
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.fit.diagnostics import chi2_test, estimate_sigma
from mxlpy.lab._core import (
    Current,
    LabState,
    Observable,
    Option,
    StepResult,
    simulate_conditions,
)
from mxlpy.runners import default_runner

if TYPE_CHECKING:
    from mxlpy import fit
    from mxlpy.lab._core import Model

__all__ = ["apply", "run"]

_LOGGER = logging.getLogger(__name__)
_STIFF = 1e3


def _outputs(model: Model) -> set[str]:
    """Names a data column can measure directly."""
    names = set(model.get_variable_names()) | set(model.get_derived_variable_names())
    if isinstance(model, KineticModelBuilder):
        names |= set(model.get_reaction_names())
    return names | set(model.get_term_names())


def _add_observables(
    model: Model,
    columns: set[str],
    observables: dict[str, Observable],
    conditions: dict[str, fit.Condition],
) -> tuple[Model, list[str], dict[str, str]]:
    """The model with the observables, their parameters and the column targets."""
    model = deepcopy(model)
    outputs = _outputs(model)
    added: list[str] = []
    targets: dict[str, str] = {}
    for col in sorted(columns):
        obs = observables.get(col)
        if obs is None:
            if col not in outputs:
                msg = (
                    f"Data column {col!r} is no model output, map it with "
                    f"observables={{{col!r}: lab.Observable(...)}}"
                )
                raise ValueError(msg)
            targets[col] = col
            continue
        if obs.target not in outputs:
            msg = (
                f"Observable {col!r} measures {obs.target!r}, which is no model output"
            )
            raise ValueError(msg)
        targets[col] = obs.target
        if col == obs.target and not (obs.scale or obs.offset):
            continue
        if col in outputs or col in model.get_addressable_names():
            msg = f"Observable column {col!r} is already a model name"
            raise ValueError(msg)
        args = [obs.target]
        values = pd.concat(
            [
                cast(pd.DataFrame, c.data)[col]
                for c in conditions.values()
                if col in c.data
            ]
        ).dropna()
        if obs.scale:
            name = f"scale_{col}"
            model.add_parameter(name, 1.0)
            added.append(name)
            args.append(name)
        if obs.offset:
            name = f"offset_{col}"
            model.add_parameter(
                name, float(values.to_numpy(dtype=float).min()) if len(values) else 0.0
            )
            added.append(name)
            args.append(name)
        model.add_derived(col, _observable_fn(obs.scale, obs.offset), args=args)
    return model, added, targets


def _scaled(target: float, scale: float) -> float:
    return target * scale


def _offset(target: float, offset: float) -> float:
    return target + offset


def _scaled_offset(target: float, scale: float, offset: float) -> float:
    return target * scale + offset


def _observable_fn(scale: bool, offset: bool) -> Any:  # noqa: FBT001
    if scale and offset:
        return _scaled_offset
    if scale:
        return _scaled
    if offset:
        return _offset
    return fns.constant


def _initial_scales(current: Current, added: list[str]) -> None:
    """Start values of the scales: the data's RMS over the simulated target's."""
    scales = [k for k in added if k.startswith("scale_")]
    if not scales:
        return
    predictions = simulate_conditions(current)
    for name in scales:
        col = name.removeprefix("scale_")
        offset = current.estimates.get(f"offset_{col}", 0.0)
        data, simulated = [], []
        for cname, cond in current.conditions.items():
            pred = predictions.get(cname)
            if pred is None or col not in cond.data:
                continue
            d = cast(pd.DataFrame, cond.data)[col]
            p = (
                pred[col]
                .reindex(np.asarray(d.index, dtype=float))
                .to_numpy(dtype=float)
            )
            data.append(d.to_numpy(dtype=float) - offset)
            simulated.append(p - offset)
        if not data:
            continue
        d_all, s_all = np.concatenate(data), np.concatenate(simulated)
        ok = np.isfinite(d_all) & np.isfinite(s_all)
        num = float(np.sqrt(np.mean(d_all[ok] ** 2))) if ok.any() else 0.0
        den = float(np.sqrt(np.mean(s_all[ok] ** 2))) if ok.any() else 0.0
        if num > 0 and den > 0:
            current.estimates[name] = num / den
            current.model.update_parameter(name, num / den)


def _sigma(
    cond: fit.Condition, given: Any
) -> tuple[pd.Series | pd.DataFrame | float, str]:
    """Noise of a data set and where it came from."""
    if cond.sigma is not None:
        return cond.sigma, "condition"
    if given is not None:
        return (
            pd.Series(given, dtype=float) if isinstance(given, dict) else given
        ), "config"
    return estimate_sigma(cast(pd.DataFrame, cond.data)), "estimated"


def _pool_estimates(
    sigmas: dict[str, Any],
    sources: dict[str, str],
    conditions: dict[str, fit.Condition],
) -> None:
    """Pool estimated noise of the same observable over data sets (in place).

    An observable's measurement noise is usually the same in all
    experiments; pooling (weighted by data points) makes the estimate more
    precise. Given values aren't changed.
    """
    estimated = [k for k, v in sources.items() if v == "estimated"]
    if len(estimated) < 2:  # noqa: PLR2004
        return
    ss: dict[str, float] = {}
    counts: dict[str, int] = {}
    for k in estimated:
        data = cast(pd.DataFrame, conditions[k].data)
        for col, value in cast(pd.Series, sigmas[k]).items():
            n = int(np.isfinite(data[col].to_numpy(dtype=float)).sum())
            ss[str(col)] = ss.get(str(col), 0.0) + n * float(value) ** 2
            counts[str(col)] = counts.get(str(col), 0) + n
    pooled = {c: math.sqrt(ss[c] / counts[c]) for c in ss if counts[c] > 0}
    for k in estimated:
        sigmas[k] = pd.Series(
            {
                c: pooled.get(str(c), float(v))
                for c, v in cast(pd.Series, sigmas[k]).items()
            },
            dtype=float,
        )
        sources[k] = "estimated (pooled)"


def _stiffness(model: Model) -> float | None:
    """Stiffness ratio of the Jacobian at the initial conditions."""
    if not isinstance(model, KineticModelBuilder):
        return None
    try:
        eig = mca.steady_state_stability(
            model, model.get_initial_conditions()
        ).eigenvalues
    except Exception:  # noqa: BLE001
        return None
    real = np.sort(np.abs(np.asarray(eig).real))
    real = real[real > 1e-12 * max(float(real.max(initial=0.0)), 1.0)]
    if len(real) < 2:  # noqa: PLR2004
        return None
    return float(real.max() / real.min())


INTEGRATORS = ("default", "native-radau", "native-lsoda", "user")


def runner_for(key: str, model: Model, given: Any) -> Any | None:
    """The runner of an integrator choice, None if it can't simulate the model."""
    if key == "user":
        return given
    if key == "default":
        return default_runner(model)
    if key not in INTEGRATORS:
        msg = f"Unknown integrator {key!r}, choose from {INTEGRATORS}"
        raise ValueError(msg)
    try:
        from mxlpy.integrators.int_native import Native  # noqa: PLC0415
    except ImportError:
        return None
    method = "Radau" if key == "native-radau" else "LSODA"
    runner: Any = type(default_runner(model))(integrator=partial(Native, method=method))
    try:
        res = runner.time_course(deepcopy(model), [0.0, 1.0])
    except Exception:  # noqa: BLE001
        return None
    return None if isinstance(res.value, Exception) else runner


def _runner(model: Model, stiffness: float | None, given: Any) -> tuple[Any, str]:
    """The runner and its integrator choice (see `INTEGRATORS`)."""
    if given is not None:
        return given, "user"
    key = (
        "native-radau"
        if stiffness is not None and stiffness > _STIFF
        else "native-lsoda"
    )
    if (runner := runner_for(key, model, given)) is not None:
        return runner, key
    return default_runner(model), "default"


def run(state: LabState) -> StepResult:
    """Prepare the model and data and seed the runtime model."""
    config = state.config
    conditions = state.conditions
    for name, cond in conditions.items():
        if not isinstance(cond.data, pd.DataFrame):
            msg = f"The lab fits time courses, data set {name!r} isn't a DataFrame"
            raise TypeError(msg)
    columns = {c for cond in conditions.values() for c in cond.data.columns}
    model, added, targets = _add_observables(
        state.model, columns, config.observables, conditions
    )

    sigmas, sources = {}, {}
    for name, cond in conditions.items():
        sigmas[name], sources[name] = _sigma(cond, config.sigma.get(name))
    _pool_estimates(sigmas, sources, conditions)
    with_sigma = {
        name: replace(cond, sigma=sigmas[name]) for name, cond in conditions.items()
    }

    stiffness = _stiffness(model)
    runner, integrator = _runner(model, stiffness, state.runner)

    # Condition-specific names of the user's conditions, and what they rename
    renamed = {v: k for c in conditions.values() for k, v in (c.local or {}).items()}
    if config.parameters is not None:
        candidates = list(config.parameters)
        known = set(model.get_parameter_names()) | set(model.get_variable_names())
        if unknown := sorted(set(candidates) - known - set(renamed)):
            msg = f"Candidate parameters unknown to the model: {unknown}"
            raise ValueError(msg)
        candidates += [k for k in added if k not in candidates]
    else:
        set_everywhere = set.intersection(
            *(set(c.parameters or {}) for c in conditions.values())
        )
        candidates = [
            k
            for k in model.get_parameter_names()
            if k not in set_everywhere and k not in model.get_weight_names()
        ]
    candidates += [k for k in renamed if k not in candidates]
    values = model.get_parameter_values() | model.get_initial_conditions()
    values |= {k: values[v] for k, v in renamed.items()}
    current = Current(
        model=model,
        conditions=with_sigma,
        estimates={k: float(values[k]) for k in candidates},
        runner=runner,
        linear={k for k in added if k.startswith("offset_")},
    )
    _initial_scales(current, added)

    t0 = time.perf_counter()
    predictions = simulate_conditions(current)
    t_sim = time.perf_counter() - t0
    if failed := [k for k, v in predictions.items() if v is None]:
        msg = f"Simulating the data sets {failed} failed with the nominal parameters"
        raise ValueError(msg)
    problem = current.problem(config)
    t0 = time.perf_counter()
    try:
        problem.loss_jacobian(current.estimates)
        t_sens = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        _LOGGER.warning("Timing the sensitivities failed: %s", e)
        t_sens = math.nan
    state.runtime.seed(t_sim, t_sens, len(candidates))

    nominal = problem.fit_result(deepcopy(problem.model), dict(current.estimates), {})
    nll = math.inf if nominal is None else nominal.loss
    gof = chi2_test(nll, current.n_data, 0)

    sigma_table = {
        name: (
            {k: float(v) for k, v in s.items()}
            if isinstance(s, pd.Series)
            else ("per data point" if isinstance(s, pd.DataFrame) else float(s))
        )
        for name, s in sigmas.items()
    }
    messages = [
        (
            f"{len(conditions)} data set(s), {current.n_data} data points, "
            f"{len(candidates)} candidate parameters"
        ),
        f"Noise: {', '.join(f'{k}: {v}' for k, v in sources.items())}",
        f"Integrator: {integrator}"
        + (f" (stiffness ratio {stiffness:.3g})" if stiffness is not None else ""),
        (
            f"One simulation of all data sets: {t_sim * 1e3:.0f} ms, "
            f"with sensitivities: {t_sens * 1e3:.0f} ms"
        ),
    ]
    if added:
        messages.append(f"Observable parameters: {', '.join(added)}")
    return StepResult(
        step="setup",
        summary={
            "n_data": {
                k: int(np.isfinite(c.data.to_numpy(dtype=float)).sum())
                for k, c in conditions.items()
            },
            "targets": targets,
            "observable_parameters": added,
            "sigma": sigma_table,
            "sigma_source": sources,
            "stiffness": stiffness,
            "integrator": integrator,
            "time_simulation": t_sim,
            "time_sensitivities": t_sens,
            "candidates": candidates,
            "nominal_chi2": gof.chi2,
        },
        proposal={"sigma": sigma_table, "integrator": integrator},
        options={
            "sigma": Option(
                "mapping",
                description=(
                    "Measurement noise per data set: a number or {observable: number}"
                ),
            ),
            "integrator": Option(
                "choice",
                tuple(
                    k for k in INTEGRATORS if k != "user" or state.runner is not None
                ),
                "Integrator of all simulations",
            ),
        },
        messages=messages,
        artifacts={"current": current, "sigmas": sigmas},
    )


def apply(
    _: Current | None, result: StepResult, decision: dict[str, Any], state: LabState
) -> Current:
    """The prepared model and data, with the noise of the decision."""
    current = cast(Current, result.artifacts["current"]).copy()
    proposed = result.proposal.get("sigma", {})
    for name, sigma in decision.get("sigma", {}).items():
        if name not in current.conditions or sigma == proposed.get(name):
            continue
        value = (
            pd.Series(sigma, dtype=float) if isinstance(sigma, dict) else float(sigma)
        )
        current.conditions[name] = replace(current.conditions[name], sigma=value)
    if (integrator := decision.get("integrator")) not in (
        None,
        result.proposal.get("integrator"),
    ):
        runner = runner_for(str(integrator), current.model, state.runner)
        if runner is None:
            msg = f"The integrator {integrator!r} can't simulate the model"
            raise ValueError(msg)
        current.runner = runner
    return current
