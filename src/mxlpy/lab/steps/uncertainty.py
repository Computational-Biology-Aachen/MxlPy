# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Step 5: prediction bands of the observables and an optional posterior."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from mxlpy import fit, identify
from mxlpy.distributions import LogUniform, Uniform
from mxlpy.lab._core import Current, LabState, StepResult

if TYPE_CHECKING:
    from mxlpy.distributions import Prior

__all__ = ["apply", "run"]

_LOGGER = logging.getLogger(__name__)


def _prediction(current: Current, condition: str, target: str) -> identify.Prediction:
    """The prediction of a target in the condition of a data set."""
    cond = current.conditions[condition]
    return identify.Prediction(
        target,
        parameters=dict(cond.parameters) if cond.parameters else None,
        y0=cond.y0,
        protocol=cond.protocol,
        local=dict(cond.local) if cond.local else None,
    )


def _priors(current: Current) -> dict[str, Prior]:
    """Wide priors around the estimates: log-uniform over four decades."""
    out: dict[str, Prior] = {}
    for k, v in current.estimates.items():
        if v > 0:
            out[k] = LogUniform(v / 100, v * 100)
        else:
            width = max(abs(v), 1.0) * 100
            out[k] = Uniform(v - width, v + width)
    return out


def run(state: LabState) -> StepResult:
    """Prediction bands per data set and observable."""
    config = state.config
    current = state.current("uncertainty")
    callback = state.callback("uncertainty")
    messages: list[str] = []
    bands: dict[str, dict[str, identify.PredictionBand]] = {}
    summary_bands: dict[str, Any] = {}
    for name, cond in current.conditions.items():
        data = cast(pd.DataFrame, cond.data)
        times = np.unique(np.asarray(data.index, dtype=float))
        grid = np.linspace(times.min(), times.max(), config.prediction_times)
        bands[name] = {}
        for col in data.columns:
            try:
                band = identify.prediction_band(
                    current.model,
                    current.conditions,
                    current.estimates,
                    prediction=_prediction(current, name, col),
                    times=grid,
                    alpha=1 - config.significance,
                    runner=current.runner,
                    bounds=current.bounds(config),
                    transform=current.transform(),
                    parallel=config.parallel,
                    max_workers=config.max_workers,
                    callback=callback,
                )
            except Exception as e:  # noqa: BLE001
                messages.append(f"Band of {col} in {name} failed: {e}")
                continue
            bands[name][col] = band
            width = (band.upper - band.lower).to_numpy(dtype=float)
            summary_bands[f"{name}/{col}"] = {
                "time": grid,
                "value": band.value,
                "lower": band.lower,
                "upper": band.upper,
                "max_width": float(np.nanmax(width)) if width.size else None,
            }
    n_bands = sum(len(v) for v in bands.values())
    messages.append(f"{n_bands} prediction band(s) at {1 - config.significance:.0%}")
    summary: dict[str, Any] = {
        "estimates": dict(current.estimates),
        "fixed": dict(current.fixed),
        "bands": summary_bands,
    }
    artifacts: dict[str, Any] = {"bands": bands}

    if config.mcmc:
        if len(current.conditions) > 1 or any(
            c.protocol is not None or c.local or c.parameters
            for c in current.conditions.values()
        ):
            messages.append(
                "MCMC skipped: only a single data set without settings is supported"
            )
        else:
            ((name, cond),) = current.conditions.items()
            res = fit.mcmc_time_course(
                current.model,
                data=cast(pd.DataFrame, cond.data),
                prior=_priors(current),
                sigma=cond.sigma if cond.sigma is not None else 1.0,
                n_samples=config.mcmc_samples,
                x0=dict(current.estimates),
                y0=dict(cond.y0) if cond.y0 else None,
                runner=current.runner,
                seed=config.seed,
                parallel=config.parallel,
                max_workers=config.max_workers,
            ).value
            if isinstance(res, Exception):
                messages.append(f"MCMC failed: {res}")
            else:
                artifacts["mcmc"] = res
                summary["mcmc"] = {"name": name, "summary": str(res)}
                messages.append("Posterior sampled with MCMC")
    return StepResult(
        step="uncertainty",
        passed=None,
        summary=summary,
        proposal={},
        messages=messages,
        artifacts=artifacts,
    )


def apply(
    current: Current | None,
    result: StepResult,
    decision: dict[str, Any],
    state: LabState,
) -> Current:
    """Nothing changes after the last step."""
    del result, decision, state
    assert current is not None  # noqa: S101
    return current
