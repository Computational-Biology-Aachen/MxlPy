# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting lab: a guided, checkpointed pipeline for fitting a model to data.

`FittingLab` takes one model and one or several time-course data sets
through the recommended steps of fitting an ODE model: setup (observables,
noise, integrator), parameter selection, multi-start fits with statistical
checks, diagnosis of missing dynamics, identifiability and prediction
uncertainty. Every step stops at a review gate with a proposal, which the
next step accepts unless `decide` overrides it; `run(auto=True)` goes
straight through. The run directory holds the configuration, decisions,
step results and a progress stream (`events.jsonl`), so runs can be resumed
and watched. Steps are addressed by key (`"fit"`), number (`2`) or method
name (`"step2_fit"`). The statistics live in `mxlpy.fit.diagnostics`.

Examples
--------
    >>> from mxlpy import lab
    >>> run = lab.FittingLab(model, {"wt": df_wt, "ko": df_ko}, "runs/glyco",
    ...                      observables={"pERK": lab.Observable("ERK_p", scale=True)},
    ...                      time_budget="1h")
    >>> run.on_event(lambda ev: print(ev.step, ev.best_loss, ev.eta))
    >>> run.run(auto=True)
    >>> run.report()

"""

from mxlpy.lab._core import LabConfig, Observable, StepName, StepResult
from mxlpy.lab._lab import FittingLab, LabEvent

__all__ = [
    "FittingLab",
    "LabConfig",
    "LabEvent",
    "Observable",
    "StepName",
    "StepResult",
]
