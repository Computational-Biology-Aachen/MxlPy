# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""The steps of the fitting lab.

Every step module has `run(state) -> StepResult`, a pure function of the
results and decisions of the steps before, and `apply(current, result,
decision, state) -> Current`, which hands the model, data and estimates on
to the next step according to the gate's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import diagnose, fit, identifiability, parameters, setup, uncertainty

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.lab._core import Current, LabState, StepResult

__all__ = ["RUNNERS", "apply"]

_MODULES = {
    "setup": setup,
    "parameters": parameters,
    "fit": fit,
    "diagnose": diagnose,
    "identifiability": identifiability,
    "uncertainty": uncertainty,
}

RUNNERS: dict[str, Callable[[LabState], StepResult]] = {
    k: m.run for k, m in _MODULES.items()
}


def apply(
    step: str,
    current: Current | None,
    result: StepResult,
    decision: dict[str, Any],
    state: LabState,
) -> Current:
    """The model, data and estimates after a step's gate."""
    if result.skipped:
        if current is None:
            msg = f"Step {step!r} can't be skipped"
            raise RuntimeError(msg)
        return current
    return _MODULES[step].apply(current, result, decision, state)
