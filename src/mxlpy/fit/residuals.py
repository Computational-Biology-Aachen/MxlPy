# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Residual functions used for the different fittting approaches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.runners import DynamicRunner
from mxlpy.types import cast

if TYPE_CHECKING:
    from mxlpy.fit.abstract import _Settings
    from mxlpy.types import Result

__all__ = [
    "protocol_time_course_residual",
    "steady_state_residual",
    "time_course_residual",
]


def _apply_updates(updates: dict[str, float], settings: _Settings[Any]) -> None:
    model = settings.model
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    if isinstance(model, SteadyStateModelBuilder):
        return
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.v_names:
        model.update_variable(p, updates[p])


def _loss(settings: _Settings[Any], res: Result[Any], names: list[str]) -> float:
    match val := res.value:
        case Exception():
            return cast(float, np.inf)
        case _:
            return settings.loss(val.get_combined().loc[:, names])


def steady_state_residual(
    updates: dict[str, float],
    settings: _Settings[Any],
) -> float:
    """Calculate residual error between model steady state and experimental data.

    Parameters
    ----------
    updates
        Parameter and variable updates as {name: value}.
    settings
        Internal fit settings containing model, data, and configuration.

    Returns
    -------
    float
        Residual error, or inf if simulation fails.

    """
    _apply_updates(updates, settings)
    return _loss(
        settings,
        settings.runner.steady_state(settings.model),
        cast(list, settings.data.index),
    )


def time_course_residual(
    updates: dict[str, float],
    settings: _Settings[Any],
) -> float:
    """Calculate residual error between model time course and experimental data.

    Parameters
    ----------
    updates
        Parameter and variable updates as {name: value}.
    settings
        Internal fit settings containing model, data, and configuration.

    Returns
    -------
    float
        Residual error, or inf if simulation fails.

    """
    _apply_updates(updates, settings)
    runner = cast(DynamicRunner[Any, Any], settings.runner)
    return _loss(
        settings,
        runner.time_course(settings.model, cast(list, settings.data.index)),
        cast(list, settings.data.columns),
    )


def protocol_time_course_residual(
    updates: dict[str, float],
    settings: _Settings[Any],
) -> float:
    """Calculate residual error between model protocol time course and experimental data.

    Parameters
    ----------
    updates
        Parameter and variable updates as {name: value}.
    settings
        Internal fit settings containing model, data, and configuration.

    Returns
    -------
    float
        Residual error, or inf if simulation fails.

    """
    if (protocol := settings.protocol) is None:
        msg = "protocol_time_course_residual requires a protocol but settings.protocol is None - set settings.protocol to a DataFrame before fitting"
        raise ValueError(msg)

    _apply_updates(updates, settings)
    runner = cast(DynamicRunner[Any, Any], settings.runner)
    return _loss(
        settings,
        runner.protocol_time_course(settings.model, protocol, settings.data.index),
        cast(list, settings.data.columns),
    )
