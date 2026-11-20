# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Residual functions used for the different fittting approaches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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


def _align_time(prediction: pd.DataFrame, time_points: pd.Index) -> pd.DataFrame:
    """Rows of a time-indexed prediction at the time points of the data.

    Simulated time points can differ from the requested ones in the last
    bits (e.g. accumulated protocol step ends), so they're matched within a
    tiny tolerance rather than exactly. Also drops other simulated times,
    e.g. the initial one. Predictions that aren't indexed by unique times
    (e.g. relabelled by a postprocess_fn) are returned unchanged.
    """
    index = pd.Index(time_points)
    if (
        len(prediction) == 0
        or not pd.api.types.is_numeric_dtype(index)
        or not pd.api.types.is_numeric_dtype(prediction.index)
        or not prediction.index.is_unique
    ):
        return prediction
    scale = max(
        1.0, float(np.abs(np.asarray(time_points, dtype=float)).max(initial=0.0))
    )
    return prediction.sort_index().reindex(
        index=time_points, method="nearest", tolerance=1e-9 * scale
    )


def _loss(settings: _Settings[Any], res: Result[Any], names: list[str]) -> float:
    match val := res.value:
        case Exception():
            return cast(float, np.inf)
        case _:
            prediction = val.get_combined()
            if (postprocess := settings.postprocess_fn) is not None:
                prediction = postprocess(prediction)
            if isinstance(settings.data, pd.DataFrame):
                prediction = _align_time(prediction, settings.data.index)
            return settings.loss(prediction.loc[:, names])


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
