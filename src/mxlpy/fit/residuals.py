"""Residual functions used for the different fittting approaches."""

import numpy as np

from mxlpy.fit.abstract import _Settings
from mxlpy.simulation import Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import cast

__all__ = [
    "protocol_time_course_residual",
    "steady_state_residual",
    "time_course_residual",
]


def steady_state_residual(
    updates: dict[str, float],
    settings: _Settings,
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
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_to_steady_state()
        .get_result()
    )
    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.index)],
            )
        case _:
            return cast(float, np.inf)


def time_course_residual(
    updates: dict[str, float],
    settings: _Settings,
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
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_time_course(cast(list, settings.data.index))
        .get_result()
    )

    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)


def protocol_time_course_residual(
    updates: dict[str, float],
    settings: _Settings,
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
    model = settings.model
    if (y0 := settings.y0) is not None:
        model.update_variables(y0)
    for p in settings.p_names:
        model.update_parameter(p, updates[p])
    for p in settings.v_names:
        model.update_variable(p, updates[p])

    if (protocol := settings.protocol) is None:
        msg = "No protocol supplied"
        raise ValueError(msg)

    res = (
        Simulator(
            model,
            integrator=settings.integrator,
        )
        .simulate_protocol_time_course(
            protocol=protocol,
            time_points=settings.data.index,
        )
        .get_result()
    )

    match val := res.value:
        case Simulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)
