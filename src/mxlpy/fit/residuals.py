# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Residual functions used for the different fittting approaches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mxlpy.simulation import OdeSimulation, Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import cast

if TYPE_CHECKING:
    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy._steady_state_builder import SteadyStateModelBuilder
    from mxlpy.fit.abstract import _Settings

__all__ = [
    "evaluate_residual",
    "protocol_time_course_residual",
    "steady_state_residual",
    "time_course_residual",
]


def steady_state_residual(
    updates: dict[str, float],
    settings: _Settings[KineticModelBuilder | OdeModelBuilder],
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
        .simulate_to_steady_state(
            oscillation_detector=settings.oscillation_detector,
        )
        .get_result()
    )
    match val := res.value:
        case Simulation() | OdeSimulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.index)],
            )
        case _:
            return cast(float, np.inf)


def time_course_residual(
    updates: dict[str, float],
    settings: _Settings[KineticModelBuilder | OdeModelBuilder],
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
        case Simulation() | OdeSimulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)


def protocol_time_course_residual(
    updates: dict[str, float],
    settings: _Settings[KineticModelBuilder | OdeModelBuilder],
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
        msg = "protocol_time_course_residual requires a protocol but settings.protocol is None - set settings.protocol to a DataFrame before fitting"
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
        case Simulation() | OdeSimulation():
            return settings.loss(
                val.get_combined().loc[:, cast(list, settings.data.columns)],
            )
        case _:
            return cast(float, np.inf)


def evaluate_residual(
    updates: dict[str, float],
    settings: _Settings[SteadyStateModelBuilder],
) -> float:
    """Calculate residual error between evaluated model and experimental data.

    Parameters
    ----------
    updates
        Parameter updates as {name: value}.
    settings
        Internal fit settings containing model, data, and configuration.

    Returns
    -------
    float
        Residual error, or inf if evaluation fails.

    """
    model = settings.model
    for p in settings.p_names:
        model.update_parameter(p, updates[p])

    try:
        res = model.evaluate(include_parameters=True)
    except ArithmeticError:
        return cast(float, np.inf)
    return settings.loss(res.loc[cast(list, settings.data.index)])
