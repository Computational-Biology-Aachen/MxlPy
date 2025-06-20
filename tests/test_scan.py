from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, fns
from mxlpy.integrators import DefaultIntegrator
from mxlpy.scan import (
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and_initial_conditions,
    steady_state,
    time_course,
)


@pytest.fixture
def simple_model() -> Model:
    model = Model()
    model.add_parameters({"k1": 1.0, "k2": 2.0})
    model.add_variables({"S": 10.0, "P": 0.0})

    model.add_reaction(
        "v1",
        fn=fns.mass_action_1s,
        args=["S", "k1"],
        stoichiometry={"S": -1.0, "P": 1.0},
    )

    model.add_reaction(
        "v2",
        fn=fns.mass_action_1s,
        args=["P", "k2"],
        stoichiometry={"P": -1.0},
    )

    return model


def test_update_parameters_and(simple_model: Model) -> None:
    params = pd.Series({"k1": 2.0})

    def get_params(model: Model) -> dict[str, float]:
        return model.get_parameter_values()

    result = _update_parameters_and_initial_conditions(params, get_params, simple_model)
    assert result["k1"] == 2.0
    assert result["k2"] == 2.0  # Unchanged


def test_steady_state_worker(simple_model: Model) -> None:
    result = _steady_state_worker(
        simple_model,
        rel_norm=False,
        integrator=DefaultIntegrator,
        y0=None,
    )

    # The model should reach steady state with S=0, P=0
    assert not np.isnan(result.variables["S"].iloc[-1])
    assert not np.isnan(result.variables["P"].iloc[-1])
    assert not np.isnan(result.fluxes["v1"].iloc[-1])
    assert not np.isnan(result.fluxes["v2"].iloc[-1])


def test_time_course_worker(simple_model: Model) -> None:
    time_points = np.linspace(0, 1, 3)
    result = _time_course_worker(
        simple_model,
        time_points=time_points,
        integrator=DefaultIntegrator,
        y0=None,
    )

    assert result.variables.shape == (3, 2)
    assert result.fluxes.shape == (3, 2)
    assert not np.isnan(result.variables.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_steady_state_scan(simple_model: Model) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0, 3.0]})

    result = steady_state(
        simple_model,
        to_scan=to_scan,
        parallel=False,
    )

    assert result.variables.shape == (3, 2)
    assert result.fluxes.shape == (3, 2)
    assert result.to_scan.equals(to_scan)
    assert not np.isnan(result.variables.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_steady_state_scan_with_multiindex(simple_model: Model) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0], "k2": [3.0, 4.0]})

    result = steady_state(
        simple_model,
        to_scan=to_scan,
        parallel=False,
    )

    assert result.variables.shape == (2, 2)
    assert result.fluxes.shape == (2, 2)
    assert isinstance(result.variables.index, pd.MultiIndex)
    assert isinstance(result.fluxes.index, pd.MultiIndex)
    assert not np.isnan(result.variables.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_time_course_scan(simple_model: Model) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0]})
    time_points = np.linspace(0, 1, 3)

    result = time_course(
        simple_model,
        to_scan=to_scan,
        time_points=time_points,
        parallel=False,
    )

    assert result.variables.shape == (6, 2)  # 2 params x 3 time points x 2 variables
    assert result.fluxes.shape == (6, 2)  # 2 params x 3 time points x 2 reactions
    assert isinstance(result.variables.index, pd.MultiIndex)
    assert isinstance(result.fluxes.index, pd.MultiIndex)
    assert result.variables.index.names == ["n", "time"]
    assert result.fluxes.index.names == ["n", "time"]
    assert not np.isnan(result.variables.values).any()
    assert not np.isnan(result.fluxes.values).any()
