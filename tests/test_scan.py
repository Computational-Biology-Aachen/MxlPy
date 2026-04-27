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
    return (
        Model()
        .add_variables({"S": 10.0, "P": 0.0})
        .add_parameters({"k1": 1.0, "k2": 2.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1.0, "P": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["P", "k2"],
            stoichiometry={"P": -1.0},
        )
    )


def test_steady_state_scan_empty_dataframe(simple_model: Model) -> None:
    """Empty to_scan must not crash and must return an empty result."""
    to_scan = pd.DataFrame({"k1": pd.Series([], dtype=float)})
    result = steady_state(simple_model, to_scan=to_scan, parallel=False)
    assert result.variables.empty
    assert result.fluxes.empty


def test_time_course_scan_empty_dataframe_raises(simple_model: Model) -> None:
    """Empty to_scan for time course scan raises ValueError on result access.

    The scan itself completes, but accessing ``.variables`` on the result calls
    ``pd.concat([])`` which raises ValueError.  This documents current behavior:
    callers must guard against empty to_scan DataFrames.
    """
    to_scan = pd.DataFrame({"k1": pd.Series([], dtype=float)})
    time_points = np.linspace(0, 1, 3)
    result = time_course(
        simple_model, to_scan=to_scan, time_points=time_points, parallel=False
    )
    with pytest.raises(ValueError, match="No objects to concatenate"):
        _ = result.variables


def test_steady_state_scan_nan_in_to_scan_raises(simple_model: Model) -> None:
    """NaN parameter causes the ODE to diverge, y0 becomes NaN, scipy raises.

    ``update_parameter("k1", NaN)`` is accepted silently by the model, but
    the NaN RHS makes the integrator produce NaN states.  On the next
    integration step scipy rejects the NaN y0 with ValueError.

    ``_steady_state_worker`` only catches ``ZeroDivisionError``, so the
    ValueError propagates to the caller.  Callers must validate to_scan inputs
    before running a scan.
    """
    to_scan = pd.DataFrame({"k1": [1.0, float("nan"), 3.0]})
    with pytest.raises(ValueError):
        steady_state(simple_model, to_scan=to_scan, parallel=False)


def test_update_parameters_and_initial_conditions_variable_vs_parameter(
    simple_model: Model,
) -> None:
    """Column named after a variable updates that variable, not a parameter.

    When the scan DataFrame has a column matching a *variable* name (e.g. "S"),
    it must be applied as an initial condition, not confused with a parameter.
    """
    from mxlpy.scan import _update_parameters_and_initial_conditions

    def get_ic(model: Model) -> dict[str, float]:
        return model.get_initial_conditions()

    pars = pd.Series({"S": 5.0, "k1": 2.0})
    ic = _update_parameters_and_initial_conditions(pars, get_ic, simple_model)

    assert ic["S"] == pytest.approx(5.0)
    assert simple_model.get_parameter_values()["k1"] == pytest.approx(2.0)


def test_protocol() -> None:
    # FIXME: implement this
    assert True


def test_protocol_time_course() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_combined() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_fluxes() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_get_agg_per_run() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_get_agg_per_time() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_get_args() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_get_by_name() -> None:
    # FIXME: implement this
    assert True


def test_protocolscan_variables() -> None:
    # FIXME: implement this
    assert True


def test_steady_state() -> None:
    # FIXME: implement this
    assert True


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


def test_steadystatescan_combined() -> None:
    # FIXME: implement this
    assert True


def test_steadystatescan_fluxes() -> None:
    # FIXME: implement this
    assert True


def test_steadystatescan_get_args() -> None:
    # FIXME: implement this
    assert True


def test_steadystatescan_variables() -> None:
    # FIXME: implement this
    assert True


def test_time_course() -> None:
    # FIXME: implement this
    assert True


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


def test_timecoursescan_combined() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_fluxes() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_get_agg_per_run() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_get_agg_per_time() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_get_args() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_get_by_name() -> None:
    # FIXME: implement this
    assert True


def test_timecoursescan_variables() -> None:
    # FIXME: implement this
    assert True


def test_update_parameters_and(simple_model: Model) -> None:
    params = pd.Series({"k1": 2.0})

    def get_params(model: Model) -> dict[str, float]:
        return model.get_parameter_values()

    result = _update_parameters_and_initial_conditions(params, get_params, simple_model)
    assert result["k1"] == 2.0
    assert result["k2"] == 2.0  # Unchanged
