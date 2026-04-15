"""Edge case tests for parameter scanning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, fns
from mxlpy.scan import steady_state, time_course


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
