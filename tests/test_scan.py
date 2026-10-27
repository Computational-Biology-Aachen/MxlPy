# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    SteadyStateModelBuilder,
    cartesian_product,
    fns,
    make_protocol,
)
from mxlpy import scan as scan_module
from mxlpy.integrators import DefaultIntegrator
from mxlpy.scan import (
    EvaluationScan,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and_initial_conditions,
    protocol,
    protocol_time_course,
    steady_state,
    time_course,
)
from mxlpy.simulation import Simulation


@pytest.fixture
def simple_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
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


@pytest.fixture
def complex_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variables({"S": 10.0, "P": 0.0})
        .add_parameters({"k1": 1.0, "k2": 2.0})
        .add_readout("1/S", fn=fns.one_div, args=["S"])
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


def test_steady_state_scan_empty_dataframe(simple_model: KineticModelBuilder) -> None:
    """Empty to_scan must not crash and must return an empty result."""
    to_scan = pd.DataFrame({"k1": pd.Series([], dtype=float)})
    result = steady_state(simple_model, to_scan=to_scan, parallel=False)
    assert result.variables.empty
    assert result.fluxes.empty


def test_time_course_scan_empty_dataframe_raises(
    simple_model: KineticModelBuilder,
) -> None:
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


def test_steady_state_scan_nan_in_to_scan_raises(
    simple_model: KineticModelBuilder,
) -> None:
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
    simple_model: KineticModelBuilder,
) -> None:
    """Column named after a variable updates that variable, not a parameter.

    When the scan DataFrame has a column matching a *variable* name (e.g. "S"),
    it must be applied as an initial condition, not confused with a parameter.
    """
    from mxlpy.scan import _update_parameters_and_initial_conditions

    def get_ic(model: KineticModelBuilder) -> dict[str, float]:
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


def test_protocolscan_combined(complex_model: KineticModelBuilder) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0, 3.0]})

    result = protocol(
        complex_model,
        to_scan=to_scan,
        protocol=make_protocol([(1, {"k2": 2}), (2, {"k2": 4})]),
        parallel=False,
    ).combined

    assert Counter(list(result.columns)) == Counter(
        complex_model.get_variable_names()
        + complex_model.get_readout_names()
        + complex_model.get_reaction_names()
    )


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


def test_steady_state_scan(simple_model: KineticModelBuilder) -> None:
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


def test_steady_state_scan_with_multiindex(simple_model: KineticModelBuilder) -> None:
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


def test_steady_state_worker(simple_model: KineticModelBuilder) -> None:
    result = _steady_state_worker(
        simple_model,
        rel_norm=False,
        integrator=DefaultIntegrator,
        y0=None,
    )

    assert isinstance(result, Simulation)
    # The model should reach steady state with S=0, P=0
    assert not np.isnan(result.variables["S"].iloc[-1])
    assert not np.isnan(result.variables["P"].iloc[-1])
    assert not np.isnan(result.fluxes["v1"].iloc[-1])
    assert not np.isnan(result.fluxes["v2"].iloc[-1])


def test_steadystatescan_combined(complex_model: KineticModelBuilder) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0, 3.0]})

    result = steady_state(
        complex_model,
        to_scan=to_scan,
        parallel=False,
    ).combined

    assert Counter(list(result.columns)) == Counter(
        complex_model.get_variable_names()
        + complex_model.get_readout_names()
        + complex_model.get_reaction_names()
    )


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


def test_time_course_scan(simple_model: KineticModelBuilder) -> None:
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


def test_time_course_worker(simple_model: KineticModelBuilder) -> None:
    time_points = np.linspace(0, 1, 3)
    result = _time_course_worker(
        simple_model,
        time_points=time_points,
        integrator=DefaultIntegrator,
        y0=None,
    )

    assert isinstance(result, Simulation)
    assert result.variables.shape == (3, 2)
    assert result.fluxes.shape == (3, 2)
    assert not np.isnan(result.variables.values).any()
    assert not np.isnan(result.fluxes.values).any()


def test_timecoursescan_combined(complex_model: KineticModelBuilder) -> None:
    to_scan = pd.DataFrame({"k1": [1.0, 2.0]})
    time_points = np.linspace(0, 1, 3)

    result = time_course(
        complex_model,
        to_scan=to_scan,
        time_points=time_points,
        parallel=False,
    ).combined

    assert Counter(list(result.columns)) == Counter(
        complex_model.get_variable_names()
        + complex_model.get_readout_names()
        + complex_model.get_reaction_names()
    )


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


def test_update_parameters_and(simple_model: KineticModelBuilder) -> None:
    params = pd.Series({"k1": 2.0})

    def get_params(model: KineticModelBuilder) -> dict[str, float]:
        return model.get_parameter_values()

    result = _update_parameters_and_initial_conditions(params, get_params, simple_model)
    assert result["k1"] == 2.0
    assert result["k2"] == 2.0  # Unchanged


def ode_model() -> OdeModelBuilder:
    """dx/dt = k_in - k * x, with steady state x = k_in / k."""
    return (
        OdeModelBuilder()
        .add_parameter("k_in", 1.0)
        .add_parameter("k", 0.5)
        .add_diff_eq(
            "x",
            fn=lambda k_in, x, k: k_in - k * x,
            args=["k_in", "x", "k"],
            initial_value=0.0,
        )
        .add_derived("d", fn=lambda x: 2 * x, args=["x"])
    )


def test_steady_state_ode_model() -> None:
    res = steady_state(
        ode_model(),
        to_scan=pd.DataFrame({"k": [0.5, 1.0, 2.0]}),
        parallel=False,
    )
    np.testing.assert_allclose(res.variables["x"], [2.0, 1.0, 0.5], rtol=1e-4)
    np.testing.assert_allclose(res.combined["d"], [4.0, 2.0, 1.0], rtol=1e-4)
    assert "k" in res.get_args(include_parameters=True).columns


def test_time_course_ode_model_initial_conditions() -> None:
    res = time_course(
        ode_model(),
        to_scan=pd.DataFrame({"x": [0.0, 5.0]}),
        time_points=np.linspace(0, 1, 3),
        parallel=False,
    )
    assert res.variables.loc[(0, 0.0), "x"] == 0.0
    assert res.variables.loc[(1, 0.0), "x"] == pytest.approx(5.0)


def test_steady_state_steady_state_model() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("k", 2.0)
        .add_derived("d", fn=lambda k: 1 / k, args=["k"])
    )
    res = steady_state(
        model,
        to_scan=pd.DataFrame({"k": [1.0, 0.0, 4.0]}),
        parallel=False,
    )
    assert isinstance(res, EvaluationScan)
    assert list(res.combined.columns) == ["d"]
    np.testing.assert_array_equal(res.combined["d"], [1.0, np.nan, 0.25])
    assert list(res.get_args(include_parameters=True).columns) == ["k", "d"]


def ratio(a: float, b: float) -> float:
    return a / b


def log(a: float) -> float:
    return math.log(a)


def test_steady_state_steady_state_model_runs_serially_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = scan_module.parallelise

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["parallel"])
        return original(*args, **kwargs)

    monkeypatch.setattr(scan_module, "parallelise", spy)
    model = (
        SteadyStateModelBuilder()
        .add_parameters({"a": 1.0, "b": 2.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
    )
    steady_state(model, to_scan=pd.DataFrame({"a": [1.0, 2.0]}))
    assert calls == [False]


def test_steady_state_steady_state_model_domain_error_is_nan() -> None:
    model = SteadyStateModelBuilder().add_parameter("a", 1.0)
    model.add_derived("d", fn=log, args=["a"])
    res = steady_state(model, to_scan=pd.DataFrame({"a": [1.0, -1.0]}))
    np.testing.assert_array_equal(res.combined["d"], [0.0, np.nan])


def test_steady_state_steady_state_model_multiindex() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameters({"a": 1.0, "b": 2.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
    )
    res = steady_state(
        model, to_scan=cartesian_product({"a": [1.0, 2.0], "b": [1.0, 4.0]})
    )
    assert res.combined.index.names == ["a", "b"]
    assert res.combined.loc[(2.0, 4.0), "d"] == 0.5


def test_ode_scan_fluxes_raise_helpful_error() -> None:
    res = steady_state(
        ode_model(), to_scan=pd.DataFrame({"k": [0.5, 1.0]}), parallel=False
    )
    with pytest.raises(TypeError, match="no fluxes"):
        _ = res.fluxes  # type: ignore[misc]
    with pytest.raises(TypeError, match="no fluxes"):
        _, _ = res  # type: ignore[misc]


def test_protocol_time_course_ode_model() -> None:
    res = protocol_time_course(
        ode_model(),
        to_scan=pd.DataFrame({"k": [0.5, 1.0]}),
        protocol=make_protocol([(2, {"k_in": 1.0}), (2, {"k_in": 3.0})]),
        time_points=np.linspace(0, 4, 11),
        parallel=False,
    )
    assert res.variables.shape == (2 * 11, 2)
    assert not res.variables.isna().any().any()
