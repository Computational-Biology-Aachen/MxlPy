from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import Model, fns
from mxlpy.integrators import DefaultIntegrator
from mxlpy.scan import (
    TimeCourse,
    TimePoint,
    _empty_conc_df,
    _empty_conc_series,
    _empty_flux_df,
    _empty_flux_series,
    _steady_state_worker,
    _time_course_worker,
    _update_parameters_and_initial_conditions,
    steady_state,
    time_course,
)
from mxlpy.simulator import Result


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


def test_empty_conc_series(simple_model: Model) -> None:
    series = _empty_conc_series(simple_model)
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert all(np.isnan(series))
    assert list(series.index) == ["S", "P"]


def test_empty_flux_series(simple_model: Model) -> None:
    series = _empty_flux_series(simple_model)
    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert all(np.isnan(series))
    assert list(series.index) == ["v1", "v2"]


def test_empty_conc_df(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    df = _empty_conc_df(simple_model, time_points)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert np.all(np.isnan(df.to_numpy()))
    assert np.all(df.index == time_points)
    assert np.all(df.columns == ["S", "P"])


def test_empty_flux_df(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    df = _empty_flux_df(simple_model, time_points)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert np.all(np.isnan(df.to_numpy()))
    assert np.all(df.index == time_points)
    assert np.all(df.columns == ["v1", "v2"])


def test_update_parameters_and(simple_model: Model) -> None:
    params = pd.Series({"k1": 2.0})

    def get_params(model: Model) -> dict[str, float]:
        return model.get_parameter_values()

    result = _update_parameters_and_initial_conditions(params, get_params, simple_model)
    assert result["k1"] == 2.0
    assert result["k2"] == 2.0  # Unchanged


def test_timepoint_from_scan(simple_model: Model) -> None:
    time_point = TimePoint.from_result(model=simple_model, result=None)
    assert isinstance(time_point, TimePoint)
    assert isinstance(time_point.variables, pd.Series)
    assert isinstance(time_point.fluxes, pd.Series)
    assert time_point.variables.index.tolist() == ["S", "P"]
    assert time_point.fluxes.index.tolist() == ["v1", "v2"]


def test_timepoint_with_data(simple_model: Model) -> None:
    result = Result(
        model=simple_model,
        raw_variables=[pd.DataFrame({"S": [1.0, 2.0], "P": [3.0, 4.0]})],
        raw_parameters=[simple_model.get_parameter_values()],
    )

    time_point = TimePoint.from_result(model=simple_model, result=result, idx=1)
    pd.testing.assert_series_equal(
        time_point.variables,
        pd.Series(
            {"S": 2.0, "P": 4.0},
            index=["S", "P"],
            dtype=float,
            name=1,
        ),
    )
    pd.testing.assert_series_equal(
        time_point.fluxes,
        pd.Series(
            {"v1": 2.0, "v2": 8.0},
            index=["v1", "v2"],
            dtype=float,
            name=1,
        ),
    )


def test_timepoint_results(simple_model: Model) -> None:
    result = Result(
        model=simple_model,
        raw_variables=[pd.DataFrame({"S": [1.0], "P": [3.0]}, dtype=float)],
        raw_parameters=[simple_model.get_parameter_values()],
    )

    time_point = TimePoint.from_result(model=simple_model, result=result, idx=0)
    results = time_point.results

    pd.testing.assert_series_equal(
        results,
        pd.Series(
            {"S": 1.0, "P": 3.0, "v1": 1.0, "v2": 6.0},
            dtype=float,
            name=0,
        ),
    )


def test_time_course_from_scan(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    time_course = TimeCourse.from_scan(
        model=simple_model, time_points=time_points, result=None
    )

    assert isinstance(time_course, TimeCourse)
    assert isinstance(time_course.variables, pd.DataFrame)
    assert isinstance(time_course.fluxes, pd.DataFrame)
    assert time_course.variables.shape == (3, 2)
    assert time_course.fluxes.shape == (3, 2)
    assert np.all(time_course.variables.index == time_points)
    assert np.all(time_course.fluxes.index == time_points)
    assert np.all(time_course.variables.columns == ["S", "P"])
    assert np.all(time_course.fluxes.columns == ["v1", "v2"])


def test_timecourse_with_data(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)
    result = Result(
        model=simple_model,
        raw_variables=[
            pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                index=time_points,
                columns=["S", "P"],
            )
        ],
        raw_parameters=[simple_model.get_parameter_values()],
    )

    time_course = TimeCourse.from_scan(
        model=simple_model, time_points=time_points, result=result
    )

    pd.testing.assert_frame_equal(
        time_course.variables,
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=time_points,
            columns=["S", "P"],
            dtype=float,
        ),
    )
    pd.testing.assert_frame_equal(
        time_course.fluxes,
        pd.DataFrame(
            [[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]],
            index=time_points,
            columns=["v1", "v2"],
            dtype=float,
        ),
    )


def test_timecourse_results(simple_model: Model) -> None:
    time_points = np.array([0.0, 1.0, 2.0], dtype=float)

    result = Result(
        model=simple_model,
        raw_variables=[
            pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                index=time_points,
                columns=["S", "P"],
            )
        ],
        raw_parameters=[simple_model.get_parameter_values()],
    )

    time_course = TimeCourse.from_scan(
        model=simple_model, time_points=time_points, result=result
    )
    results = time_course.results

    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            {
                "S": [1.0, 3.0, 5.0],
                "P": [2.0, 4.0, 6.0],
                "v1": [1.0, 3.0, 5.0],
                "v2": [4.0, 8.0, 12.0],
            },
            index=[0.0, 1.0, 2.0],
            dtype=float,
        ),
    )


def test_steady_state_worker(simple_model: Model) -> None:
    result = _steady_state_worker(
        simple_model,
        rel_norm=False,
        integrator=DefaultIntegrator,
        y0=None,
    )
    assert isinstance(result, TimePoint)

    # The model should reach steady state with S=0, P=0
    assert not np.isnan(result.variables["S"])
    assert not np.isnan(result.variables["P"])
    assert not np.isnan(result.fluxes["v1"])
    assert not np.isnan(result.fluxes["v2"])


def test_time_course_worker(simple_model: Model) -> None:
    time_points = np.linspace(0, 1, 3)
    result = _time_course_worker(
        simple_model,
        time_points=time_points,
        integrator=DefaultIntegrator,
        y0=None,
    )

    assert isinstance(result, TimeCourse)
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
    assert result.parameters.equals(to_scan)
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
