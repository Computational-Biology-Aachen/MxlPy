# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Post-processing of the simulation result before the loss calculation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mxlpy import OdeModelBuilder, Simulator, SteadyStateModelBuilder, fit
from mxlpy.fit import losses
from mxlpy.fit.abstract import _Settings
from mxlpy.runners import EvaluationRunner, OdeRunner

TIME_POINTS = np.linspace(0, 5, 11)


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
    )


def relative_to_last(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise x by its last value, which depends on the whole time course."""
    return (df["x"] / df["x"].iloc[-1]).to_frame("x_rel")


def log(a: float) -> float:
    return math.log(a)


def raise_key_error(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ["missing"]]


def time_course_data(k: float) -> pd.DataFrame:
    return relative_to_last(
        Simulator(ode_model().update_parameter("k", k))
        .simulate_time_course(TIME_POINTS)
        .get_result()
        .unwrap_or_err()
        .get_combined()
    )


def time_course_settings(
    data: pd.DataFrame,
    postprocess_fn: fit.abstract.PostprocessFn | None,
) -> _Settings[OdeModelBuilder]:
    return _Settings(
        model=ode_model(),
        data=data,
        y0=None,
        runner=OdeRunner(),
        loss_fn=losses.rmse,
        p_names=["k"],
        v_names=[],
        standard_scale=False,
        postprocess_fn=postprocess_fn,
    )


def test_time_course_residual_applies_postprocess() -> None:
    settings = time_course_settings(time_course_data(0.25), relative_to_last)
    assert fit.time_course_residual({"k": 0.25}, settings) == pytest.approx(0.0)
    assert fit.time_course_residual({"k": 1.0}, settings) > 0.0


def test_time_course_residual_without_postprocess_unchanged() -> None:
    data = (
        Simulator(ode_model())
        .simulate_time_course(TIME_POINTS)
        .get_result()
        .unwrap_or_err()
        .get_combined()
    )
    settings = time_course_settings(data, None)
    assert fit.time_course_residual({"k": 0.5}, settings) == pytest.approx(0.0)


def test_steady_state_residual_applies_postprocess() -> None:
    settings = _Settings(
        model=ode_model(),
        data=pd.Series({"x2": 8.0}),
        y0=None,
        runner=OdeRunner(),
        loss_fn=losses.rmse,
        p_names=["k"],
        v_names=[],
        standard_scale=False,
        postprocess_fn=lambda df: (2 * df["x"]).to_frame("x2"),
    )
    assert fit.steady_state_residual({"k": 0.25}, settings) == pytest.approx(
        0.0, abs=1e-4
    )


def test_postprocess_exception_propagates() -> None:
    settings = time_course_settings(time_course_data(0.25), raise_key_error)
    with pytest.raises(KeyError):
        fit.time_course_residual({"k": 0.25}, settings)


def test_failed_simulation_skips_postprocess() -> None:
    model = SteadyStateModelBuilder().add_parameter("a", 1.0)
    model.add_derived("d", fn=log, args=["a"])
    settings = _Settings(
        model=model,
        data=pd.Series({"d": 0.0}),
        y0=None,
        runner=EvaluationRunner(),
        loss_fn=losses.rmse,
        p_names=["a"],
        v_names=[],
        standard_scale=False,
        postprocess_fn=raise_key_error,
    )
    assert fit.steady_state_residual({"a": -1.0}, settings) == np.inf


def test_fit_time_course_with_postprocess() -> None:
    res = fit.time_course(
        ode_model(),
        p0={"k": 1.0},
        data=time_course_data(0.25),
        minimizer=fit.LocalScipyMinimizer(),
        postprocess_fn=relative_to_last,
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)


def test_joint_time_course_per_settings_postprocess_overrides() -> None:
    res = fit.joint_time_course(
        [
            fit.FitSettings(
                model=ode_model(),
                data=time_course_data(0.25),
                postprocess_fn=relative_to_last,
            )
        ],
        p0={"k": 1.0},
        minimizer=fit.LocalScipyMinimizer(),
        postprocess_fn=raise_key_error,
        max_workers=1,
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)
