# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting of the non-kinetic model builders."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    OdeModelBuilder,
    Simulator,
    SteadyStateModelBuilder,
    fit,
    make_protocol,
)
from mxlpy.fit import losses
from mxlpy.fit.abstract import _Settings


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
    res = fit.steady_state(
        ode_model(),
        p0={"k": 1.0},
        data=pd.Series({"x": 4.0, "d": 8.0}),
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap_or_err()
    assert isinstance(res.model, OdeModelBuilder)
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)


def test_time_course_ode_model() -> None:
    data = (
        Simulator(ode_model().update_parameter("k", 0.25))
        .simulate_time_course(np.linspace(0, 5, 11))
        .get_result()
        .unwrap_or_err()
        .get_combined()
    )
    res = fit.time_course(
        ode_model(),
        p0={"k": 1.0},
        data=data,
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)


def test_steady_state_steady_state_model() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("k", 1.0)
        .add_derived("d1", fn=lambda k: 2 * k, args=["k"])
        .add_derived("d2", fn=lambda k: k**2, args=["k"])
    )
    res = fit.steady_state(
        model,
        p0={"k": 1.0},
        data=pd.Series({"d1": 6.0, "d2": 9.0}),
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap_or_err()
    assert isinstance(res.model, SteadyStateModelBuilder)
    assert res.best_pars["k"] == pytest.approx(3.0, rel=1e-3)


def ratio(a: float, b: float) -> float:
    return a / b


def log(a: float) -> float:
    return math.log(a)


def test_steady_state_single_data_point_skips_scaling() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameters({"a": 2.0, "b": 2.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
    )
    res = fit.steady_state(
        model,
        p0={"a": 2.0},
        data=pd.Series({"d": 0.25}),
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap_or_err()
    assert res.best_pars["a"] == pytest.approx(0.5, rel=1e-3)


def test_protocol_time_course_ode_model() -> None:
    protocol = make_protocol([(2, {"k_in": 1.0}), (2, {"k_in": 3.0})])
    time_points = np.linspace(0, 4, 21)
    data = (
        Simulator(ode_model().update_parameter("k", 0.25))
        .simulate_protocol_time_course(protocol, time_points=time_points)
        .get_result()
        .unwrap_or_err()
        .get_combined()
    )
    res = fit.protocol_time_course(
        ode_model(),
        p0={"k": 1.0},
        data=data,
        protocol=protocol,
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)


def test_evaluate_residual_domain_error_is_inf() -> None:
    model = SteadyStateModelBuilder().add_parameter("a", 1.0)
    model.add_derived("d", fn=log, args=["a"])
    settings = _Settings(
        model=model,
        data=pd.Series({"d": 0.0}),
        y0=None,
        integrator=None,
        loss_fn=losses.rmse,
        p_names=["a"],
        v_names=[],
        standard_scale=False,
    )
    assert fit.evaluate_residual({"a": -1.0}, settings) == np.inf


def test_evaluate_residual_rejects_integrable_model() -> None:
    settings = _Settings(
        model=ode_model(),
        data=pd.Series({"x": 1.0}),
        y0=None,
        integrator=None,
        loss_fn=losses.rmse,
        p_names=["k"],
        v_names=[],
        standard_scale=False,
    )
    with pytest.raises(TypeError, match="steady_state_residual"):
        fit.evaluate_residual({"k": 1.0}, settings)  # type: ignore[arg-type]


def test_simulating_residual_rejects_steady_state_model() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameters({"a": 2.0, "b": 2.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
    )
    with pytest.raises(TypeError, match="evaluate"):
        fit.steady_state(
            model,
            p0={"a": 2.0},
            data=pd.Series({"d": 0.25}),
            minimizer=fit.LocalScipyMinimizer(),
            residual_fn=fit.steady_state_residual,
        )
