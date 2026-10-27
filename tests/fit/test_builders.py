# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting of the non-kinetic model builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import OdeModelBuilder, Simulator, SteadyStateModelBuilder, fit


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
        # L-BFGS-B's finite differences stall on steady-state residuals
        minimizer=fit.LocalScipyMinimizer(method="Nelder-Mead"),
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
