# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Support of OdeModelBuilder and SteadyStateModelBuilder across the library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    SteadyStateModelBuilder,
    fit,
    fns,
    mc,
    mca,
    scan,
    sensitivity,
)
from mxlpy.simulation import Evaluation, OdeSimulation


def constant(k: float) -> float:
    return k


def decay(k_in: float, x: float, k: float) -> float:
    return k_in - k * x


def ratio(a: float, b: float) -> float:
    return a / b


def double(a: float) -> float:
    return 2 * a


def kinetic_model() -> KineticModelBuilder:
    """Steady state x = k_in / k."""
    return (
        KineticModelBuilder()
        .add_parameters({"k_in": 1.0, "k": 0.5})
        .add_variable("x", 0.0)
        .add_reaction("v_in", fn=constant, args=["k_in"], stoichiometry={"x": 1})
        .add_reaction(
            "v_out", fn=fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1}
        )
    )


def ode_model() -> OdeModelBuilder:
    """Steady state x = k_in / k."""
    return (
        OdeModelBuilder()
        .add_parameters({"k_in": 1.0, "k": 0.5})
        .add_diff_eq("x", fn=decay, args=["k_in", "x", "k"], initial_value=0.0)
    )


def ss_model() -> SteadyStateModelBuilder:
    return (
        SteadyStateModelBuilder()
        .add_parameters({"a": 1.0, "b": 2.0, "k_in": 1.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
        .add_derived("e", fn=double, args=["k_in"])
    )


###############################################################################
# Fits
###############################################################################


def test_group_steady_state_ode_and_steady_state_models() -> None:
    res = fit.group_steady_state(
        ode_model(),
        p0=pd.DataFrame({"k": [1.0, 2.0]}),
        data=pd.Series({"x": 4.0}),
        minimizer=fit.LocalScipyMinimizer(),
        parallel=False,
    )
    assert isinstance(res.get_best_fit().model, OdeModelBuilder)
    assert res.get_best_fit().best_pars["k"] == pytest.approx(0.25, rel=1e-2)

    res_ss = fit.group_steady_state(
        ss_model(),
        p0=pd.DataFrame({"a": [1.0, 2.0]}),
        data=pd.Series({"d": 0.25}),
        minimizer=fit.LocalScipyMinimizer(),
        parallel=False,
    )
    assert res_ss.get_best_fit().best_pars["a"] == pytest.approx(0.5, rel=1e-3)


def test_group_time_course_ode_model() -> None:
    data = (
        scan.time_course(
            ode_model().update_parameter("k", 0.25),
            to_scan=pd.DataFrame({"k_in": [1.0]}),
            time_points=np.linspace(0, 5, 11),
            parallel=False,
        )
        .get_by_name("x")
        .rename(columns={0: "x"})
    )
    res = fit.group_time_course(
        ode_model(),
        p0=pd.DataFrame({"k": [1.0, 0.5]}),
        data=data,
        minimizer=fit.LocalScipyMinimizer(),
        parallel=False,
    )
    assert res.get_best_fit().best_pars["k"] == pytest.approx(0.25, rel=1e-2)


def test_ensemble_steady_state_steady_state_models() -> None:
    res = fit.ensemble_steady_state(
        [ss_model(), ss_model().update_parameter("b", 4.0)],
        p0={"a": 1.0},
        data=pd.Series({"d": 0.25}),
        minimizer=fit.LocalScipyMinimizer(),
        parallel=False,
    )
    assert res.get_best_fit().loss == pytest.approx(0.0, abs=1e-6)


def test_joint_steady_state_mixed_builders() -> None:
    # Both models share k_in, which is 2 in both datasets
    res = fit.joint_steady_state(
        [
            fit.FitSettings(model=kinetic_model(), data=pd.Series({"x": 4.0})),
            fit.FitSettings(model=ss_model(), data=pd.Series({"e": 4.0})),
        ],
        p0={"k_in": 1.0},
        minimizer=fit.LocalScipyMinimizer(),
        y0={"x": 0.0},  # skipped for the steady-state model
        parallel=False,
    ).unwrap_or_err()
    assert res.best_pars["k_in"] == pytest.approx(2.0, rel=1e-3)


def test_joint_mixed_serial() -> None:
    res = fit.joint_mixed(
        [
            fit.MixedSettings(
                model=kinetic_model(),
                data=pd.Series({"x": 4.0}),
                residual_fn=fit.steady_state_residual,
                y0={"x": 0.0},
            ),
            fit.MixedSettings(
                model=ss_model(),
                data=pd.Series({"e": 4.0}),
                residual_fn=fit.steady_state_residual,
            ),
        ],
        p0={"k_in": 1.0},
        minimizer=fit.LocalScipyMinimizer(),
        parallel=False,
    ).unwrap_or_err()
    assert res.best_pars["k_in"] == pytest.approx(2.0, rel=1e-3)


def test_joint_time_course_rejects_steady_state_model() -> None:
    with pytest.raises(TypeError, match="steady-state data"):
        fit.joint_time_course(
            [fit.FitSettings(model=ss_model(), data=pd.DataFrame({"d": [0.5]}))],
            p0={"a": 1.0},
            minimizer=fit.LocalScipyMinimizer(),
        )


def test_joint_rejects_y0_for_steady_state_model() -> None:
    with pytest.raises(TypeError, match="y0"):
        fit.joint_steady_state(
            [
                fit.FitSettings(
                    model=ss_model(), data=pd.Series({"d": 0.5}), y0={"x": 1.0}
                )
            ],
            p0={"a": 1.0},
            minimizer=fit.LocalScipyMinimizer(),
        )


###############################################################################
# Monte Carlo
###############################################################################


def test_mc_steady_state_per_builder() -> None:
    mc_to_scan = pd.DataFrame({"k": [0.5, 1.0]})
    res = mc.steady_state(ode_model(), mc_to_scan=mc_to_scan, parallel=False)
    assert all(isinstance(i, OdeSimulation) for i in res.raw_results)
    np.testing.assert_allclose(res.variables["x"], [2.0, 1.0], rtol=1e-4)

    res_ss = mc.steady_state(
        ss_model(), mc_to_scan=pd.DataFrame({"b": [2.0, 4.0]}), parallel=False
    )
    assert all(isinstance(i, Evaluation) for i in res_ss.raw_results)
    np.testing.assert_allclose(res_ss.combined["d"], [0.5, 0.25])


def test_mc_time_course_ode_model() -> None:
    res = mc.time_course(
        ode_model(),
        time_points=np.linspace(0, 1, 3),
        mc_to_scan=pd.DataFrame({"k": [0.5, 1.0]}),
        parallel=False,
    )
    assert res.variables.shape == (6, 1)


def test_mc_scan_steady_state_steady_state_model() -> None:
    res = mc.scan_steady_state(
        ss_model(),
        to_scan=pd.DataFrame({"a": [1.0, 2.0]}),
        mc_to_scan=pd.DataFrame({"b": [2.0, 4.0]}),
        parallel=False,
    )
    np.testing.assert_allclose(res.combined["d"], [0.5, 1.0, 0.25, 0.5])
    with pytest.raises(TypeError, match="no variables"):
        _ = res.variables  # type: ignore[misc]


def test_mc_scan_steady_state_kinetic_model_keeps_split() -> None:
    res = mc.scan_steady_state(
        kinetic_model(),
        to_scan=pd.DataFrame({"k": [0.5, 1.0]}),
        mc_to_scan=pd.DataFrame({"k_in": [1.0, 2.0]}),
        parallel=False,
    )
    assert list(res.variables.columns) == ["x"]
    assert list(res.fluxes.columns) == ["v_in", "v_out"]
    np.testing.assert_allclose(res.variables["x"], [2.0, 1.0, 4.0, 2.0], rtol=1e-4)


###############################################################################
# MCA
###############################################################################


def test_response_coefficients_kinetic_model_combined_has_no_gaps() -> None:
    res = mca.response_coefficients(kinetic_model(), parallel=False)
    assert list(res.combined.index) == ["x", "v_in", "v_out"]
    assert not res.combined.isna().any().any()
    assert res.variables.loc["x", "k"] == pytest.approx(-1.0, rel=1e-3)


def test_response_coefficients_ode_model() -> None:
    res = mca.response_coefficients(ode_model(), parallel=False)
    assert res.variables.loc["x", "k"] == pytest.approx(-1.0, rel=1e-3)
    assert res.variables.loc["x", "k_in"] == pytest.approx(1.0, rel=1e-3)
    with pytest.raises(TypeError, match="no fluxes"):
        _ = res.fluxes  # type: ignore[misc]


def test_response_coefficients_steady_state_model() -> None:
    res = mca.response_coefficients(ss_model(), to_scan=["a", "b"], parallel=False)
    # d = a / b, so the normalised responses are exactly 1 and -1
    assert res.combined.loc["d", "a"] == pytest.approx(1.0, rel=1e-6)
    assert res.combined.loc["d", "b"] == pytest.approx(-1.0, rel=1e-6)
    with pytest.raises(TypeError, match="no variables"):
        _ = res.variables  # type: ignore[misc]


def test_mc_response_coefficients_steady_state_model() -> None:
    res = mc.response_coefficients(
        ss_model(),
        mc_to_scan=pd.DataFrame({"b": [2.0, 4.0]}),
        to_scan=["a"],
        parallel=False,
    )
    np.testing.assert_allclose(res.combined.xs("d", level=-1)["a"], [1.0, 1.0])


###############################################################################
# Sensitivity
###############################################################################


def ss_output(model: SteadyStateModelBuilder, samples: pd.DataFrame) -> np.ndarray:
    return scan.steady_state(model, to_scan=samples).combined["d"].to_numpy()


def test_morris_steady_state_model() -> None:
    res = sensitivity.morris(
        ss_model(),
        output=ss_output,
        param_bounds={"a": (0.5, 2.0), "k_in": (0.5, 2.0)},
        n_trajectories=4,
        seed=0,
    )
    # d only depends on a
    assert res.loc["a", "mu_star"] > 0
    assert res.loc["k_in", "mu_star"] == pytest.approx(0.0, abs=1e-12)
