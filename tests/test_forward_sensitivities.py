# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import functools
import math

import numpy as np
import pandas as pd
import pytest

import mxlpy.forward_sensitivities._core
from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    fns,
    forward_sensitivities,
)
from mxlpy.integrators import Scipy
from mxlpy.runners import KineticRunner, OdeRunner
from mxlpy.symbolic import jacobian_fn
from mxlpy.symbolic.jacobian import compile_derivatives
from mxlpy.types import Derived, InitialAssignment

T = np.array([0.5, 1.0, 2.0, 4.0])


def _chain() -> KineticModelBuilder:
    """x -> y -> ; x(t) = 2 exp(-k1 t)."""
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["x", "k1"],
            stoichiometry={"x": -1, "y": 1},
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


def _y_dk2(
    t: np.ndarray, k1: float = 0.5, k2: float = 2.0, x0: float = 2.0
) -> np.ndarray:
    """d/dk2 of y(t) = x0 k1 / (k2 - k1) (exp(-k1 t) - exp(-k2 t))."""
    d = k2 - k1
    return (
        x0
        * k1
        * (-(np.exp(-k1 * t) - np.exp(-k2 * t)) / d**2 + t * np.exp(-k2 * t) / d)
    )


def _reversible() -> KineticModelBuilder:
    """A <-> B with A + B conserved."""
    return (
        KineticModelBuilder()
        .add_parameters({"kf": 1.0, "kr": 0.5})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "kf"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
    )


def test_compile_derivatives_parameters_and_outputs() -> None:
    compiled = compile_derivatives(_chain(), wrt=["k1", "k2"], outputs=["v1", "v2"])
    values = compiled.fn(
        0.0, np.array([1.0, 0.5]), np.array([1.0, 2.0]), compiled.constants
    )
    entries = dict(
        zip(
            compiled.rhs_nonzero + [(-1 - o, k) for o, k in compiled.out_nonzero],
            values,
            strict=True,
        )
    )
    # columns: x, y, k1, k2
    assert entries[(0, 2)] == pytest.approx(-1.0)  # d(-k1 x)/dk1
    assert entries[(1, 3)] == pytest.approx(-0.5)  # d(k1 x - k2 y)/dk2
    assert entries[(-1, 2)] == pytest.approx(1.0)  # dv1/dk1 = x
    assert entries[(-2, 3)] == pytest.approx(0.5)  # dv2/dk2 = y


def test_compile_derivatives_rejects_unknown_wrt() -> None:
    with pytest.raises(ValueError, match="parameters"):
        compile_derivatives(_chain(), wrt=["x"])


@pytest.mark.parametrize("method", ["forward", "finite_difference"])
def test_time_course_matches_analytic(method: str) -> None:
    sens = forward_sensitivities.time_course(
        _chain(),
        T,
        ["k1", "k2", "x"],
        method=method,  # type: ignore[arg-type]
    ).unwrap_or_err()
    assert sens.method == method
    assert sens.wrt == ["k1", "k2", "x"]
    assert list(sens.simulated.columns) == ["x", "y", "v1", "v2"]
    np.testing.assert_allclose(sens.simulated.index, T)

    tol = 1e-6 if method == "forward" else 1e-5
    s = sens.derivatives
    np.testing.assert_allclose(s["k1"]["x"], -T * 2.0 * np.exp(-0.5 * T), atol=tol)
    np.testing.assert_allclose(s["x"]["x"], np.exp(-0.5 * T), atol=tol)
    np.testing.assert_allclose(s["k2"]["y"], _y_dk2(T), atol=tol)
    np.testing.assert_allclose(s["k2"]["x"], 0.0, atol=tol)
    # dv1/dk1 = x + k1 dx/dk1
    np.testing.assert_allclose(
        s["k1"]["v1"], sens.simulated["x"] + 0.5 * s["k1"]["x"], atol=tol
    )
    assert sens.to_array().shape == (len(T), 4, 3)


def test_time_course_outputs_subset() -> None:
    sens = forward_sensitivities.time_course(
        _chain(), T, ["k1"], outputs=["v2"]
    ).unwrap_or_err()
    assert list(sens.simulated.columns) == ["x", "y", "v2"]
    assert list(sens.derivatives["k1"].columns) == ["x", "y", "v2"]


def test_steady_state_with_conservation() -> None:
    model = _reversible()
    fw = forward_sensitivities.steady_state(model, ["kf", "kr", "A"]).unwrap_or_err()
    fd = forward_sensitivities.steady_state(
        model, ["kf", "kr", "A"], method="finite_difference"
    ).unwrap_or_err()
    # A* = kr / (kf + kr) * (A0 + B0)
    assert fw.derivatives["kf"]["A"].iloc[0] == pytest.approx(-0.5 / 1.5**2)
    assert fw.derivatives["kr"]["A"].iloc[0] == pytest.approx(1.0 / 1.5**2)
    assert fw.derivatives["A"]["A"].iloc[0] == pytest.approx(0.5 / 1.5)
    for k in fw.wrt:
        np.testing.assert_allclose(fw.derivatives[k], fd.derivatives[k], atol=1e-5)


def test_initial_assignments_are_differentiated() -> None:
    # x0 = 2 s, k2 = k s, x(t) = 2 s exp(-k s t)
    model = (
        KineticModelBuilder()
        .add_parameters(
            {"k": 1.0, "s": 2.0, "k2": InitialAssignment(fn=fns.mul, args=["k", "s"])}
        )
        .add_variables({"x": InitialAssignment(fn=fns.twice, args=["s"])})
        .add_reaction(
            "v", fns.mass_action_1s, args=["x", "k2"], stoichiometry={"x": -1}
        )
    )
    sens = forward_sensitivities.time_course(model, T, ["k", "s"]).unwrap_or_err()
    e = np.exp(-2.0 * T)
    np.testing.assert_allclose(sens.derivatives["s"]["x"], 2 * e - 4 * T * e, atol=1e-6)
    np.testing.assert_allclose(sens.derivatives["k"]["x"], -8 * T * e, atol=1e-6)


def test_rejects_derived_and_unknown_parameters() -> None:
    model = _chain().add_parameter("k3", InitialAssignment(fn=fns.twice, args=["k1"]))
    with pytest.raises(ValueError, match="derived"):
        forward_sensitivities.time_course(model, T, ["k3"])
    with pytest.raises(ValueError, match="Unknown"):
        forward_sensitivities.time_course(model, T, ["nope"])


def _unparsable(x: float, k: float) -> float:
    return k * math.fsum([x, 0.0])


def test_falls_back_to_finite_differences() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0)
        .add_variable("x", 1.0)
        .add_reaction("v", _unparsable, args=["x", "k"], stoichiometry={"x": -1})
    )
    sens = forward_sensitivities.time_course(model, T, ["k"]).unwrap_or_err()
    assert sens.method == "finite_difference"
    np.testing.assert_allclose(sens.derivatives["k"]["x"], -T * np.exp(-T), atol=1e-5)
    with pytest.raises(ValueError, match="parse"):
        forward_sensitivities.time_course(model, T, ["k"], method="forward")


def _total(d: pd.Series) -> float:
    return float(d.sum())


def test_data_constants_stay_symbolic() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0)
        .add_variable("x", 1.0)
        .add_data("d", pd.Series([0.5, 0.5]))
        .add_derived("tot", _total, args=["d"])
        .add_reaction(
            "v", fns.mass_action_2s, args=["x", "k", "tot"], stoichiometry={"x": -1}
        )
    )
    sens = forward_sensitivities.time_course(model, T, ["k"]).unwrap_or_err()
    assert sens.method == "forward"
    np.testing.assert_allclose(sens.derivatives["k"]["x"], -T * np.exp(-T), atol=1e-6)


def _decay(x: float, k: float) -> float:
    return -k * x


def test_ode_model_and_runners() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", 2.0, _decay, args=["x", "k"])
    )
    sens = OdeRunner().time_course_sensitivities(model, T, ["k"]).unwrap_or_err()
    assert sens.method == "forward"
    np.testing.assert_allclose(
        sens.derivatives["k"]["x"], -T * 2 * np.exp(-0.5 * T), atol=1e-6
    )

    # An explicitly given initial value doesn't follow the model
    sens = forward_sensitivities.time_course(
        model, T, ["k"], y0={"x": 3.0}, integrator=Scipy
    ).unwrap_or_err()
    np.testing.assert_allclose(
        sens.derivatives["k"]["x"], -T * 3 * np.exp(-0.5 * T), atol=1e-6
    )

    ss = (
        KineticRunner()
        .steady_state_sensitivities(_reversible(), ["kf"])
        .unwrap_or_err()
    )
    assert ss.derivatives["kf"]["A"].iloc[0] == pytest.approx(-0.5 / 1.5**2)


def test_runner_chooses_the_method() -> None:
    runner = KineticRunner(derivatives="finite_difference", displacement=1e-4)
    sens = runner.time_course_sensitivities(_chain(), T, ["k1"]).unwrap_or_err()
    assert sens.method == "finite_difference"
    np.testing.assert_allclose(
        sens.derivatives["k1"]["x"], -T * 2.0 * np.exp(-0.5 * T), atol=1e-5
    )


def test_sensitivities_leave_model_unchanged() -> None:
    model = _chain()
    before = model.get_parameter_values()
    forward_sensitivities.time_course(
        model, T, ["k1", "k2"], method="finite_difference"
    )
    assert model.get_parameter_values() == before


def test_readouts_are_differentiated() -> None:
    model = _chain().add_readout("total", fns.add, args=["x", "y"])
    sens = forward_sensitivities.time_course(model, T, ["k1"]).unwrap_or_err()
    assert sens.method == "forward"
    s = sens.derivatives["k1"]
    np.testing.assert_allclose(s["total"], s["x"] + s["y"], atol=1e-12)


def test_unknown_outputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown outputs"):
        forward_sensitivities.time_course(_chain(), T, ["k1"], outputs=["nope"])


def test_any_compile_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(mxlpy.forward_sensitivities._core, "compile_derivatives", fail)
    sens = forward_sensitivities.time_course(_chain(), T, ["k1"]).unwrap_or_err()
    assert sens.method == "finite_difference"
    with pytest.raises(RuntimeError, match="boom"):
        forward_sensitivities.time_course(_chain(), T, ["k1"], method="forward")


def test_partial_native_uses_finite_differences() -> None:
    native = pytest.importorskip("mxlpy.integrators.int_native")
    if native.problem("llvm") is not None and native.problem("wasm") is not None:
        pytest.skip("No native backend installed")
    integrator = functools.partial(native.Native, rtol=1e-10, atol=1e-10)
    sens = forward_sensitivities.time_course(
        _chain(), T, ["k1"], integrator=integrator
    ).unwrap_or_err()
    assert sens.method == "finite_difference"
    with pytest.raises(ValueError, match="Native"):
        forward_sensitivities.time_course(
            _chain(), T, ["k1"], integrator=integrator, method="forward"
        )


def test_events_use_finite_differences() -> None:
    model = _chain().add_event(
        "reset",
        fns.minus,
        trigger_args=["x", "k1"],
        assignments={"y": Derived(fn=fns.constant, args=["k2"])},
        direction="falling",
    )
    sens = forward_sensitivities.time_course(
        model, T, ["k1"], integrator=Scipy
    ).unwrap_or_err()
    assert sens.method == "finite_difference"


def test_jacobian_refreshes_data_constants() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_data("d", pd.Series([1.0, 2.0]))
        .add_derived("tot", _total, args=["d"])
        .add_reaction("v", fns.proportional, args=["x", "tot"], stoichiometry={"x": -1})
    )
    jac = jacobian_fn(model)
    assert jac(0.0, np.array([1.0]))[0, 0] == pytest.approx(-3.0)
    model.update_data("d", pd.Series([10.0, 20.0]))
    assert jac(0.0, np.array([1.0]))[0, 0] == pytest.approx(-30.0)

