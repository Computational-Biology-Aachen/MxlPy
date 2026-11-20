# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import sympy
from scipy.interpolate import CubicSpline, PchipInterpolator

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    fit,
    fns,
    fns_input,
    load,
    save,
)
from mxlpy.meta import generate_model_code_jax, generate_model_code_mxlweb
from mxlpy.meta.source_tools import fn_to_sympy_expr

KNOTS = [0.0, 5.0, 10.0, 30.0, 60.0]
VALUES = [1.0, 3.0, 2.0, 5.0, 4.0]
TS = np.linspace(-5.0, 65.0, 141)
INSIDE = TS[(TS >= 0) & (TS <= 60)]


def _eval(fn, ts, values):  # noqa: ANN001, ANN202
    return np.array([fn(t, *values) for t in ts])


###############################################################################
# Plain functions
###############################################################################


def test_smooth_step() -> None:
    f = fns_input.smooth_step
    assert f(10.0, 1.0, 10.0, 3.0, 2.0) == pytest.approx(2.0)
    assert f(-100.0, 1.0, 10.0, 3.0, 2.0) == pytest.approx(1.0)
    assert f(100.0, 1.0, 10.0, 3.0, 2.0) == pytest.approx(3.0)
    # Logistic function, without overflow for extreme arguments
    assert f(12.0, 0.0, 10.0, 1.0, 2.0) == pytest.approx(1 / (1 + math.exp(-4.0)))
    assert f(1e6, 0.0, 0.0, 1.0, 1e3) == 1.0


def test_step() -> None:
    assert fns_input.step(9.9, 1.0, 10.0, 3.0) == 1.0
    assert fns_input.step(10.0, 1.0, 10.0, 3.0) == 3.0


def test_periodic_step() -> None:
    f = fns_input.periodic_step
    # 16 h on, 8 h off
    on = [f(t, 0.5, 2.0, 24.0, 2 / 3, 50.0) for t in (1.0, 8.0, 15.0, 25.0)]
    off = [f(t, 0.5, 2.0, 24.0, 2 / 3, 50.0) for t in (17.0, 20.0, 23.0, 41.0)]
    np.testing.assert_allclose(on, 2.0, atol=1e-6)
    np.testing.assert_allclose(off, 0.5, atol=1e-6)
    # Switches halfway at the phase boundaries
    assert f(16.0, 0.0, 1.0, 24.0, 2 / 3, 50.0) == pytest.approx(0.5)
    # The on fraction is exact for any sharpness
    ts = np.linspace(0, 24, 24001)
    values = np.array([f(t, 0.0, 1.0, 24.0, 0.25, 200.0) for t in ts])
    assert (values > 0.5).mean() == pytest.approx(0.25, abs=1e-3)


def test_bolus_area() -> None:
    ts = np.linspace(-10, 30, 4001)
    values = np.array([fns_input.bolus(t, 3.0, 10.0, 2.0) for t in ts])
    assert np.trapezoid(values, ts) == pytest.approx(3.0, rel=1e-6)
    assert ts[values.argmax()] == pytest.approx(10.0)


def test_plain_functions_parse_to_sympy() -> None:
    for fn in (
        fns_input.smooth_step,
        fns_input.step,
        fns_input.periodic_step,
        fns_input.bolus,
    ):
        expr = fn_to_sympy_expr(fn, origin="test")
        assert expr is not None
        assert expr.has(sympy.Symbol("t"))


###############################################################################
# Splines
###############################################################################


def test_natural_spline_matches_scipy() -> None:
    f = fns_input.cubic_spline(KNOTS)
    ref = CubicSpline(KNOTS, VALUES, bc_type="natural")
    np.testing.assert_allclose(_eval(f, INSIDE, VALUES), ref(INSIDE), atol=1e-12)
    # Constant extrapolation
    assert f(-5.0, *VALUES) == VALUES[0]
    assert f(100.0, *VALUES) == VALUES[-1]


def test_initial_slope_matches_scipy() -> None:
    f = fns_input.cubic_spline(KNOTS, initial_slope=0.5)
    ref = CubicSpline(KNOTS, VALUES, bc_type=((1, 0.5), (2, 0.0)))  # type: ignore[arg-type]
    np.testing.assert_allclose(_eval(f, INSIDE, VALUES), ref(INSIDE), atol=1e-12)


def test_positive_spline() -> None:
    f = fns_input.cubic_spline(KNOTS, positive=True)
    ref = CubicSpline(KNOTS, np.log(VALUES), bc_type="natural")
    np.testing.assert_allclose(
        _eval(f, INSIDE, VALUES), np.exp(ref(INSIDE)), rtol=1e-12
    )
    # Values whose natural spline dips below zero stay positive
    dips = [1.0, 1.0, 0.01, 1.0, 1.0]
    assert (_eval(fns_input.cubic_spline(KNOTS), TS, dips) < 0).any()
    assert (_eval(f, TS, dips) > 0).all()


def test_monotone_spline() -> None:
    f = fns_input.cubic_spline(KNOTS, monotone=True)
    values = [1.0, 2.0, 2.0, 5.0, 6.0]
    y = _eval(f, TS, values)
    assert (np.diff(y) >= -1e-12).all()
    np.testing.assert_allclose([f(k, *values) for k in KNOTS], values)
    # Interior intervals match scipy's PCHIP (which uses other end slopes)
    mid = TS[(TS >= 5) & (TS <= 30)]
    ref = PchipInterpolator(KNOTS, values)
    np.testing.assert_allclose(_eval(f, mid, values), ref(mid), atol=1e-12)
    # A natural spline overshoots the flat part
    assert (np.diff(_eval(fns_input.cubic_spline(KNOTS), TS, values)) < -1e-6).any()


def test_two_knot_spline_is_linear() -> None:
    f = fns_input.cubic_spline([0.0, 10.0])
    assert f(2.5, 1.0, 5.0) == pytest.approx(2.0)


def test_spline_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fns_input.cubic_spline([1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        fns_input.cubic_spline([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="initial_slope"):
        fns_input.cubic_spline([0.0, 1.0], initial_slope=math.inf)


def test_spline_carries_expression() -> None:
    f = fns_input.cubic_spline(KNOTS)
    expr = fn_to_sympy_expr(f, origin="test")
    assert isinstance(expr, sympy.Piecewise)
    sub = dict(zip(sympy.symbols("y0:5"), VALUES, strict=True))
    assert float(expr.subs({sympy.Symbol("t"): 7.0, **sub})) == pytest.approx(
        f(7.0, *VALUES)
    )


###############################################################################
# Multi step
###############################################################################


def test_multi_step() -> None:
    hard = fns_input.multi_step([10.0, 20.0], steepness=None)
    assert [hard(t, 1.0, 2.0, 3.0) for t in (5.0, 10.0, 15.0, 25.0)] == [
        1.0,
        2.0,
        2.0,
        3.0,
    ]
    smooth = fns_input.multi_step([10.0, 20.0], steepness=100.0)
    assert smooth(15.0, 1.0, 2.0, 3.0) == pytest.approx(2.0)
    assert smooth(20.0, 1.0, 2.0, 3.0) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="steepness"):
        fns_input.multi_step([1.0], steepness=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        fns_input.multi_step([], steepness=None)


###############################################################################
# Models
###############################################################################


def _driven() -> KineticModelBuilder:
    m = KineticModelBuilder().add_variables({"x": 0.0}).add_parameters({"k": 0.5})
    m.add_spline_input("u", knots=[0.0, 5.0, 10.0, 20.0], initial=1.0, positive=True)
    m.add_reaction("vin", fns.constant, args=["u"], stoichiometry={"x": 1})
    m.add_reaction("vout", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1})
    return m


TRUE_KNOTS = {"u_0": 0.5, "u_1": 3.0, "u_2": 1.0, "u_3": 2.0}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    t = np.linspace(0, 20, 41)
    truth = _driven().update_parameters(TRUE_KNOTS)
    return (
        Simulator(truth)
        .simulate_time_course(t)
        .get_result()
        .unwrap_or_err()
        .variables.loc[t]
    )


def test_add_spline_input() -> None:
    m = _driven()
    assert fns_input.spline_parameter_names("u", 4) == ["u_0", "u_1", "u_2", "u_3"]
    assert {"u_0", "u_1", "u_2", "u_3"} <= set(m.get_parameter_names())
    assert "u" in m.get_derived_variable_names()
    m2 = KineticModelBuilder().add_spline_input("v", [0.0, 1.0], initial=[2.0, 4.0])
    assert m2.get_parameter_values() == {"v_0": 2.0, "v_1": 4.0}
    with pytest.raises(ValueError, match="one value per knot"):
        KineticModelBuilder().add_spline_input("v", [0.0, 1.0], initial=[1.0])
    with pytest.raises(ValueError, match="positive initial"):
        KineticModelBuilder().add_spline_input(
            "v", [0.0, 1.0], initial=0.0, positive=True
        )


def test_input_estimation(data) -> None:  # noqa: ANN001
    res = fit.time_course(
        _driven(),
        p0=dict.fromkeys(TRUE_KNOTS, 1.0),
        data=data,
        loss_fn=fit.GaussianNll(0.01),
        minimizer=fit.LeastSquaresMinimizer(),
        transform="log",
    ).unwrap_or_err()
    for k, v in TRUE_KNOTS.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-4)


def test_ode_builder_spline_input(data) -> None:  # noqa: ANN001
    u, k, x = sympy.symbols("u k x")
    m = (
        OdeModelBuilder()
        .add_parameters({"k": 0.5})
        .add_spline_input("u", knots=[0.0, 5.0, 10.0, 20.0], initial=1.0, positive=True)
        .add_diff_eq("x", 0.0, u - k * x)
        .update_parameters(TRUE_KNOTS)
    )
    t = data.index.to_numpy()
    res = Simulator(m).simulate_time_course(t).get_result().unwrap_or_err()
    np.testing.assert_allclose(
        res.variables.loc[t, "x"].to_numpy(), data["x"].to_numpy(), rtol=1e-4, atol=1e-6
    )


def test_serialisation_roundtrip(tmp_path: Path, data) -> None:  # noqa: ANN001
    m = _driven().update_parameters(TRUE_KNOTS)
    save(m, tmp_path / "m.json")
    loaded = cast(KineticModelBuilder, load(tmp_path / "m.json"))
    t = data.index.to_numpy()
    res = Simulator(loaded).simulate_time_course(t).get_result().unwrap_or_err()
    np.testing.assert_allclose(res.variables.loc[t].to_numpy(), data.to_numpy())


def test_codegen() -> None:
    m = _driven()
    assert "jnp.select" in str(generate_model_code_jax(m))
    assert "Piecewise" in generate_model_code_mxlweb(m)


def test_spline_accuracy_at_large_time_scales() -> None:
    knots = [k * 1e5 for k in KNOTS]
    f = fns_input.cubic_spline(knots)
    ref = CubicSpline(knots, VALUES, bc_type="natural")
    ts = INSIDE * 1e5
    np.testing.assert_allclose(_eval(f, ts, VALUES), ref(ts), rtol=1e-10)


def test_monotone_jax_gradient_finite_for_flat_values() -> None:
    jax = pytest.importorskip("jax")
    from mxlpy.meta.sympy_tools import sympy_to_inline_jax  # noqa: PLC0415

    f = fns_input.cubic_spline([0.0, 1.0, 2.0, 3.0], monotone=True)
    expr = fn_to_sympy_expr(f, origin="test")
    assert expr is not None
    code = sympy_to_inline_jax(expr)
    fn = eval(  # noqa: S307
        f"lambda t, y0, y1, y2, y3: {code}",
        {"jnp": jax.numpy, "jax": jax},
    )
    grad = jax.grad(fn, argnums=(1, 2, 3, 4))(1.5, 1.0, 1.0, 1.0, 1.0)
    assert all(np.isfinite(float(g)) for g in grad)


def test_initial_accepts_numpy_scalars() -> None:
    m = KineticModelBuilder().add_spline_input("v", [0.0, 1.0], initial=np.int64(2))  # type: ignore[arg-type]
    assert m.get_parameter_values() == {"v_0": 2.0, "v_1": 2.0}
