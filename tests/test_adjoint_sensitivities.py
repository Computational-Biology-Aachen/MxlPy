# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import functools
import math
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    adjoint_sensitivities,
    fit,
    fns,
    forward_sensitivities,
)
from mxlpy.adjoint_sensitivities import _native as native_adjoint
from mxlpy.distributions import Normal
from mxlpy.forward_sensitivities._core import _Derivatives, _output_names
from mxlpy.integrators import Scipy
from mxlpy.runners import KineticRunner, OdeRunner
from mxlpy.symbolic import jacobian
from mxlpy.symbolic.jacobian import (
    all_parameter_values,
    compile_derivatives,
    compile_rhs,
    compile_vjp,
    constant_values,
)
from mxlpy.types import Derived, InitialAssignment

T = [0.5, 1.0, 2.0, 4.0]
RTOL = 1e-5


def _chain() -> KineticModelBuilder:
    """x -> y -> ; x(t) = 2 exp(-k1 t)."""
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
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


def _michaelis_menten(s: float, vmax: float, km: float) -> float:
    return vmax * s / (km + s)


def _stiff() -> KineticModelBuilder:
    """Fast equilibrium A <-> B (rates 1e4) feeding a slow saturable sink."""
    return (
        KineticModelBuilder()
        .add_parameters({"kf": 1e4, "kr": 2e4, "vmax": 1.0, "km": 0.5})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "kf"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
        .add_reaction(
            "sink", _michaelis_menten, args=["B", "vmax", "km"], stoichiometry={"B": -1}
        )
    )


def _cotangents(columns: list[str], times: list[float], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(len(times), len(columns))), index=times, columns=columns
    )


def _compare(
    model: KineticModelBuilder | OdeModelBuilder,
    w: pd.DataFrame,
    wrt: list[str],
    **kwargs: Any,
) -> None:
    adj = adjoint_sensitivities.time_course(model, w, wrt, method="adjoint", **kwargs)
    fwd = adjoint_sensitivities.time_course(model, w, wrt, method="forward", **kwargs)
    a, f = adj.unwrap_or_err(), fwd.unwrap_or_err()
    assert a.method == "adjoint"
    assert f.method == "forward"
    np.testing.assert_allclose(
        a.gradient, f.gradient, rtol=RTOL, atol=RTOL * np.abs(f.gradient).max()
    )
    pd.testing.assert_frame_equal(
        cast(pd.DataFrame, a.simulated),
        cast(pd.DataFrame, f.simulated),
        rtol=1e-5,
        atol=1e-8,
    )


###############################################################################
# Codegen
###############################################################################


@pytest.mark.parametrize("model_fn", [_chain, _reversible, _stiff])
def test_vjp_matches_dense_products(
    model_fn: Callable[[], KineticModelBuilder],
) -> None:
    model = model_fn()
    pars = model.get_parameter_names()
    outputs = _output_names(model)
    dense = _Derivatives(model, compile_derivatives(model, wrt=pars, outputs=outputs))
    vjp = compile_vjp(model, wrt=pars, outputs=outputs)
    values = all_parameter_values(model)
    p = np.array([values[k] for k in pars])
    c = constant_values(model, vjp.constant_names)

    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, len(model.get_variable_names()))
    lam, w = rng.normal(size=len(x)), rng.normal(size=len(outputs))
    jx, jp = dense.rhs(0.3, x)
    gx, gp = dense.outputs(0.3, x)
    x_bar, p_bar = vjp.fn(0.3, x, p, c, lam, w)
    np.testing.assert_allclose(x_bar, lam @ jx + w @ gx, atol=1e-12)
    np.testing.assert_allclose(p_bar, lam @ jp + w @ gp, atol=1e-12)


def test_compiled_rhs_matches_model() -> None:
    model = _stiff()
    rhs, constants = compile_rhs(model)
    values = all_parameter_values(model)
    p = np.array([values[k] for k in model.get_parameter_names()])
    x = np.array([0.7, 0.2])
    np.testing.assert_allclose(
        rhs(0.0, x, p, constant_values(model, constants)), model(0.0, x)
    )


def test_vjp_is_cached_per_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _chain()
    calls: list[tuple[Any, ...]] = []
    original = jacobian._compile_vjp

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(jacobian, "_compile_vjp", counting)
    compile_vjp(model, wrt=["k1"])
    model.update_parameter("k1", 3.0)
    compile_vjp(model, wrt=["k1"])
    assert len(calls) == 1
    compile_vjp(model, wrt=["k2"])
    assert len(calls) == 2


###############################################################################
# Time courses
###############################################################################


def test_time_course_matches_analytic() -> None:
    # x(t) = 2 exp(-k1 t), so d/dk1 sum_t x(t) = sum_t -2 t exp(-k1 t)
    w = pd.DataFrame({"x": np.ones(len(T))}, index=T)
    res = adjoint_sensitivities.time_course(
        _chain(), w, ["k1", "x"], method="adjoint"
    ).unwrap_or_err()
    t = np.array(T)
    assert cast(float, res.gradient["k1"]) == pytest.approx(
        np.sum(-2 * t * np.exp(-0.5 * t)), rel=1e-6
    )
    assert cast(float, res.gradient["x"]) == pytest.approx(
        np.sum(np.exp(-0.5 * t)), rel=1e-6
    )
    # λ(0) is the gradient w.r.t. the initial values
    assert res.adjoint is not None
    assert cast(float, res.adjoint.loc[0.0, "x"]) == pytest.approx(res.gradient["x"])


@pytest.mark.parametrize("model_fn", [_chain, _reversible])
def test_time_course_matches_forward(
    model_fn: Callable[[], KineticModelBuilder],
) -> None:
    model = model_fn()
    wrt = model.get_parameter_names() + model.get_variable_names()
    columns = model.get_variable_names() + _output_names(model)
    _compare(model, _cotangents(columns, T), wrt)


def test_stiff_model() -> None:
    model = _stiff()
    _compare(
        model,
        _cotangents(["A", "B", "sink"], [0.01, 0.5, 2.0]),
        ["kf", "kr", "vmax", "km"],
    )


def test_time_point_zero_and_scipy() -> None:
    w = _cotangents(["x", "y", "v2"], [0.0, *T])
    _compare(_chain(), w, ["k1", "k2", "x"], integrator=Scipy)


def test_initial_assignments_are_differentiated() -> None:
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
    w = pd.DataFrame({"x": np.ones(len(T))}, index=T)
    res = adjoint_sensitivities.time_course(
        model, w, ["k", "s"], method="adjoint"
    ).unwrap_or_err()
    t = np.array(T)
    e = np.exp(-2.0 * t)
    assert cast(float, res.gradient["s"]) == pytest.approx(
        np.sum(2 * e - 4 * t * e), rel=1e-6
    )
    assert cast(float, res.gradient["k"]) == pytest.approx(np.sum(-8 * t * e), rel=1e-6)

    # An explicit y0 doesn't follow the assignment
    res = adjoint_sensitivities.time_course(
        model, w, ["s"], y0={"x": 4.0}, method="adjoint"
    ).unwrap_or_err()
    assert cast(float, res.gradient["s"]) == pytest.approx(np.sum(-4 * t * e), rel=1e-6)


def _decay(x: float, k: float) -> float:
    return -k * x


def test_ode_model_and_runner() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", 2.0, _decay, args=["x", "k"])
    )
    w = pd.DataFrame({"x": np.ones(len(T))}, index=T)
    res = (
        OdeRunner(derivatives="adjoint")
        .time_course_vjp(model, w, ["k"])
        .unwrap_or_err()
    )
    assert res.method == "adjoint"
    t = np.array(T)
    assert cast(float, res.gradient["k"]) == pytest.approx(
        np.sum(-2 * t * np.exp(-0.5 * t)), rel=1e-6
    )


def test_auto_switches_at_the_threshold() -> None:
    model = _chain()
    w = _cotangents(["x"], T)
    small = adjoint_sensitivities.time_course(model, w, ["k1"]).unwrap_or_err()
    assert small.method == "forward"
    assert not adjoint_sensitivities.uses_adjoint(model, 1)
    n = adjoint_sensitivities.ADJOINT_THRESHOLD // 2
    assert adjoint_sensitivities.uses_adjoint(model, n)


def test_finite_difference_method() -> None:
    w = _cotangents(["x", "y"], T)
    fd = adjoint_sensitivities.time_course(
        _chain(), w, ["k1"], method="finite_difference"
    ).unwrap_or_err()
    adj = adjoint_sensitivities.time_course(
        _chain(), w, ["k1"], method="adjoint"
    ).unwrap_or_err()
    assert fd.method == "finite_difference"
    assert cast(float, fd.gradient["k1"]) == pytest.approx(adj.gradient["k1"], rel=1e-4)


def test_invalid_inputs() -> None:
    model = _chain()
    with pytest.raises(ValueError, match="ascending"):
        adjoint_sensitivities.time_course(model, _cotangents(["x"], [2.0, 1.0]), ["k1"])
    with pytest.raises(ValueError, match="ascending"):
        adjoint_sensitivities.time_course(
            model, _cotangents(["x"], [-1.0, 1.0]), ["k1"]
        )
    with pytest.raises(ValueError, match="Unknown cotangent"):
        adjoint_sensitivities.time_course(model, _cotangents(["nope"], T), ["k1"])
    with pytest.raises(ValueError, match="Unknown"):
        adjoint_sensitivities.time_course(model, _cotangents(["x"], T), ["nope"])


def _with_event() -> KineticModelBuilder:
    return _chain().add_event(
        "reset",
        fns.minus,
        trigger_args=["x", "k1"],
        assignments={"y": Derived(fn=fns.constant, args=["k2"])},
        direction="falling",
    )


def test_events(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _cotangents(["x"], T)
    with pytest.raises(ValueError, match="events"):
        adjoint_sensitivities.time_course(
            _with_event(), w, ["k1"], method="adjoint", integrator=Scipy
        )
    # auto falls back, even where it would use adjoints
    monkeypatch.setattr(adjoint_sensitivities._core, "ADJOINT_THRESHOLD", 1)
    res = adjoint_sensitivities.time_course(
        _with_event(), w, ["k1"], integrator=Scipy
    ).unwrap_or_err()
    assert res.method == "finite_difference"


def _unparsable(x: float, k: float) -> float:
    return k * math.fsum([x, 0.0])


def test_unparsable_model() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0)
        .add_variable("x", 1.0)
        .add_reaction("v", _unparsable, args=["x", "k"], stoichiometry={"x": -1})
    )
    w = _cotangents(["x"], T)
    with pytest.raises(ValueError, match="parse"):
        adjoint_sensitivities.time_course(model, w, ["k"], method="adjoint")


def test_model_is_unchanged() -> None:
    model = _chain()
    before = model.get_parameter_values() | model.get_initial_conditions()
    adjoint_sensitivities.time_course(
        model, _cotangents(["x"], T), ["k1", "x"], method="adjoint"
    ).unwrap_or_err()
    assert model.get_parameter_values() | model.get_initial_conditions() == before


def _native(**kwargs: Any) -> Any:
    native = pytest.importorskip("mxlpy.integrators.int_native")
    backend = kwargs.get("backend", "llvm")
    if native.problem(backend) is not None:
        pytest.skip(f"Native's {backend} backend isn't installed")
    return functools.partial(native.Native, rtol=1e-10, atol=1e-12, **kwargs)


@pytest.mark.parametrize("method", ["Radau", "LSODA"])
def test_compiled_adjoint(method: str) -> None:
    integrator = _native(method=method)
    for model, w in [
        (_chain(), _cotangents(["x", "y", "v1"], T)),
        (_stiff(), _cotangents(["A", "B", "sink"], [0.01, 0.5, 2.0])),
    ]:
        wrt = model.get_parameter_names() + model.get_variable_names()
        adj = adjoint_sensitivities.time_course(
            model, w, wrt, method="adjoint", integrator=integrator
        ).unwrap_or_err()
        fwd = adjoint_sensitivities.time_course(
            model, w, wrt, method="forward"
        ).unwrap_or_err()
        # Only λ(0) comes back from the compiled solve
        assert adj.adjoint is not None
        assert list(adj.adjoint.index) == [0.0]
        np.testing.assert_allclose(
            adj.gradient, fwd.gradient, rtol=1e-6, atol=1e-6 * fwd.gradient.abs().max()
        )
        np.testing.assert_allclose(adj.simulated, fwd.simulated, rtol=1e-6, atol=1e-9)


def test_compiled_adjoint_initial_assignments() -> None:
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
    w = pd.DataFrame({"x": np.ones(len(T))}, index=T)
    res = adjoint_sensitivities.time_course(
        model, w, ["k", "s"], method="adjoint", integrator=_native()
    ).unwrap_or_err()
    t = np.array(T)
    e = np.exp(-2.0 * t)
    assert cast(float, res.gradient["s"]) == pytest.approx(
        np.sum(2 * e - 4 * t * e), rel=1e-6
    )
    assert cast(float, res.gradient["k"]) == pytest.approx(np.sum(-8 * t * e), rel=1e-6)


def test_wasm_backend_uses_the_python_adjoint() -> None:
    integrator = _native(backend="wasm")
    w = _cotangents(["x", "y"], T)
    adj = adjoint_sensitivities.time_course(
        _chain(), w, ["k1", "k2"], method="adjoint", integrator=integrator
    ).unwrap_or_err()
    fwd = adjoint_sensitivities.time_course(
        _chain(), w, ["k1", "k2"], method="forward"
    ).unwrap_or_err()
    assert adj.adjoint is not None
    assert len(adj.adjoint) > 1
    np.testing.assert_allclose(adj.gradient, fwd.gradient, rtol=RTOL)


def test_native_always_uses_adjoints() -> None:
    integrator = _native()
    assert adjoint_sensitivities.uses_adjoint(_chain(), 1, integrator)
    # Not where the compiled adjoint can't run
    native = pytest.importorskip("mxlpy.integrators.int_native")
    for kwargs in ({"backend": "wasm"}, {"method": "CVODE_BDF"}):
        other = functools.partial(native.Native, **kwargs)
        assert not adjoint_sensitivities.uses_adjoint(_chain(), 1, other)
    res = adjoint_sensitivities.time_course(
        _chain(), _cotangents(["x"], T), ["k1"], integrator=integrator
    ).unwrap_or_err()
    assert res.method == "adjoint"


def test_fit_with_compiled_adjoint() -> None:
    integrator = _native()
    data = _data(_chain()).groupby(level=0).first()
    res = fit.time_course(
        _chain(),
        p0={"k1": 0.3, "k2": 3.0},
        data=data,
        minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
        runner=KineticRunner(integrator=integrator),
    ).unwrap_or_err()
    assert cast(float, res.best_pars["k1"]) == pytest.approx(0.5, rel=1e-3)
    assert cast(float, res.best_pars["k2"]) == pytest.approx(2.0, rel=1e-3)


###############################################################################
# Steady states
###############################################################################


def test_steady_state_with_conservation() -> None:
    model = _reversible()
    w = pd.Series({"A": 0.3, "B": -1.2, "vf": 0.7})
    wrt = ["kf", "kr", "A", "B"]
    adj = adjoint_sensitivities.steady_state(
        model, w, wrt, method="adjoint"
    ).unwrap_or_err()
    fwd = adjoint_sensitivities.steady_state(
        model, w, wrt, method="forward"
    ).unwrap_or_err()
    assert adj.method == "adjoint"
    np.testing.assert_allclose(adj.gradient, fwd.gradient, rtol=1e-6, atol=1e-9)
    pd.testing.assert_series_equal(
        cast(pd.Series, adj.simulated), cast(pd.Series, fwd.simulated)
    )
    # A* = (A0 + B0) kr / (kf + kr)
    assert cast(float, adj.gradient["A"]) == pytest.approx(
        0.3 * 1 / 3 - 1.2 * 2 / 3 + 0.7 * 1 / 3, rel=1e-6
    )


def test_steady_state_runner() -> None:
    model = _reversible()
    res = KineticRunner().steady_state_vjp(model, pd.Series({"A": 1.0}), ["kf"])
    assert cast(float, res.unwrap_or_err().gradient["kf"]) == pytest.approx(
        -0.5 / 1.5**2, rel=1e-5
    )


###############################################################################
# Fitting
###############################################################################


def _data(model: KineticModelBuilder) -> pd.DataFrame:
    sim = Simulator(model).simulate_time_course(np.linspace(0, 5, 11))
    data = sim.get_result().unwrap_or_err().get_variables().loc[:, ["x", "y"]]
    # Replicates share time points
    return pd.concat([data, data.iloc[3:5] * 1.1]).sort_index()


@pytest.mark.parametrize(
    "loss_fn",
    [fit.rmse, fit.mean_squared, fit.GaussianNll(0.1)],
    ids=["rmse", "mean_squared", "gaussian_nll"],
)
def test_loss_gradient_matches_sensitivities(loss_fn: fit.LossFn) -> None:
    model = _chain()
    data = _data(model)
    p = {"k1": 0.6, "k2": 1.7, "x": 2.1}
    adj = fit.loss_gradient(
        model, data, p, loss_fn=loss_fn, runner=KineticRunner(derivatives="adjoint")
    )
    fwd = fit.loss_gradient(
        model, data, p, loss_fn=loss_fn, runner=KineticRunner(derivatives="forward")
    )
    np.testing.assert_allclose(adj, fwd, rtol=RTOL)
    assert list(adj.index) == list(p)


def test_loss_gradient_steady_state_and_penalty() -> None:
    model = _reversible()
    data = pd.Series({"A": 0.4, "B": 0.6})
    p = {"kf": 1.2, "kr": 0.4}
    prior = {"kf": Normal(1.0, 0.1)}
    adj = fit.loss_gradient(
        model, data, p, runner=KineticRunner(derivatives="adjoint"), prior=prior
    )
    fwd = fit.loss_gradient(
        model, data, p, runner=KineticRunner(derivatives="forward"), prior=prior
    )
    np.testing.assert_allclose(adj, fwd, rtol=1e-6)


def test_fit_with_adjoint_gradients() -> None:
    model = _chain()
    data = _data(_chain()).groupby(level=0).first()
    p0 = {"k1": 0.3, "k2": 3.0}
    results = {}
    for derivatives in ("adjoint", "forward"):
        res = fit.time_course(
            model,
            p0=p0,
            data=data,
            minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
            runner=KineticRunner(derivatives=derivatives),
        ).unwrap_or_err()
        results[derivatives] = res.best_pars
    assert results["adjoint"]["k1"] == pytest.approx(0.5, rel=1e-3)
    assert results["adjoint"]["k2"] == pytest.approx(results["forward"]["k2"], rel=1e-3)


def test_adjoint_failure_falls_back_to_sensitivities() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0)
        .add_variable("x", 1.0)
        .add_reaction("v", _unparsable, args=["x", "k"], stoichiometry={"x": -1})
    )
    data = pd.DataFrame({"x": np.exp(-0.8 * np.array(T))}, index=T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = fit.loss_gradient(
            model, data, {"k": 1.0}, runner=KineticRunner(derivatives="adjoint")
        )
    reference = fit.loss_gradient(
        model, data, {"k": 1.0}, runner=KineticRunner(derivatives="finite_difference")
    )
    assert cast(float, grad["k"]) == pytest.approx(reference["k"], rel=1e-4)


def test_forward_sensitivities_unchanged_by_adjoint_runner() -> None:
    sens = (
        KineticRunner(derivatives="adjoint")
        .time_course_sensitivities(_chain(), T, ["k1"])
        .unwrap_or_err()
    )
    ref = forward_sensitivities.time_course(_chain(), T, ["k1"]).unwrap_or_err()
    pd.testing.assert_frame_equal(sens.derivatives["k1"], ref.derivatives["k1"])


def test_adjoint_state_includes_the_jump_at_zero() -> None:
    w = pd.DataFrame({"x": [3.0, 1.0], "y": [0.0, 1.0]}, index=[0.0, 1.0])
    res = adjoint_sensitivities.time_course(
        _chain(), w, ["x", "k1"], method="adjoint"
    ).unwrap_or_err()
    assert res.adjoint is not None
    assert cast(float, res.adjoint.loc[0.0, "x"]) == pytest.approx(res.gradient["x"])


def test_only_initial_values() -> None:
    w = pd.DataFrame({"x": [2.0], "v1": [1.0]}, index=[0.0])
    res = adjoint_sensitivities.time_course(
        _chain(), w, ["x", "k1"], method="adjoint"
    ).unwrap_or_err()
    # v1 = k1 x at t = 0
    assert cast(float, res.gradient["x"]) == pytest.approx(2.0 + 0.5)
    assert cast(float, res.gradient["k1"]) == pytest.approx(2.0)


@pytest.mark.parametrize("compiled", [False, True], ids=["python", "native"])
def test_small_cotangents_are_accurate(compiled: bool) -> None:  # noqa: FBT001
    # Close to an optimum the cotangents are tiny, λ must still be accurate
    kwargs = {"integrator": _native()} if compiled else {}
    w = _cotangents(["x", "y", "v1"], T) * 1e-8
    adj = adjoint_sensitivities.time_course(
        _chain(), w, ["k1", "k2", "x"], method="adjoint", **kwargs
    ).unwrap_or_err()
    fwd = adjoint_sensitivities.time_course(
        _chain(), w, ["k1", "k2", "x"], method="forward"
    ).unwrap_or_err()
    np.testing.assert_allclose(adj.gradient, fwd.gradient, rtol=1e-5)


def test_nan_cotangents_have_no_weight() -> None:
    w = _cotangents(["x", "y"], T)
    nan = w.copy()
    nan.iloc[1, 0] = np.nan
    zero = w.copy()
    zero.iloc[1, 0] = 0.0
    for method in ("adjoint", "forward"):
        a = adjoint_sensitivities.time_course(_chain(), nan, ["k1"], method=method)
        b = adjoint_sensitivities.time_course(_chain(), zero, ["k1"], method=method)
        assert cast(float, a.unwrap_or_err().gradient["k1"]) == pytest.approx(
            b.unwrap_or_err().gradient["k1"]
        )


def test_failed_native_compiles_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    integrator = _native()
    calls = []

    def failing(*args: Any) -> Any:
        calls.append(args)
        msg = "nope"
        raise native_adjoint.NativeUnsupported(msg)

    monkeypatch.setattr(native_adjoint, "_compile", failing)
    model = _chain()
    w = _cotangents(["x"], T)
    for _ in range(3):
        adjoint_sensitivities.time_course(
            model, w, ["k1"], method="adjoint", integrator=integrator
        ).unwrap_or_err()
    assert len(calls) == 1


def _kinked(x: float, y: float, k: float, k2: float) -> float:
    return k * max(x - 0.5, 0.0) + abs(y - 1.0) * k2


def _smaller(x: float, y: float, k: float) -> float:
    return min(x, y) * k


def test_max_abs_min_are_differentiated() -> None:
    model = (
        KineticModelBuilder()
        .add_parameters({"k": 0.8, "k2": 0.3, "k3": 1.5})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_derived("d", _smaller, args=["x", "y", "k3"])
        .add_reaction(
            "v1", _kinked, args=["x", "y", "k", "k2"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k3"], stoichiometry={"y": -1}
        )
    )
    w = _cotangents(["x", "y", "d", "v1"], T)
    wrt = ["k", "k2", "k3", "x"]
    # The kinks need tight tolerances for accurate finite differences
    tight = functools.partial(Scipy, rtol=1e-12, atol=1e-14, method="LSODA")
    fd = adjoint_sensitivities.time_course(
        model, w, wrt, method="finite_difference", displacement=1e-5, integrator=tight
    ).unwrap_or_err()
    for method in ("adjoint", "forward"):
        res = adjoint_sensitivities.time_course(
            model, w, wrt, method=method
        ).unwrap_or_err()
        assert res.method == method
        np.testing.assert_allclose(
            res.gradient, fd.gradient, rtol=1e-4, atol=1e-4 * fd.gradient.abs().max()
        )
