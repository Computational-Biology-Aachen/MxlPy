# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Discrete adjoints (Native) and the pullback API.

The discrete gradient is exact for the recorded solve with frozen step sizes,
so it is checked against central differences of the tape's replay, which
re-solves the same steps at perturbed parameters. Errors are relative,
max-norm over the gradient.
"""

from __future__ import annotations

import functools
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    adjoint_sensitivities,
)
from mxlpy.adjoint_sensitivities import _discrete
from mxlpy.fit._gradient import GradientResidual
from mxlpy.fit.abstract import _Settings
from mxlpy.fit.losses import mean_squared
from mxlpy.fit.residuals import time_course_residual
from mxlpy.runners import KineticRunner
from mxlpy.symbolic.jacobian import all_parameter_values
from tests.test_adjoint_sensitivities import (
    _PROTOCOL,
    _PROTOCOL_TIMES,
    T,
    _chain,
    _cotangents,
    _stiff,
    _with_input,
)

METHODS = ["Radau", "LSODA", "CVODE_BDF", "CVODE_Adams"]


def _native(**kwargs: Any) -> Any:
    native = pytest.importorskip("mxlpy.integrators.int_native")
    backend = kwargs.get("backend", "llvm")
    if native.problem(backend) is not None:
        pytest.skip(f"Native's {backend} backend isn't installed")
    return functools.partial(native.Native, **kwargs)


def _relative_error(got: Any, expected: Any) -> float:
    got, expected = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    return float(np.abs(got - expected).max() / np.abs(expected).max())


@pytest.mark.parametrize("backend", ["llvm", "wasm"])
@pytest.mark.parametrize("method", METHODS)
def test_matches_forward_sensitivities(method: str, backend: str) -> None:
    integrator = _native(method=method, backend=backend, rtol=1e-10, atol=1e-12)
    cases = [(_chain(), _cotangents(["x", "y", "v1"], T))]
    if method != "CVODE_Adams":  # fixed-point iteration doesn't do stiff models
        cases.append((_stiff(), _cotangents(["A", "B", "sink"], [0.01, 0.5, 2.0])))
    for model, w in cases:
        wrt = model.get_parameter_names() + model.get_variable_names()
        adj = adjoint_sensitivities.time_course(
            model, w, wrt, method="discrete_adjoint", integrator=integrator
        ).unwrap_or_err()
        fwd = adjoint_sensitivities.time_course(
            model, w, wrt, method="forward"
        ).unwrap_or_err()
        assert adj.method == "discrete_adjoint"
        assert adj.adjoint is not None
        assert list(adj.adjoint.index) == [0.0]
        assert _relative_error(adj.gradient, fwd.gradient) < 1e-6
        np.testing.assert_allclose(adj.simulated, fwd.simulated, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_exact_for_the_recorded_solve(method: str) -> None:
    """At a loose tolerance the gradient is that of the computed solution."""
    model = _chain()
    integrator = _native(method=method, rtol=1e-4, atol=1e-6)
    columns, wrt = ["x", "y", "v1"], ["k1", "k2"]
    times = np.array(T)
    x0 = np.array([2.0, 0.0])
    predictions, tape = cast(
        "tuple[Any, _discrete.Tape]",
        _discrete.forward(model, times, columns, wrt, x0=x0, integrator=integrator),
    )
    w = _cotangents(columns, T).to_numpy()
    theta_bar, lam0 = cast("tuple[Any, Any]", tape.backward(w))

    names = tape.parameter_names
    values = all_parameter_values(model)
    pars = np.array([[values[k] for k in names]])
    replayed = cast(np.ndarray, tape.replay(pars, x0))
    # Up to the solver's Newton tolerance (tightened while recording)
    assert _relative_error(replayed, predictions) < 1e-5

    def loss(p: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(w * cast(np.ndarray, tape.replay(p, y))))

    fd = []
    for k in wrt:
        i, h = names.index(k), 1e-6
        up, down = pars.copy(), pars.copy()
        up[0, i] += h
        down[0, i] -= h
        fd.append((loss(up, x0) - loss(down, x0)) / (2 * h))
    for i in range(len(x0)):
        up, down = x0.copy(), x0.copy()
        up[i] += 1e-6
        down[i] -= 1e-6
        fd.append((loss(pars, up) - loss(pars, down)) / 2e-6)
    assert _relative_error([*theta_bar, *lam0], fd) < 1e-5


@pytest.mark.parametrize("backend", ["llvm", "wasm"])
@pytest.mark.parametrize("method", ["Radau", "LSODA", "CVODE_BDF"])
def test_protocol(method: str, backend: str) -> None:
    model = _with_input()
    integrator = _native(method=method, backend=backend, rtol=1e-10, atol=1e-12)
    w = _cotangents(["x", "y", "v1"], _PROTOCOL_TIMES)
    wrt = ["k1", "k2", "x"]
    adj = adjoint_sensitivities.protocol_time_course(
        model, _PROTOCOL, w, wrt, method="discrete_adjoint", integrator=integrator
    ).unwrap_or_err()
    ref = adjoint_sensitivities.protocol_time_course(
        model, _PROTOCOL, w, wrt, method="continuous_adjoint"
    ).unwrap_or_err()
    assert adj.method == "discrete_adjoint"
    # The time point after the protocol is ignored, as in the simulation
    assert list(adj.simulated.index) == _PROTOCOL_TIMES[:-1]
    assert _relative_error(adj.gradient, ref.gradient) < 1e-5


def test_initial_assignments_and_derived_targets() -> None:
    model = _chain()
    w = _cotangents(["x", "v2"], T)
    wrt = ["k1", "k2", "x", "y"]
    integrator = _native(rtol=1e-10, atol=1e-12)
    adj = adjoint_sensitivities.time_course(
        model, w, wrt, method="discrete_adjoint", integrator=integrator
    ).unwrap_or_err()
    fwd = adjoint_sensitivities.time_course(
        model, w, wrt, method="forward"
    ).unwrap_or_err()
    assert _relative_error(adj.gradient, fwd.gradient) < 1e-6


def test_pullback_is_reusable_and_checks_its_cotangents() -> None:
    model = _chain()
    pullback = adjoint_sensitivities.time_course_pullback(
        model, T, ["k1", "k2"], columns=["x", "y"], integrator=_native()
    ).unwrap_or_err()
    assert pullback.method == "discrete_adjoint"
    assert list(pullback.simulated.columns) == ["x", "y"]
    for seed in (0, 1):
        w = _cotangents(["x", "y"], T, seed=seed)
        once = adjoint_sensitivities.time_course(
            model, w, ["k1", "k2"], integrator=_native()
        ).unwrap_or_err()
        again = pullback(w).unwrap_or_err()
        np.testing.assert_allclose(again.gradient, once.gradient, rtol=1e-12)
    # Missing entries weigh nothing
    partial = pullback(_cotangents(["x"], T[:2])).unwrap_or_err()
    full = pullback(
        _cotangents(["x"], T[:2]).reindex(index=T, columns=["x", "y"]).fillna(0.0)
    ).unwrap_or_err()
    np.testing.assert_allclose(partial.gradient, full.gradient, rtol=1e-12)
    with pytest.raises(ValueError, match="Unknown cotangent columns"):
        pullback(_cotangents(["v1"], T))
    with pytest.raises(ValueError, match="weren't simulated"):
        pullback(_cotangents(["x"], [0.75]))
    with pytest.raises(TypeError):
        pullback(pd.Series({"x": 1.0}))


def test_tape_limit() -> None:
    model = _chain()
    w = _cotangents(["x"], T)
    small = _native(max_tape_bytes=100)
    with pytest.raises(adjoint_sensitivities.TapeTooLarge):
        adjoint_sensitivities.time_course(
            model, w, ["k1"], method="discrete_adjoint", integrator=small
        )
    res = adjoint_sensitivities.time_course(
        model, w, ["k1"], method="adjoint", integrator=small
    ).unwrap_or_err()
    assert res.method == "continuous_adjoint"


def test_discrete_needs_native() -> None:
    with pytest.raises(ValueError, match="Native"):
        adjoint_sensitivities.time_course_pullback(
            _chain(), T, ["k1"], method="discrete_adjoint"
        )


def test_fit_loss_comes_from_the_pullback(monkeypatch: pytest.MonkeyPatch) -> None:
    """One forward solve per loss and gradient, and the loss is the pullback's."""
    model = _chain()
    data = (
        adjoint_sensitivities.time_course_pullback(
            model, T, ["k1"], columns=["x", "y"], integrator=_native()
        )
        .unwrap_or_err()
        .simulated
    )
    runner = KineticRunner(integrator=_native(rtol=1e-6, atol=1e-8))
    settings = _Settings(
        model=model,
        data=cast(pd.DataFrame, data),
        y0=None,
        runner=runner,
        loss_fn=mean_squared,
        p_names=["k1", "k2"],
        v_names=[],
        standard_scale=False,
    )
    residual = GradientResidual(time_course_residual, settings)
    calls = []
    original = KineticRunner.time_course_pullback

    def counting(self: KineticRunner, *args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KineticRunner, "time_course_pullback", counting)
    updates = {"k1": 1.1, "k2": 0.4}
    # Residual vectors (least squares) and their finite differences don't
    # need a pullback, so they don't record a tape
    residual.residuals(updates)
    residual.jacobian(updates)
    assert not calls
    value = residual(updates)
    gradient = residual.gradient(updates)
    assert len(calls) == 1
    assert residual.uses_pullback(updates)
    prediction = cast(pd.DataFrame, residual._prediction)
    assert value == pytest.approx(settings.loss(prediction))
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("method", METHODS)
def test_analytic_jacobian(method: str) -> None:
    """Native's simulations with the compiled Jacobian agree with finite differences."""
    model = _stiff()
    times = np.linspace(0, 2, 11)
    results = {}
    for jac in ("analytic", "finite_difference"):
        integrator = _native(method=method, jacobian_method=jac, rtol=1e-9, atol=1e-11)
        sim = (
            Simulator(model, integrator=integrator)
            .simulate_time_course(times)
            .get_result()
            .unwrap_or_err()
        )
        results[jac] = sim.get_variables().loc[:, ["A", "B"]].to_numpy()
    np.testing.assert_allclose(
        results["analytic"], results["finite_difference"], rtol=1e-6, atol=1e-9
    )


def test_compiled_jacobian_matches_the_model() -> None:
    native = pytest.importorskip("mxlpy.integrators.int_native")
    if native.problem("llvm") is not None:
        pytest.skip("Native's llvm backend isn't installed")
    model = _stiff()
    y0 = np.array([1.0, 0.5])
    rhs = native._compile(model, "llvm", 0.0, y0)
    entry = native._compile_jacobian(model, "llvm", 0.0, y0, rhs)
    assert entry is not None


def _switch(x: float, threshold: float, width: float) -> float:
    return 1 / (1 + np.exp((x - threshold) / width))


def _decay(x: float, k: float) -> float:
    return k * x


def _overflowing() -> Any:
    """A steep logistic switch: its symbolic derivative is inf / inf (NaN) once exp overflows.

    The model itself stays finite there; x starts in the overflow region
    (x0 = 2, exp((2 - 1) / 5e-4) = inf) and decays through the switch.
    """
    return (
        KineticModelBuilder()
        .add_parameters({"threshold": 1.0, "width": 5e-4, "k": 0.5})
        .add_variables({"x": 2.0})
        .add_reaction(
            "influx", _switch, args=["x", "threshold", "width"], stoichiometry={"x": 1}
        )
        .add_reaction("decay", _decay, args=["x", "k"], stoichiometry={"x": -1})
    )


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
@pytest.mark.parametrize("method", METHODS)
def test_analytic_jacobian_with_non_finite_entries(method: str) -> None:
    """Columns the compiled Jacobian can't evaluate are differenced by the solvers.

    The Jacobian is NaN at y0, so without that every solver fails.
    """
    model = _overflowing()
    native = pytest.importorskip("mxlpy.integrators.int_native")
    y0 = np.array([2.0])
    rhs = native._compile(model, "llvm", 0.0, y0)
    entry = native._compile_jacobian(model, "llvm", 0.0, y0, rhs)
    assert entry is not None
    times = np.linspace(0, 4, 21)
    results = {}
    for jac in ("analytic", "finite_difference"):
        integrator = _native(method=method, jacobian_method=jac, rtol=1e-9, atol=1e-11)
        sim = (
            Simulator(model, integrator=integrator)
            .simulate_time_course(times)
            .get_result()
            .unwrap_or_err()
        )
        results[jac] = sim.get_variables()["x"].to_numpy()
    np.testing.assert_allclose(
        results["analytic"], results["finite_difference"], rtol=1e-6, atol=1e-9
    )


def _offset_sqrt(x: float, c: float, k: float) -> float:
    return np.sqrt(c - 1) - k * x


def test_self_check_uses_the_model_parameters() -> None:
    """The adjoint self-check runs at the model's parameters.

    At random parameters U(0.5, 1.5), sqrt(c - 1) is NaN for c < 1, which
    used to reject correct adjoints of real models.
    """
    model = (
        OdeModelBuilder()
        .add_parameters({"c": 5.0, "k": 0.7})
        .add_diff_eq("x", 1.0, _offset_sqrt, args=["x", "c", "k"])
    )
    assert not isinstance(
        _discrete._compile(model, ["c", "k"], ["x"], "llvm"), Exception
    )
    w = _cotangents(["x"], T)
    integrator = _native(rtol=1e-10, atol=1e-12)
    adj = adjoint_sensitivities.time_course(
        model, w, ["c", "k"], method="discrete_adjoint", integrator=integrator
    ).unwrap_or_err()
    fd = adjoint_sensitivities.time_course(
        model, w, ["c", "k"], method="finite_difference", integrator=integrator
    ).unwrap_or_err()
    assert _relative_error(adj.gradient, fd.gradient) < 1e-5
