# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import logging

import numpy as np
import pandas as pd
import pytest

import mxlpy.simulator
import mxlpy.symbolic.jacobian
from mxlpy import KineticModelBuilder, Simulator, fns
from mxlpy.surrogates import qss
from mxlpy.symbolic import jacobian_fn

METHODS = ["chain_rule", "expanded"]


def _times_sum(x: float, d: pd.Series) -> float:
    return x * d.sum()


def _data_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_data("d", pd.Series([1.0, 2.0]))
        .add_reaction("v", _times_sum, args=["x", "d"], stoichiometry={"x": -1})
    )


def _model() -> KineticModelBuilder:
    # "3PGA" is not a valid python identifier
    return (
        KineticModelBuilder()
        .add_variable("3PGA", 1.0)
        .add_variable("x", 2.0)
        .add_parameter("k1", 2.0)
        .add_parameter("k2", 3.0)
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["3PGA", "k1"],
            stoichiometry={"3PGA": -1, "x": 1},
        )
        .add_reaction(
            "v2",
            fns.mass_action_2s,
            args=["3PGA", "x", "k2"],
            stoichiometry={"x": -1},
        )
    )


def _fd_jacobian(
    model: KineticModelBuilder, y: np.ndarray, t: float = 0.0
) -> np.ndarray:
    f0 = np.array(model(t, y))
    out = np.empty((len(y), len(y)))
    for j in range(len(y)):
        yp = y.copy()
        yp[j] += 1e-7
        out[:, j] = (np.array(model(t, yp)) - f0) / 1e-7
    return out


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_matches_finite_differences(method: str) -> None:
    model = _model()
    y = np.array([1.5, 0.5])
    np.testing.assert_allclose(
        jacobian_fn(model, method=method)(0.0, y),  # type: ignore[arg-type]
        _fd_jacobian(model, y),
        rtol=1e-5,
    )


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_tracks_parameter_updates(method: str) -> None:
    model = _model()
    jac = jacobian_fn(model, method=method)  # type: ignore[arg-type]
    y = np.array([1.5, 0.5])
    before = jac(0.0, y)
    model.update_parameter("k1", 4.0)
    after = jac(0.0, y)
    assert after[0, 0] == pytest.approx(-4.0)
    assert before[0, 0] == pytest.approx(-2.0)
    np.testing.assert_allclose(after, _fd_jacobian(model, y), rtol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_rejects_data(method: str) -> None:
    with pytest.raises(ValueError):
        jacobian_fn(_data_model(), method=method)  # type: ignore[arg-type]


def test_jacobian_defaults_to_chain_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: KineticModelBuilder) -> None:
        raise NotImplementedError

    monkeypatch.setattr(mxlpy.symbolic.jacobian, "_chain_rule", fail)
    with pytest.raises(NotImplementedError):
        jacobian_fn(_model())


def _saturate(a: float, b: float) -> float:
    return min(a, b) * max(a - b, 0.1)


def _switch(a: float, b: float) -> float:
    if a > b:
        return a * b
    return a + b


def _time_rate(x: float, k: float, time: float) -> float:
    return k * x * time


def _nested_model() -> KineticModelBuilder:
    # derived quantities re-used several levels deep, with min/max/if, which
    # the fully expanded expressions duplicate
    return (
        KineticModelBuilder()
        .add_variables({"x": 1.0, "y": 2.0, "z": 0.5})
        .add_parameters({"k": 0.7})
        .add_derived("d1", _saturate, args=["x", "y"])
        .add_derived("d2", _switch, args=["d1", "z"])
        .add_derived("d3", _saturate, args=["d2", "d1"])
        .add_reaction("v1", _switch, args=["d3", "y"], stoichiometry={"x": -1})
        .add_reaction(
            "v2", _saturate, args=["d2", "d3"], stoichiometry={"y": -1, "z": 1}
        )
        .add_reaction(
            "v3", _time_rate, args=["z", "k", "time"], stoichiometry={"z": -1}
        )
    )


@pytest.mark.parametrize("y", [[1.0, 2.0, 0.5], [3.0, 1.2, 2.5]])
def test_chain_rule_matches_expanded_for_nested_min_max(y: list[float]) -> None:
    model = _nested_model()
    state = np.array(y)
    chain = jacobian_fn(model, method="chain_rule")(2.0, state)
    expanded = jacobian_fn(model, method="expanded")(2.0, state)
    np.testing.assert_allclose(chain, expanded, rtol=1e-12)
    np.testing.assert_allclose(
        chain, _fd_jacobian(model, state, t=2.0), rtol=1e-5, atol=1e-6
    )


def _qss_flux(v1: float, k1: float) -> tuple[float]:
    return (k1 * v1**2,)


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_with_qss_surrogate(method: str) -> None:
    model = (
        KineticModelBuilder()
        .add_variable("v1", 1.0)
        .add_parameter("k1", 0.1)
        .add_surrogate(
            "srg1",
            qss.Surrogate(
                model=_qss_flux,
                args=["v1", "k1"],
                outputs=["v1_efflux"],
                stoichiometries={"v1_efflux": {"v1": -1.0}},
            ),
        )
    )
    jac = jacobian_fn(model, method=method)  # type: ignore[arg-type]
    assert jac(0.0, np.array([3.0]))[0, 0] == pytest.approx(-0.6)


def test_simulator_uses_jacobian_for_invalid_identifiers() -> None:
    sim = Simulator(_model(), use_jacobian=True)
    assert sim.integrator.jacobian is not None  # type: ignore[attr-defined]
    with_jac = sim.simulate(10).get_result().unwrap_or_err().variables
    without = Simulator(_model()).simulate(10).get_result().unwrap_or_err().variables
    np.testing.assert_allclose(with_jac.iloc[-1], without.iloc[-1], rtol=1e-6)


def test_simulator_reuses_jacobian_until_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def counting(model: KineticModelBuilder):  # noqa: ANN202
        calls.append(model)
        return jacobian_fn(model)

    monkeypatch.setattr(mxlpy.simulator, "jacobian_fn", counting)
    model = _model()
    sim = Simulator(model, use_jacobian=True).simulate(1)
    sim.clear_results()
    sim.update_variable("x", 1.0)
    assert len(calls) == 1

    model.add_parameter("k3", 1.0)
    sim.clear_results()
    assert len(calls) == 2


def test_simulator_falls_back_without_jacobian(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _data_model()
    with caplog.at_level(logging.WARNING):
        sim = Simulator(model, use_jacobian=True)
    assert sim.integrator.jacobian is None  # type: ignore[attr-defined]
    assert "couldn't be compiled" in caplog.text
    assert sim.simulate(1).get_result().unwrap_or_err() is not None
