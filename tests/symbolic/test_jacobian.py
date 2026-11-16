# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import logging

import numpy as np
import pandas as pd
import pytest

import mxlpy.simulator
from mxlpy import KineticModelBuilder, Simulator, fns
from mxlpy.symbolic import jacobian_fn


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


def _fd_jacobian(model: KineticModelBuilder, y: np.ndarray) -> np.ndarray:
    f0 = np.array(model(0.0, y))
    out = np.empty((len(y), len(y)))
    for j in range(len(y)):
        yp = y.copy()
        yp[j] += 1e-7
        out[:, j] = (np.array(model(0.0, yp)) - f0) / 1e-7
    return out


def test_jacobian_matches_finite_differences() -> None:
    model = _model()
    y = np.array([1.5, 0.5])
    np.testing.assert_allclose(
        jacobian_fn(model)(0.0, y), _fd_jacobian(model, y), rtol=1e-5
    )


def test_jacobian_tracks_parameter_updates() -> None:
    model = _model()
    jac = jacobian_fn(model)
    y = np.array([1.5, 0.5])
    before = jac(0.0, y)
    model.update_parameter("k1", 4.0)
    after = jac(0.0, y)
    assert after[0, 0] == pytest.approx(-4.0)
    assert before[0, 0] == pytest.approx(-2.0)
    np.testing.assert_allclose(after, _fd_jacobian(model, y), rtol=1e-5)


def test_jacobian_rejects_data() -> None:
    model = _data_model()
    with pytest.raises(Exception):  # noqa: B017
        jacobian_fn(model)


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
