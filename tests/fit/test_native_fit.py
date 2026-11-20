# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import logging

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    InitialAssignment,
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    fit,
    fns,
    make_protocol,
)
from mxlpy.fit import _native
from mxlpy.integrators._native.backends import problem

pytestmark = pytest.mark.skipif(
    problem("llvm") is not None, reason="needs the llvm backend (mxlpy[native])"
)


@pytest.fixture(autouse=True)
def _reset_warnings() -> None:
    # Fallback warnings are logged once per process, which would hide later ones
    _native._WARNED.clear()


TRUE = {"k1": 0.5, "k2": 2.0}
P0 = {"k1": 1.5, "k2": 0.7}
T = np.linspace(0.2, 6.0, 30)


def _model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters(TRUE)
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


def _data(columns: list[str] | None = None) -> pd.DataFrame:
    res = Simulator(_model()).simulate_time_course(T).get_result().unwrap_or_err()
    return res.get_combined().loc[T, columns or ["x", "y", "v2"]]


def _fit(caplog: pytest.LogCaptureFixture, **kwargs: object) -> fit.Fit:
    defaults: dict[str, object] = {
        "p0": P0,
        "minimizer": fit.NativeLeastSquaresMinimizer(),
        "sensitivities": "forward",
    }
    with caplog.at_level(logging.WARNING):
        return fit.time_course(_model(), **(defaults | kwargs)).unwrap_or_err()  # type: ignore[arg-type]


def _assert_native(caplog: pytest.LogCaptureFixture) -> None:
    assert "instead of the compiled loop" not in caplog.text


@pytest.mark.parametrize("transform", [None, "log"])
@pytest.mark.parametrize("method", ["Radau", "LSODA"])
def test_time_course_with_derived_target(
    caplog: pytest.LogCaptureFixture, transform: str | None, method: str
) -> None:
    res = _fit(
        caplog,
        data=_data(),
        transform=transform,
        minimizer=fit.NativeLeastSquaresMinimizer(method=method),  # type: ignore[arg-type]
    )
    _assert_native(caplog)
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-6)
    # The reported loss is the fit's own loss function
    assert res.loss == pytest.approx(0.0, abs=1e-5)


def test_missing_constant_and_replicate_data(caplog: pytest.LogCaptureFixture) -> None:
    data = _data(["x", "y", "v2"])
    data.iloc[3, 1] = np.nan
    data["x"] = 1.0  # constant: skipped like the losses do
    data = pd.concat([data, data]).sort_index()
    with caplog.at_level(logging.WARNING):
        res = fit.time_course(
            _model(),
            p0=P0,
            data=data,
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    _assert_native(caplog)
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-6)


def test_matches_scipy_least_squares_on_noisy_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = _data(["x", "y"])
    rng = np.random.default_rng(1)
    data += rng.normal(0, 0.01, data.shape)
    native = _fit(caplog, data=data, transform="log")
    _assert_native(caplog)
    scipy = fit.time_course(
        _model(),
        p0=P0,
        data=data,
        minimizer=fit.LeastSquaresMinimizer(),
        sensitivities="forward",
        transform="log",
    ).unwrap_or_err()
    for k in TRUE:
        assert native.best_pars[k] == pytest.approx(scipy.best_pars[k], rel=1e-5)
    assert native.loss == pytest.approx(scipy.loss, rel=1e-6)


def test_steady_state(caplog: pytest.LogCaptureFixture) -> None:
    model = (
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
    with caplog.at_level(logging.WARNING):
        res = fit.steady_state(
            model,
            p0={"kf": 1.0, "kr": 2.0},
            data=pd.Series({"A": 1 / 3, "vf": 2 / 3}),
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
            standard_scale=False,
            transform="log",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["kf"] == pytest.approx(2.0, rel=1e-5)
    assert res.best_pars["kr"] == pytest.approx(1.0, rel=1e-5)


def test_protocol_time_course(caplog: pytest.LogCaptureFixture) -> None:
    protocol = make_protocol([(2.0, {"k1": 0.5}), (4.0, {"k1": 1.0})])
    data = (
        Simulator(_model())
        .simulate_protocol_time_course(protocol, T)
        .get_result()
        .unwrap_or_err()
        .variables.reindex(T, method="nearest", tolerance=1e-9)
        .loc[:, ["x", "y"]]
    )
    with caplog.at_level(logging.WARNING):
        res = fit.protocol_time_course(
            _model(),
            p0={"k2": 0.7},
            data=data,
            protocol=protocol,
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["k2"] == pytest.approx(2.0, rel=1e-6)


def test_ode_model_builder(caplog: pytest.LogCaptureFixture) -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 1.5)
        .add_derived("rate", fns.mass_action_1s, args=["x", "k"])
        .add_diff_eq("x", fn=fns.neg, args=["rate"], initial_value=2.0)
    )
    data = pd.DataFrame({"x": 2.0 * np.exp(-0.7 * T)}, index=T)
    with caplog.at_level(logging.WARNING):
        res = fit.time_course(
            model,
            p0={"k": 1.5},
            data=data,
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["k"] == pytest.approx(0.7, rel=1e-6)


def test_falls_back_to_scipy(caplog: pytest.LogCaptureFixture) -> None:
    def double(df: pd.DataFrame) -> pd.DataFrame:
        return df * 2

    res = _fit(caplog, data=_data() * 2, postprocess_fn=double)
    assert "instead of the compiled loop" in caplog.text
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-4)

    # Fitted initial values can't be expressed either
    caplog.clear()
    res = _fit(caplog, data=_data(), p0=P0 | {"x": 1.0})
    assert res.best_pars["x"] == pytest.approx(2.0, rel=1e-4)


def test_needs_sensitivities() -> None:
    with pytest.raises(TypeError, match="sensitivities"):
        fit.time_course(
            _model(), p0=P0, data=_data(), minimizer=fit.NativeLeastSquaresMinimizer()
        )


def test_steady_state_with_standard_scaling(caplog: pytest.LogCaptureFixture) -> None:
    # The scale of a Series is a single number
    model = (
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
    with caplog.at_level(logging.WARNING):
        res = fit.steady_state(
            model,
            p0={"kf": 1.0, "kr": 2.0},
            data=pd.Series({"A": 1 / 3, "vf": 2 / 3}),
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
            transform="log",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["kf"] == pytest.approx(2.0, rel=1e-5)
    assert res.loss == pytest.approx(0.0, abs=1e-6)


def _abs_rate(x: float, k: float) -> float:
    return k * abs(x - 3.0)


def _max_rate(x: float, k: float) -> float:
    return max(k * x, 0.1)


@pytest.mark.parametrize("rate", [_abs_rate, _max_rate])
def test_abs_and_max_are_differentiated(
    caplog: pytest.LogCaptureFixture, rate: object
) -> None:
    def model() -> KineticModelBuilder:
        return (
            KineticModelBuilder()
            .add_parameters({"k": 0.4})
            .add_variables({"x": 2.0})
            .add_reaction("v", rate, args=["x", "k"], stoichiometry={"x": -1})  # type: ignore[arg-type]
        )

    data = Simulator(model()).simulate_time_course(T).get_result().unwrap_or_err()
    with caplog.at_level(logging.WARNING):
        res = fit.time_course(
            model(),
            p0={"k": 1.0},
            data=data.variables.loc[T],
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["k"] == pytest.approx(0.4, rel=1e-6)


def test_parameter_assigned_from_fitted_one_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = (
        KineticModelBuilder()
        .add_parameters({"ka": 0.5, "kb": InitialAssignment(fn=fns.twice, args=["ka"])})
        .add_variables({"x": 2.0})
        .add_reaction(
            "v", fns.mass_action_1s, args=["x", "kb"], stoichiometry={"x": -1}
        )
    )
    data = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    with caplog.at_level(logging.WARNING):
        res = fit.time_course(
            model,
            p0={"ka": 1.0},
            data=data.variables.loc[T],
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    assert "instead of the compiled loop" in caplog.text
    assert res.best_pars["ka"] == pytest.approx(0.5, rel=1e-4)


def test_fallback_leaves_model_at_best_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def double(df: pd.DataFrame) -> pd.DataFrame:
        return df * 2

    res = _fit(caplog, data=_data() * 2, postprocess_fn=double)
    assert "instead of the compiled loop" in caplog.text
    for k, v in TRUE.items():
        assert res.model.get_parameter_values()[k] == pytest.approx(v, rel=1e-4)


def test_protocol_data_after_the_protocol_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    protocol = make_protocol([(2.0, {"k1": 0.5}), (2.0, {"k1": 1.0})])
    data = (
        Simulator(_model())
        .simulate_protocol_time_course(protocol, T[T <= 4.0])
        .get_result()
        .unwrap_or_err()
        .variables.reindex(T[T <= 4.0], method="nearest", tolerance=1e-9)
        .loc[:, ["x", "y"]]
    )
    late = pd.DataFrame({"x": [9.0], "y": [9.0]}, index=[5.0])
    with caplog.at_level(logging.WARNING):
        res = fit.protocol_time_course(
            _model(),
            p0={"k2": 0.7},
            data=pd.concat([data, late]),
            protocol=protocol,
            minimizer=fit.NativeLeastSquaresMinimizer(),
            sensitivities="forward",
        ).unwrap_or_err()
    _assert_native(caplog)
    assert res.best_pars["k2"] == pytest.approx(2.0, rel=1e-6)
