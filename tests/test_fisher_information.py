# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns, identify
from mxlpy.runners import KineticRunner

T = np.linspace(0.25, 5.0, 20)
A = 2.0
K = 0.7


def _decay(k: float = K) -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k": k})
        .add_variables({"x": A})
        .add_reaction("v", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1})
    )


def _product() -> KineticModelBuilder:
    # Only the product k1 * k2 is identifiable
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 1.4})
        .add_variables({"x": A})
        .add_reaction(
            "v", fns.mass_action_2s, args=["x", "k1", "k2"], stoichiometry={"x": -1}
        )
    )


def _data(model: KineticModelBuilder) -> pd.DataFrame:
    res = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    return res.variables.loc[T, ["x"]]


def test_known_sigma_matches_analytic_std_error() -> None:
    sigma = 0.05
    fim = identify.fisher_information(_decay(), _data(_decay()), {"k": K}, sigma=sigma)
    # dx/dk = -t A exp(-k t)
    expected = sigma / np.sqrt(np.sum((T * A * np.exp(-K * T)) ** 2))
    assert fim.std_errors.to_dict()["k"] == pytest.approx(expected, rel=1e-5)
    assert fim.rank == 1
    assert fim.non_identifiable().empty
    assert fim.chi2 == pytest.approx(0.0, abs=1e-10)
    assert fim.design.index.names == ["time", "observable"]

    ci = fim.wald_ci(alpha=0.95)
    assert ci.loc["k", "upper"] - ci.loc["k", "lower"] == pytest.approx(
        2 * 1.959964 * expected, rel=1e-5
    )


def test_estimated_sigma_uses_residual_variance() -> None:
    data = _data(_decay())
    noise = 0.02 * np.sin(np.arange(len(T)) * 1.7)
    data["x"] += noise
    res = fit.time_course(
        _decay(1.0),
        p0={"k": 1.0},
        data=data,
        loss_fn=fit.mean_squared,
        standard_scale=False,
        minimizer=fit.LeastSquaresMinimizer(),
    ).unwrap_or_err()
    fim = identify.fisher_information(_decay(), data, res)
    fd = identify.fisher_information(
        _decay(), data, res, runner=KineticRunner(derivatives="finite_difference")
    )
    s = fim.design.to_numpy()
    raw = fim.residuals.to_numpy() * np.sqrt(fim.residual_variance)
    assert fim.sigma_estimated
    assert fim.residual_variance == pytest.approx(np.sum(raw**2) / (len(T) - 1))
    # Weighted by the estimated noise, so chi2 = n - p and F⁻¹ = covariance
    assert fim.chi2 == pytest.approx(len(T) - 1)
    assert fim.covariance.iloc[0, 0] == pytest.approx(1 / (s.T @ s)[0, 0])
    assert fim.covariance.iloc[0, 0] == pytest.approx(1 / fim.matrix.iloc[0, 0])
    assert fim.std_errors.to_dict()["k"] == pytest.approx(
        fd.std_errors.to_dict()["k"], rel=1e-4
    )
    # t distribution with n - p degrees of freedom
    ci = fim.wald_ci(alpha=0.9)
    q = (ci.loc["k", "upper"] - ci.loc["k", "lower"]) / (2 * fim.std_errors["k"])
    assert q == pytest.approx(1.7291, rel=1e-4)


def test_non_identifiable_combination() -> None:
    values = {"k1": 0.5, "k2": 1.4}
    fim = identify.fisher_information(_product(), _data(_product()), values, sigma=0.1)
    assert fim.rank == 1
    null = fim.non_identifiable()
    assert len(null) == 1
    # d(k1 k2) = k2 dk1 + k1 dk2 vanishes along (k1, -k2)
    direction = np.array([0.5, -1.4]) / np.hypot(0.5, 1.4)
    assert abs(null.iloc[0].to_numpy() @ direction) == pytest.approx(1.0)
    assert np.isinf(fim.std_errors).all()
    assert fim.correlation.isna().all().all()
    eigenvalues, vectors = fim.eigen()
    assert eigenvalues.sum() == pytest.approx(np.trace(fim.matrix))
    assert vectors.shape == (2, 2)

    log = identify.fisher_information(
        _product(), _data(_product()), values, sigma=0.1, transform="log"
    )
    assert log.log_parameters == {"k1", "k2"}
    assert abs(log.non_identifiable().iloc[0].to_numpy() @ [1, -1]) == pytest.approx(
        np.sqrt(2)
    )


def test_log_transform_gives_relative_uncertainty() -> None:
    lin = identify.fisher_information(_decay(), _data(_decay()), {"k": K}, sigma=0.05)
    log = identify.fisher_information(
        _decay(), _data(_decay()), {"k": K}, sigma=0.05, transform="log"
    )
    assert log.std_errors.to_dict()["k"] == pytest.approx(
        lin.std_errors.to_dict()["k"] / (K * np.log(10)), rel=1e-8
    )
    ci = log.wald_ci()
    assert ci.loc["k", "lower"] * ci.loc["k", "upper"] == pytest.approx(K**2)


def test_steady_state_with_sigma_per_observable() -> None:
    model = (
        KineticModelBuilder()
        .add_parameters({"kf": 2.0, "kr": 1.0})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "kf"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
    )
    data = pd.Series({"A": 1 / 3, "vf": 2 / 3})
    fim = identify.fisher_information(
        model, data, {"kf": 2.0, "kr": 1.0}, sigma=pd.Series({"A": 0.01, "vf": 0.02})
    )
    assert fim.rank == 2
    assert list(fim.design.index) == ["A", "vf"]
    # A* = kr / (kf + kr): dA/dkf = -kr / (kf + kr)^2
    assert fim.design.loc["A", "kf"] == pytest.approx(-1 / 9 / 0.01, rel=1e-5)


def test_validation_and_model_untouched() -> None:
    model = _decay()
    data = _data(model)
    with pytest.raises(ValueError, match="positive"):
        identify.fisher_information(model, data, {"k": K}, sigma=-1.0)
    with pytest.raises(ValueError, match="positive"):
        identify.fisher_information(model, data, {"k": K}, sigma=pd.Series({"y": 0.1}))
    with pytest.raises(ValueError, match="more data points"):
        identify.fisher_information(model, data.iloc[:1], {"k": K})
    identify.fisher_information(model, data, {"k": 3.0}, sigma=0.1)
    assert model.get_parameter_values()["k"] == K


def test_finite_differences_detect_non_identifiability() -> None:
    # Only a * c is identifiable from y = c x with x' = a - b x, x(0) = 0
    model = (
        KineticModelBuilder()
        .add_parameters({"a": 1.0, "b": 0.5, "c": 2.0})
        .add_variables({"x": 0.0})
        .add_reaction("v_in", fns.constant, args=["a"], stoichiometry={"x": 1})
        .add_reaction(
            "v_out", fns.mass_action_1s, args=["x", "b"], stoichiometry={"x": -1}
        )
        .add_derived("y", fns.mass_action_1s, args=["x", "c"])
    )
    res = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    data = res.get_combined().loc[T, ["y"]]
    values = {"a": 1.0, "b": 0.5, "c": 2.0}
    for derivatives in ("forward", "finite_difference"):
        fim = identify.fisher_information(
            model,
            data,
            values,
            sigma=0.1,
            runner=KineticRunner(derivatives=derivatives),
        )
        assert fim.rank == 2, derivatives
        assert len(fim.non_identifiable()) == 1
        assert np.isinf(fim.std_errors[["a", "c"]]).all()
        assert np.isfinite(fim.std_errors["b"])


def test_initial_conditions_of_the_fit() -> None:
    data = (
        Simulator(_decay(), y0={"x": 5.0})
        .simulate_time_course(T)
        .get_result()
        .unwrap_or_err()
        .variables.loc[T, ["x"]]
    )
    fim = identify.fisher_information(
        _decay(), data, {"k": K}, sigma=0.05, y0={"x": 5.0}
    )
    assert fim.chi2 == pytest.approx(0.0, abs=1e-10)

    # Fitted initial values are variables in the parameters
    fim = identify.fisher_information(_decay(), data, {"k": K, "x": 5.0}, sigma=0.05)
    assert fim.chi2 == pytest.approx(0.0, abs=1e-10)
    assert list(fim.std_errors.index) == ["k", "x"]
    assert np.isfinite(fim.std_errors).all()
    with pytest.raises(ValueError, match="Unknown"):
        identify.fisher_information(_decay(), data, {"q": 1.0}, sigma=0.05)


def test_more_validation() -> None:
    data = _data(_decay())
    with pytest.raises(ValueError, match="residuals are zero"):
        identify.fisher_information(_decay(), data, {"k": K})
    with pytest.raises(ValueError, match="DataFrame data"):
        identify.fisher_information(
            _decay(), data.iloc[-1], {"k": K}, sigma=data.iloc[[-1]]
        )
