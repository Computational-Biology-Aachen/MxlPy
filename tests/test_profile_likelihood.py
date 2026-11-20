# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mxlpy import KineticModelBuilder, Simulator, fit, fns, identify, make_protocol
from mxlpy.distributions import Normal, Uniform
from mxlpy.fit.penalties import PriorPenalty, fix_penalties
from mxlpy.types import Array

NLL = fit.GaussianNll(1.0)
THRESHOLD = stats.chi2.ppf(0.95, 1) / 2
SIGMA = 0.01


def _source() -> KineticModelBuilder:
    """Steady state x = p / k."""
    return (
        KineticModelBuilder()
        .add_parameters({"p": 3.0, "k": 1.0})
        .add_variables({"x": 1.0})
        .add_reaction("vin", fns.constant, args=["p"], stoichiometry={"x": 1})
        .add_reaction(
            "vout", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1}
        )
    )


def _decay() -> KineticModelBuilder:
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


def _decay_data(seed: int = 0) -> pd.DataFrame:
    t = np.linspace(0.2, 6.0, 20)
    res = Simulator(_decay()).simulate_time_course(t).get_result().unwrap_or_err()
    data = res.get_combined().loc[t, ["x", "y"]]
    return data + np.random.default_rng(seed).normal(0, SIGMA, data.shape)


def _decay_fit(data: pd.DataFrame, p0: dict[str, float]) -> fit.Fit:
    return fit.time_course(
        _decay(),
        p0=p0,
        data=data,
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=fit.GaussianNll(SIGMA),
    ).unwrap_or_err()


def test_analytic_interval() -> None:
    # Δ = (p - 2)² / 2, so the bounds are 2 ± sqrt(2 * threshold)
    pl = identify.profile_likelihood(
        _source(), pd.Series({"x": 2.0}), {"p": 2.5}, loss_fn=NLL, parallel=False
    )
    prof = pl["p"]
    half = math.sqrt(2 * THRESHOLD)
    assert pl.optimum["p"] == pytest.approx(2.0, rel=1e-6)
    assert pl.objective == pytest.approx(0.0, abs=1e-10)
    assert pl.alpha == 0.95
    assert pl.threshold == pytest.approx(THRESHOLD)
    assert prof.ci_lower == pytest.approx(2 - half, rel=2e-3)
    assert prof.ci_upper == pytest.approx(2 + half, rel=2e-3)
    assert prof.identifiability == "identifiable"
    assert prof.delta.index.is_monotonic_increasing
    assert prof.delta.loc[pl.optimum["p"]] == 0.0
    np.testing.assert_allclose(
        prof.delta, (prof.delta.index.to_numpy() - 2) ** 2 / 2, atol=1e-8
    )
    assert pl.better_optimum is None


@pytest.mark.parametrize("transform", [None, "log"])
def test_profiled_prior_stays_in_the_objective(transform: str | None) -> None:
    # (p - 4)² / 2 + p² / 2 = (p - 2)² + 4: optimum 2, bounds 2 ± sqrt(threshold)
    pl = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 4.0}),
        {"p": 2.5},
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        transform=transform,  # type: ignore[arg-type]
        parallel=False,
    )
    assert pl.objective == pytest.approx(4.0, rel=1e-6)
    assert pl["p"].log == (transform == "log")
    assert pl["p"].ci_lower == pytest.approx(2 - math.sqrt(THRESHOLD), rel=2e-3)
    assert pl["p"].ci_upper == pytest.approx(2 + math.sqrt(THRESHOLD), rel=2e-3)


def test_structurally_non_identifiable() -> None:
    # Only the ratio p / k is determined
    pl = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.0, "k": 1.0},
        loss_fn=NLL,
        transform="log",
        parallel=False,
    )
    for name in ("p", "k"):
        prof = pl[name]
        assert prof.identifiability == "structurally_non_identifiable"
        assert (prof.lower_status, prof.upper_status) == ("bound", "bound")
        assert prof.ci_lower == -math.inf
        assert prof.ci_upper == math.inf
        # Profiles span three orders of magnitude in both directions
        assert prof.delta.index.min() == pytest.approx(prof.value / 1e3)
        assert prof.delta.index.max() == pytest.approx(prof.value * 1e3)
    # The other parameter compensates along the profile
    path = pl["p"].parameters
    np.testing.assert_allclose(path["p"] / path["k"], 2.0, rtol=1e-4)
    assert (pl.ci["identifiability"] == "structurally_non_identifiable").all()


def test_practically_non_identifiable() -> None:
    # Measured long after the decay: faster decay can't be excluded
    model = (
        KineticModelBuilder()
        .add_parameter("k", 3.0)
        .add_variable("x", 1.0)
        .add_reaction("v", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1})
    )
    t = np.array([1.0, 1.5, 2.0])
    data = pd.DataFrame({"x": np.exp(-3.0 * t)}, index=t)
    pl = identify.profile_likelihood(
        model,
        data,
        {"k": 3.0},
        loss_fn=fit.GaussianNll(0.05),
        transform="log",
        bounds={"k": (0.1, 100.0)},
        parallel=False,
    )
    prof = pl["k"]
    assert prof.identifiability == "practically_non_identifiable"
    assert (prof.lower_status, prof.upper_status) == ("threshold", "bound")
    assert 1.0 < prof.ci_lower < 3.0
    assert prof.ci_upper == math.inf
    assert prof.delta.index.max() == pytest.approx(100.0)


def test_time_course_matches_wald_intervals() -> None:
    # Nearly linear around the optimum, so the intervals agree
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 2.5})
    pl = identify.profile_likelihood(
        _decay(), data, res, loss_fn=fit.GaussianNll(SIGMA)
    )
    wald = identify.fisher_information(_decay(), data, res, sigma=SIGMA).wald_ci()
    ci = pl.ci
    assert list(ci.columns) == ["value", "lower", "upper", "identifiability"]
    assert (ci["identifiability"] == "identifiable").all()
    np.testing.assert_allclose(ci["lower"], wald["lower"], rtol=2e-2)
    np.testing.assert_allclose(ci["upper"], wald["upper"], rtol=2e-2)
    assert pl.objective == pytest.approx(res.objective, rel=1e-6)


def test_initial_value_and_subset() -> None:
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 2.5, "x": 1.5})
    pl = identify.profile_likelihood(
        _decay(),
        data,
        res,
        loss_fn=fit.GaussianNll(SIGMA),
        names=["x"],
        parallel=False,
    )
    assert list(pl.profiles) == ["x"]
    prof = pl["x"]
    assert prof.identifiability == "identifiable"
    assert prof.ci_lower < 2.0 < prof.ci_upper
    assert list(prof.parameters.columns) == ["k1", "k2", "x"]


def test_heuristic_loss_needs_threshold() -> None:
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 2.5})
    with pytest.raises(ValueError, match="likelihood"):
        identify.profile_likelihood(_decay(), data, res, loss_fn=fit.mean_squared)
    pl = identify.profile_likelihood(
        _decay(),
        data,
        res,
        loss_fn=fit.mean_squared,
        threshold=1e-4,
        names=["k1"],
        parallel=False,
    )
    assert pl.alpha is None
    assert pl.threshold == 1e-4
    assert pl["k1"].identifiability == "identifiable"


def test_coupling_penalty_with_fixed_parameter() -> None:
    class Difference:
        names = ("k1", "k2")

        def __call__(self, values: Array) -> float:
            return float(values[0] - values[1]) ** 2

        def residuals(self, values: Array) -> Array:
            return np.array([math.sqrt(2) * (values[0] - values[1])])

    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 2.5})
    plain = identify.profile_likelihood(
        _decay(), data, res, loss_fn=fit.GaussianNll(SIGMA), parallel=False
    )
    penalised = identify.profile_likelihood(
        _decay(),
        data,
        res,
        loss_fn=fit.GaussianNll(SIGMA),
        penalty=Difference(),
        parallel=False,
    )
    # The penalty pulls the parameters together, but the data dominates
    assert penalised.objective > plain.objective
    for name in ("k1", "k2"):
        assert penalised[name].identifiability == "identifiable"
        assert penalised[name].ci_lower == pytest.approx(plain[name].ci_lower, rel=5e-2)


def test_fix_penalties() -> None:
    prior = PriorPenalty({"a": Normal(0, 1), "b": Normal(1, 2), "c": Normal(2, 3)})
    only_a = PriorPenalty({"a": Normal(0, 1)})
    free, constant = fix_penalties([prior, only_a], {"a": 0.5, "c": 1.0})
    assert len(free) == 1
    assert constant == pytest.approx(only_a(np.array([0.5])))
    (fixed,) = free
    assert list(fixed.names) == ["b"]
    full = np.array([0.5, 3.0, 1.0])
    assert fixed(np.array([3.0])) == pytest.approx(prior(full))
    np.testing.assert_allclose(
        fixed.residuals(np.array([3.0])),  # type: ignore[attr-defined]
        prior.residuals(full),
    )
    np.testing.assert_allclose(
        fixed.jacobian(np.array([3.0])),  # type: ignore[attr-defined]
        prior.jacobian(full)[:, [1]],
    )


def test_protocol_time_course() -> None:
    protocol = make_protocol([(2, {"k1": 0.5}), (2, {"k1": 1.5})])
    t = np.linspace(0.2, 4.0, 15)
    data = (
        Simulator(_decay())
        .simulate_protocol_time_course(protocol, time_points=t)
        .get_result()
        .unwrap_or_err()
        .get_combined()
        .loc[t, ["y"]]
    )
    pl = identify.profile_likelihood(
        _decay(),
        data,
        {"k2": 2.0},
        loss_fn=fit.GaussianNll(SIGMA),
        protocol=protocol,
        parallel=False,
    )
    assert pl["k2"].ci_lower < 2.0 < pl["k2"].ci_upper


def test_validation() -> None:
    data = pd.Series({"x": 2.0})
    with pytest.raises(ValueError, match="fitted ones"):
        identify.profile_likelihood(
            _source(), data, {"p": 2.0}, loss_fn=NLL, names=["k"]
        )
    with pytest.raises(ValueError, match="Unknown"):
        identify.profile_likelihood(_source(), data, {"q": 2.0}, loss_fn=NLL)
    with pytest.raises(ValueError, match="alpha"):
        identify.profile_likelihood(_source(), data, {"p": 2.0}, loss_fn=NLL, alpha=1.0)
    with pytest.raises(ValueError, match="threshold"):
        identify.profile_likelihood(
            _source(), data, {"p": 2.0}, loss_fn=NLL, threshold=-1.0
        )
    with pytest.raises(ValueError, match="max_steps"):
        identify.profile_likelihood(
            _source(), data, {"p": 2.0}, loss_fn=NLL, max_steps=0
        )


def test_max_steps_is_inconclusive() -> None:
    pl = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.0},
        loss_fn=NLL,
        max_steps=2,
        parallel=False,
    )
    prof = pl["p"]
    assert (prof.lower_status, prof.upper_status) == ("max_steps", "max_steps")
    assert prof.identifiability == "inconclusive"
    assert len(prof.delta) <= 5


def test_model_untouched() -> None:
    model = _source()
    identify.profile_likelihood(
        model, pd.Series({"x": 2.0}), {"p": 2.5}, loss_fn=NLL, parallel=False
    )
    assert model.get_parameter_values() == {"p": 3.0, "k": 1.0}


def test_better_optimum_is_reported() -> None:
    class DoubleWell:
        """Local minimum near d = 1, the global one near d = -1."""

        names = ("d",)

        def __call__(self, values: Array) -> float:
            d = float(values[0])
            return (d**2 - 1) ** 2 + 0.1 * d

    model = _source().add_parameter("d", 1.0)
    with pytest.warns(UserWarning, match="better fit"):
        pl = identify.profile_likelihood(
            model,
            pd.Series({"x": 2.0}),
            {"p": 2.0, "d": 1.0},
            loss_fn=NLL,
            penalty=DoubleWell(),
            minimizer=fit.LocalScipyMinimizer(),
            names=["d"],
            bounds={"d": (-2.0, 2.0)},
            parallel=False,
        )
    assert pl.better_optimum is not None
    assert pl.better_optimum["d"] == pytest.approx(-1.0, abs=0.15)
    assert pl["d"].delta.min() < -0.1


@pytest.mark.parametrize("x", [0.5, 1e-6])
def test_linear_profile_crosses_zero(x: float) -> None:
    # Δ = (p - x)² / 2 with an optimum close to zero: the interval x ± 1.96
    # includes negative values
    pl = identify.profile_likelihood(
        _source(), pd.Series({"x": x}), {"p": 1.0}, loss_fn=NLL, parallel=False
    )
    half = math.sqrt(2 * THRESHOLD)
    assert pl["p"].identifiability == "identifiable"
    assert pl["p"].ci_lower == pytest.approx(x - half, rel=2e-3)
    assert pl["p"].ci_upper == pytest.approx(x + half, rel=2e-3)


def test_prior_support_bounds_the_profile() -> None:
    # Only p / k is determined, the uniform prior's support limits p like bounds
    pl = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.0, "k": 1.0},
        loss_fn=NLL,
        prior={"p": Uniform(0.5, 8.0)},
        transform="log",
        names=["p"],
        parallel=False,
    )
    prof = pl["p"]
    assert prof.identifiability == "structurally_non_identifiable"
    assert (prof.lower_status, prof.upper_status) == ("bound", "bound")
    assert np.isfinite(prof.delta).all()
    assert not prof.delta.index.duplicated().any()
    assert prof.delta.index.min() == pytest.approx(0.5, rel=1e-2)
    assert prof.delta.index.max() == pytest.approx(8.0, rel=1e-2)


def test_custom_likelihood_loss() -> None:
    class Nll:
        """The same likelihood as GaussianNll(SIGMA), marked as one."""

        is_likelihood = True

        def __call__(self, y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
            return 0.5 * float((((y_pred - y_true) / SIGMA) ** 2).sum().sum())

    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 2.5})
    custom = identify.profile_likelihood(
        _decay(),
        data,
        res,
        loss_fn=Nll(),
        minimizer=fit.LocalScipyMinimizer(),
        names=["k1"],
        parallel=False,
    )
    gaussian = identify.profile_likelihood(
        _decay(),
        data,
        res,
        loss_fn=fit.GaussianNll(SIGMA),
        names=["k1"],
        parallel=False,
    )
    assert custom.objective == pytest.approx(gaussian.objective, rel=1e-4)
    assert custom["k1"].ci_lower == pytest.approx(gaussian["k1"].ci_lower, rel=1e-3)
    assert custom["k1"].ci_upper == pytest.approx(gaussian["k1"].ci_upper, rel=1e-3)
