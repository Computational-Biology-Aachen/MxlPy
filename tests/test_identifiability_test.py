# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mxlpy import KineticModelBuilder, Simulator, fit, fns, identify
from mxlpy.identify._pulling import _radius_penalty, _starts

SIGMA = 0.01
NLL = fit.GaussianNll(SIGMA)
THRESHOLD = stats.chi2.ppf(0.95, 1) / 2


def _source() -> KineticModelBuilder:
    """Steady state x = p / k, so only p / k is identifiable from x."""
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
        .add_parameters({"k1": 0.5, "k2": 2.0, "a": 1.0, "b": 1.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_derived("kab", fns.mul, args=["a", "b"])
        .add_reaction(
            "v1",
            fns.mass_action_2s,
            args=["x", "k1", "kab"],
            stoichiometry={"x": -1, "y": 1},
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


@pytest.fixture(scope="module")
def decay_data() -> pd.DataFrame:
    t = np.linspace(0.2, 6.0, 20)
    res = Simulator(_decay()).simulate_time_course(t).get_result().unwrap_or_err()
    data = res.get_combined().loc[t, ["x", "y"]]
    return data + np.random.default_rng(0).normal(0, SIGMA, data.shape)


def test_identifiable(decay_data: pd.DataFrame) -> None:
    res = identify.identifiability_test(
        _decay(), decay_data, {"k1": 0.6, "k2": 1.5}, loss_fn=NLL, parallel=False
    )
    assert res.is_identifiable
    assert res.non_identifiable == []
    assert res.fixed == []
    assert len(res.rounds) == 1
    assert res.threshold == pytest.approx(THRESHOLD)
    assert res.log_parameters == ["k1", "k2"]
    assert res.optimum["k1"] == pytest.approx(0.5, rel=0.05)
    rnd = res.rounds[0]
    assert rnd.fits.shape == (10, 4)
    assert rnd.n_blocked == 0
    assert not rnd.fits["flat"].to_numpy().any()
    assert (rnd.fits["delta"].to_numpy() > THRESHOLD).all()
    assert rnd.n_failed == 0
    assert list(rnd.displacement.columns) == ["k1", "k2"]
    assert res.displacement.is_monotonic_decreasing
    assert res.better_optimum is None


def test_flat_direction(decay_data: pd.DataFrame) -> None:
    p0 = {"k1": 0.6, "k2": 1.5, "a": 1.2, "b": 0.9}
    res = identify.identifiability_test(
        _decay(), decay_data, p0, loss_fn=NLL, n_fits=5, parallel=False
    )
    assert not res.is_identifiable
    assert sorted(res.non_identifiable) == ["a", "b", "k1"]
    assert res.fixed == []
    rnd = res.rounds[0]
    assert rnd.fits["flat"].to_numpy().all()
    np.testing.assert_allclose(rnd.fits["distance"], 1.0, rtol=0.05)
    assert (np.abs(rnd.fits["delta"].to_numpy()) < THRESHOLD).all()
    assert res.displacement.index[-1] == "k2"
    assert res.displacement.to_dict()["k2"] < 0.1


def test_recursive(decay_data: pd.DataFrame) -> None:
    p0 = {"k1": 0.6, "k2": 1.5, "a": 1.2, "b": 0.9}
    res = identify.identifiability_test(
        _decay(),
        decay_data,
        p0,
        loss_fn=NLL,
        n_fits=4,
        recursive=True,
        parallel=False,
    )
    assert not res.is_identifiable
    # Fixing two of the three leaves the third identifiable
    assert len(res.fixed) == 2
    assert set(res.fixed) < {"a", "b", "k1"}
    assert [r.fixed for r in res.rounds] == [[], res.fixed[:1], res.fixed]
    assert res.rounds[-1].is_identifiable
    assert sorted(res.non_identifiable) == ["a", "b", "k1"]
    assert list(res.rounds[-1].displacement.columns) == [
        k for k in p0 if k not in res.fixed
    ]
    fixed_values = res.rounds[1].parameters[res.fixed[0]].dropna()
    np.testing.assert_allclose(fixed_values, res.optimum[res.fixed[0]])


def test_steady_state_ratio() -> None:
    res = identify.identifiability_test(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5, "k": 1.0},
        loss_fn=fit.GaussianNll(0.1),
        n_fits=3,
        recursive=True,
        parallel=False,
    )
    assert sorted(res.non_identifiable) == ["k", "p"]
    assert len(res.fixed) == 1
    # p / k stays at 2 in the flat fits
    pars = res.rounds[0].parameters
    np.testing.assert_allclose(pars["p"] / pars["k"], 2.0, rtol=1e-4)


def test_relative_threshold(decay_data: pd.DataFrame) -> None:
    res = identify.identifiability_test(
        _decay(),
        decay_data,
        {"k1": 0.6, "k2": 1.5},
        loss_fn=fit.losses.rmse,
        minimizer=fit.LocalScipyMinimizer(),
        threshold_rtol=0.1,
        n_fits=3,
        parallel=False,
    )
    assert res.threshold == pytest.approx(0.1 * res.objective)
    assert res.is_identifiable


def test_given_threshold_and_linear(decay_data: pd.DataFrame) -> None:
    res = identify.identifiability_test(
        _decay(),
        decay_data,
        {"k1": 0.6, "k2": 1.5, "a": 1.2},
        loss_fn=NLL,
        threshold=5.0,
        radius=0.5,
        transform="linear",
        n_fits=3,
        parallel=False,
    )
    assert res.threshold == 5.0
    assert res.log_parameters == []
    assert sorted(res.non_identifiable) == ["a", "k1"]
    # Linear distances are relative to the optimal values
    pars = res.rounds[0].parameters
    rel = (pars[["k1", "k2", "a"]] - pd.Series(res.optimum)) / pd.Series(res.optimum)
    np.testing.assert_allclose(
        np.linalg.norm(rel.to_numpy(dtype=float), axis=1),
        res.rounds[0].fits["distance"],
    )


def test_blocked_by_bounds_warns(decay_data: pd.DataFrame) -> None:
    with pytest.warns(UserWarning, match="inconclusive"):
        res = identify.identifiability_test(
            _decay(),
            decay_data,
            {"k1": 0.5, "k2": 2.0, "a": 1.0},
            loss_fn=NLL,
            bounds={"k1": (0.45, 0.55), "a": (0.9, 1.1)},
            n_fits=3,
            parallel=False,
        )
    assert res.is_identifiable
    assert res.rounds[0].n_blocked > 0
    assert not res.rounds[0].fits["reached"].to_numpy().any()


def test_parallel_matches_serial() -> None:
    kwargs = {
        "loss_fn": fit.GaussianNll(1.0),
        "n_fits": 2,
    }
    args = (_source(), pd.Series({"x": 2.0}), {"p": 2.5, "k": 1.0})
    serial = identify.identifiability_test(*args, parallel=False, **kwargs)
    par = identify.identifiability_test(*args, parallel=True, max_workers=2, **kwargs)
    pd.testing.assert_frame_equal(serial.rounds[0].fits, par.rounds[0].fits)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"radius": 0.0}, "radius"),
        ({"n_fits": 0}, "n_fits"),
        ({"min_displacement": 0.0}, "min_displacement"),
        ({"min_displacement": 1.5}, "min_displacement"),
        ({"threshold": -1.0}, "threshold"),
        ({"loss_fn": fit.losses.rmse, "threshold_rtol": 0.0}, "threshold_rtol"),
        ({"alpha": 1.5}, "alpha"),
    ],
)
def test_invalid_arguments(kwargs: dict, match: str) -> None:
    kwargs = {"loss_fn": NLL, **kwargs}
    with pytest.raises(ValueError, match=match):
        identify.identifiability_test(
            _source(), pd.Series({"x": 2.0}), {"p": 2.5}, parallel=False, **kwargs
        )


def test_zero_objective_needs_threshold() -> None:
    with pytest.raises(ValueError, match="threshold="):
        identify.identifiability_test(
            _source(),
            pd.Series({"x": 2.0}),
            {"p": 2.5},
            loss_fn=fit.losses.mean_squared,
            parallel=False,
        )


def test_radius_penalty_derivatives() -> None:
    optimum = {"a": 2.0, "b": -0.5, "c": 0.0}
    pen = _radius_penalty(optimum, ["a", "b", "c"], {"a"}, radius=1.0, weight=3.0)
    x = np.array([5.0, -0.2, 0.3])
    u = np.array([math.log10(5.0 / 2.0), (-0.2 + 0.5) / 0.5, 0.3])
    assert pen.distance(x) == pytest.approx(float(np.linalg.norm(u)))
    r = pen.residuals(x)
    assert 0.5 * float(r @ r) == pytest.approx(pen(x))
    jac = pen.jacobian(x)
    h = 1e-6
    fd = np.array(
        [
            (pen.residuals(x + h * e) - pen.residuals(x - h * e))[0] / (2 * h)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(jac[0], fd, rtol=1e-5)
    np.testing.assert_allclose(pen.jacobian(np.array([2.0, -0.5, 0.0])), 0.0)


def test_starts_in_ball_and_bounds() -> None:
    optimum = {"a": 1.0, "b": 2.0}
    pen = _radius_penalty(optimum, ["a", "b"], {"a", "b"}, radius=1.0, weight=1.0)
    starts = _starts(pen, {"a": (0.5, None)}, 200, np.random.default_rng(1))
    values = np.array([[s["a"], s["b"]] for s in starts])
    assert (values[:, 0] >= 0.5).all()
    assert all(pen.distance(v) <= 1.0 + 1e-12 for v in values)
    assert (values[:, 1] < 2.0).any()
    assert (values[:, 1] > 2.0).any()


###############################################################################
# related_parameters
###############################################################################


def test_related_parameters_ratio() -> None:
    profiles = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5, "k": 1.0},
        loss_fn=fit.GaussianNll(1.0),
        transform="log",
        max_steps=30,
        parallel=False,
    )
    assert profiles.log_parameters == ["k", "p"]
    related = identify.related_parameters(profiles)
    assert related.groups == [["k", "p"]]
    assert len(related.relations) == 2
    rel = next(r for r in related.relations if r.profile == "p")
    # log p - log k = log 2
    assert rel.coefficients.to_dict()["p"] == 1.0
    assert rel.coefficients.to_dict()["k"] == pytest.approx(-1.0, rel=1e-3)
    assert rel.r2 > 0.999
    assert rel.collinear == []
    assert str(rel).startswith("1·log10(p) -1·log10(k)")


def test_related_parameters_collinear(decay_data: pd.DataFrame) -> None:
    profiles = identify.profile_likelihood(
        _decay(),
        decay_data,
        {"k1": 0.6, "k2": 1.5, "a": 1.2, "b": 0.9},
        loss_fn=NLL,
        transform="log",
        names=["k1", "k2"],
        max_steps=20,
        parallel=False,
    )
    related = identify.related_parameters(profiles)
    assert [r.profile for r in related.relations] == ["k1"]
    assert related.groups == [["a", "b", "k1"]]
    assert identify.related_parameters(profiles, names=["k2"]).relations == []


def test_related_parameters_invalid() -> None:
    profiles = identify.profile_likelihood(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5},
        loss_fn=fit.GaussianNll(1.0),
        parallel=False,
    )
    with pytest.raises(ValueError, match="No profiles"):
        identify.related_parameters(profiles, names=["k"])
    with pytest.raises(ValueError, match="r2"):
        identify.related_parameters(profiles, r2=0.0)
    assert identify.related_parameters(profiles).relations == []
