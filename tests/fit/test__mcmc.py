# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Bayesian posterior sampling (#120)."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mxlpy import Model, Simulator, fit, fns, make_protocol
from mxlpy.distributions import (
    Beta,
    LogNormal,
    LogUniform,
    Normal,
    Skewnorm,
    Uniform,
)
from mxlpy.fit import AdaptiveMetropolis, McmcResult
from mxlpy.fit._mcmc import _ess, _r_hat

mpl.use("Agg")

###############################################################################
# Priors
###############################################################################


@pytest.mark.parametrize(
    ("dist", "x", "expected", "positive"),
    [
        (Beta(2, 3), 0.3, stats.beta.logpdf(0.3, 2, 3), True),
        (Uniform(1, 3), 2.0, np.log(0.5), True),
        (Uniform(-1, 1), 0.0, np.log(0.5), False),
        (LogUniform(0.1, 10), 1.0, stats.loguniform.logpdf(1.0, 0.1, 10), True),
        (Normal(1, 2), 0.5, stats.norm.logpdf(0.5, 1, 2), False),
        (LogNormal(0, 1), 2.0, stats.lognorm.logpdf(2.0, 1), True),
        (Skewnorm(0, 1, 2), 0.1, stats.skewnorm.logpdf(0.1, 2), False),
    ],
)
def test_prior_log_pdf(dist, x: float, expected: float, positive: bool) -> None:  # noqa: ANN001, FBT001
    assert dist.log_pdf(x) == pytest.approx(expected)
    assert dist.positive_support is positive


def test_prior_log_pdf_outside_support() -> None:
    assert LogNormal(0, 1).log_pdf(-1.0) == -np.inf
    assert Uniform(1, 3).log_pdf(4.0) == -np.inf


###############################################################################
# Sampler
###############################################################################

_COV = np.array([[1.0, 0.8], [0.8, 2.0]])
_INV = np.linalg.inv(_COV)


def _gaussian(z: np.ndarray) -> float:
    return float(-0.5 * z @ _INV @ z)


def test_adaptive_metropolis_samples_target() -> None:
    chain = AdaptiveMetropolis(n_warmup=2000)(
        _gaussian, np.zeros(2), 20_000, np.random.default_rng(0)
    )
    assert chain.samples.shape == (20_000, 2)
    assert chain.samples.mean(axis=0) == pytest.approx([0, 0], abs=0.1)
    assert np.cov(chain.samples, rowvar=False) == pytest.approx(_COV, abs=0.15)
    # The optimal acceptance rate in 2D is ~0.35
    assert 0.2 < chain.acceptance_rate < 0.6


def test_adaptive_metropolis_recovers_from_tiny_initial_scale() -> None:
    chain = AdaptiveMetropolis(n_warmup=2000, initial_scale=1e-4)(
        _gaussian, np.zeros(2), 5000, np.random.default_rng(1)
    )
    assert np.cov(chain.samples, rowvar=False) == pytest.approx(_COV, abs=0.4)


def test_adaptive_metropolis_is_reproducible() -> None:
    a, b = (
        AdaptiveMetropolis()(_gaussian, np.zeros(2), 200, np.random.default_rng(3))
        for _ in range(2)
    )
    np.testing.assert_array_equal(a.samples, b.samples)


###############################################################################
# Diagnostics
###############################################################################


def test_diagnostics_of_independent_draws() -> None:
    draws = np.random.default_rng(0).normal(size=(4, 1000))
    assert _r_hat(draws) == pytest.approx(1.0, abs=0.01)
    assert _ess(draws) == pytest.approx(4000, rel=0.15)


def test_diagnostics_detect_unmixed_chains() -> None:
    draws = np.random.default_rng(0).normal(size=(4, 1000)) + np.arange(4)[:, None]
    assert _r_hat(draws) > 1.5


def test_ess_of_autocorrelated_draws_is_smaller() -> None:
    rng = np.random.default_rng(0)
    draws = np.zeros((4, 2000))
    for t in range(1, 2000):
        draws[:, t] = 0.9 * draws[:, t - 1] + rng.normal(size=4)
    # AR(1) with phi = 0.9: ESS ~ N (1 - phi) / (1 + phi)
    assert _ess(draws) == pytest.approx(8000 * 0.1 / 1.9, rel=0.3)


###############################################################################
# Posterior sampling
###############################################################################


def _model() -> Model:
    return (
        Model()
        .add_variable("x", 1.0)
        .add_parameters({"k_in": 1.0, "k_out": 0.5})
        .add_reaction("vin", fns.constant, args=["k_in"], stoichiometry={"x": 1})
        .add_reaction(
            "vout", fns.mass_action_1s, args=["x", "k_out"], stoichiometry={"x": -1}
        )
    )


def _data(sigma: float = 0.05) -> pd.DataFrame:
    truth = (
        Simulator(_model())
        .simulate_time_course(np.linspace(0, 10, 21))
        .get_result()
        .unwrap_or_err()
        .variables
    )
    return truth + np.random.default_rng(1).normal(0, sigma, truth.shape)


_PRIOR = {"k_in": LogNormal(0, 1), "k_out": LogNormal(0, 1)}


def _sample(**kwargs: object) -> McmcResult:
    settings: dict = {
        "data": _data(),
        "prior": _PRIOR,
        "sigma": 0.05,
        "n_samples": 500,
        "n_chains": 2,
        "seed": 0,
        "parallel": False,
    } | kwargs
    return fit.mcmc_time_course(_model(), **settings).unwrap_or_err()


def test_time_course_recovers_parameters() -> None:
    res = _sample(n_chains=4, n_samples=600, parallel=True)
    assert res.chains.shape == (4, 600, 2)
    assert res.log_posterior.shape == (4, 600)
    ci = res.credible_intervals()
    assert (ci["lower"] < pd.Series({"k_in": 1.0, "k_out": 0.5})).all()
    assert (ci["upper"] > pd.Series({"k_in": 1.0, "k_out": 0.5})).all()
    assert (res.r_hat() < 1.1).all()
    assert (res.ess() > 100).all()


def test_seed_makes_samples_reproducible() -> None:
    first, again, other = (
        _sample(n_samples=50, seed=seed).chains for seed in (0, 0, 1)
    )
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_parallel_chains_match_sequential() -> None:
    np.testing.assert_array_equal(
        _sample(n_samples=100).chains, _sample(n_samples=100, parallel=True).chains
    )


def test_log_posterior_is_in_natural_units() -> None:
    res = _sample(n_samples=50, n_chains=1)
    theta = res.chains[0, -1]
    data = _data()
    pred = (
        Simulator(
            _model().update_parameters(
                dict(zip(res.parameter_names, theta, strict=True))
            )
        )
        .simulate_time_course(data.index)
        .get_result()
        .unwrap_or_err()
        .variables
    )
    log_lik = stats.norm.logpdf(data.to_numpy(), pred.to_numpy(), 0.05).sum()
    log_prior = sum(
        _PRIOR[k].log_pdf(v) for k, v in zip(res.parameter_names, theta, strict=True)
    )
    assert res.log_posterior[0, -1] == pytest.approx(log_lik + log_prior, rel=1e-4)


def test_linear_priors_can_be_negative() -> None:
    # Sampled linearly, a Normal prior must not be squeezed onto (0, inf)
    model = _model().add_parameter("offset", 0.0)
    res = fit.mcmc_time_course(
        model,
        data=_data(),
        prior={"offset": Normal(0, 1)},
        sigma=0.05,
        n_samples=300,
        n_chains=1,
        seed=0,
        parallel=False,
    ).unwrap_or_err()
    assert (res.chains < 0).any()


def test_missing_data_is_ignored() -> None:
    data = _data()
    data.iloc[::2] = np.nan
    res = _sample(data=data, n_samples=200)
    assert np.isfinite(res.log_posterior).all()


def test_per_column_sigma() -> None:
    res = _sample(sigma=pd.Series({"x": 0.05}), n_samples=50)
    np.testing.assert_allclose(res.chains, _sample(n_samples=50).chains)


def test_steady_state() -> None:
    # x* = k_in / k_out; with k_out fixed, k_in is identifiable
    res = fit.mcmc_steady_state(
        _model(),
        data=pd.Series({"x": 2.0}),
        prior={"k_in": LogNormal(0, 1)},
        sigma=0.1,
        n_samples=500,
        n_chains=2,
        seed=0,
        parallel=False,
    ).unwrap_or_err()
    assert res.credible_intervals().loc["k_in", "median"] == pytest.approx(1.0, abs=0.1)


def test_protocol_time_course() -> None:
    protocol = make_protocol([(5, {"k_in": 1.0}), (5, {"k_in": 2.0})])
    data = (
        Simulator(_model())
        .simulate_protocol_time_course(protocol, time_points=np.linspace(0, 10, 21))
        .get_result()
        .unwrap_or_err()
        .variables
    )
    res = fit.mcmc_protocol_time_course(
        _model(),
        data=data,
        protocol=protocol,
        prior={"k_out": LogNormal(0, 1)},
        sigma=0.05,
        n_samples=300,
        n_chains=1,
        seed=0,
        parallel=False,
    ).unwrap_or_err()
    assert res.credible_intervals().loc["k_out", "median"] == pytest.approx(
        0.5, abs=0.05
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prior": {}}, "prior is empty"),
        ({"x0": {"k_other": 1.0}}, "without a prior"),
        ({"prior": {"nope": LogNormal(0, 1)}}, "neither a parameter nor a variable"),
        ({"x0": {"k_in": -1.0}}, "starting point isn't finite"),
        ({"sigma": 0.0}, "sigma has to be positive"),
    ],
)
def test_invalid_input_is_an_err(kwargs: dict, match: str) -> None:
    settings: dict = {
        "data": _data(),
        "prior": _PRIOR,
        "sigma": 0.05,
        "n_samples": 10,
        "n_chains": 1,
        "parallel": False,
    } | kwargs
    res = fit.mcmc_time_course(_model(), **settings)
    with pytest.raises(ValueError, match=match):
        res.unwrap_or_err()


def test_result_tables_and_plots() -> None:
    res = _sample(n_samples=100)
    samples = res.samples()
    assert samples.shape == (200, 2)
    assert list(samples.index.names) == ["chain", "draw"]
    assert list(res.credible_intervals(0.5).columns) == ["lower", "median", "upper"]
    fig, _ = res.plot_traces()
    assert fig is not None
    fig, _ = res.plot_marginals()
    assert fig is not None


def test_data_is_aligned_by_time_not_position() -> None:
    # Data without t = 0: the prediction has an extra row, matched by label
    res = _sample(data=_data().iloc[1:], n_samples=50)
    assert np.isfinite(res.log_posterior).all()


def test_protocol_data_between_protocol_steps() -> None:
    protocol = make_protocol([(5, {"k_in": 1.0}), (5, {"k_in": 2.0})])
    data = (
        Simulator(_model())
        .simulate_protocol_time_course(protocol, time_points=[1.0, 3.0, 7.0, 9.0])
        .get_result()
        .unwrap_or_err()
        .variables
    ).loc[[1.0, 3.0, 7.0, 9.0]]
    res = fit.mcmc_protocol_time_course(
        _model(),
        data=data,
        protocol=protocol,
        prior={"k_out": LogNormal(0, 1)},
        sigma=0.05,
        n_samples=50,
        n_chains=1,
        seed=0,
        parallel=False,
    ).unwrap_or_err()
    assert np.isfinite(res.log_posterior).all()


@pytest.mark.parametrize(
    ("model_kwargs", "kwargs", "match"),
    [
        # Start on the boundary of a log-sampled prior: log(0) = -inf
        ({}, {"prior": {"k_in": Uniform(0, 2)}, "x0": {"k_in": 0.0}}, "isn't finite"),
        # A data column the model doesn't have
        ({"rename": {"x": "unknown"}}, {}, "isn't finite"),
        # Per-point sigma for steady-state data
        ({"steady_state": True}, {"sigma": pd.DataFrame({"x": [0.1]})}, "time course"),
    ],
)
def test_more_invalid_input_is_an_err(
    model_kwargs: dict, kwargs: dict, match: str
) -> None:
    data = _data()
    if (rename := model_kwargs.get("rename")) is not None:
        data = data.rename(columns=rename)
    settings: dict = {
        "prior": {"k_in": LogNormal(0, 1)},
        "sigma": 0.05,
        "n_samples": 10,
        "n_chains": 1,
        "parallel": False,
    } | kwargs
    if model_kwargs.get("steady_state"):
        res = fit.mcmc_steady_state(_model(), data=data.iloc[-1], **settings)
    else:
        res = fit.mcmc_time_course(_model(), data=data, **settings)
    with pytest.raises(ValueError, match=match):
        res.unwrap_or_err()


def test_y0_sets_the_default_start_of_sampled_variables() -> None:
    res = _sample(
        prior={"x": LogNormal(0, 1)},
        y0={"x": 5.0},
        n_samples=10,
        n_chains=1,
        sampler=AdaptiveMetropolis(n_warmup=0, initial_scale=1e-12),
    )
    assert res.chains[0, 0, 0] == pytest.approx(5.0, rel=0.5)


def test_linear_parameters_at_zero_do_not_warn() -> None:
    model = _model().add_parameter("offset", 0.0)
    with np.errstate(all="raise"):
        fit.mcmc_time_course(
            model,
            data=_data(),
            prior={"offset": Normal(0, 1)},
            sigma=0.05,
            n_samples=5,
            n_chains=1,
            seed=0,
            parallel=False,
        ).unwrap_or_err()
