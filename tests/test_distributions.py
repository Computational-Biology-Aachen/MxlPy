# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from mxlpy.distributions import (
    RNG,
    Beta,
    GaussianKde,
    LogNormal,
    LogUniform,
    Normal,
    Skewnorm,
    Uniform,
    sample,
)


def test_beta_sample() -> None:
    # FIXME: implement this
    assert True


def test_distribution_sample() -> None:
    # FIXME: implement this
    assert True


def test_gaussiankde_from_data() -> None:
    # FIXME: implement this
    assert True


def test_gaussiankde_plot() -> None:
    # FIXME: implement this
    assert True


def test_gaussiankde_sample() -> None:
    # FIXME: implement this
    assert True


def test_lognormal_sample() -> None:
    # FIXME: implement this
    assert True


def test_loguniform_sample() -> None:
    # FIXME: implement this
    assert True


def test_normal_sample() -> None:
    # FIXME: implement this
    assert True


def test_sample() -> None:
    samples = sample(
        {
            "beta": Beta(a=1.0, b=1.0),
            "log_normal": LogNormal(mean=1.0, sigma=0.1),
            "log_uniform": LogUniform(lower_bound=0.01, upper_bound=1.0),
            "normal": Normal(loc=1.0, scale=0.1),
            "skewnorm": Skewnorm(loc=1.0, scale=0.1, a=5.0),
            "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
            "kde": GaussianKde.from_data(RNG.normal(loc=1.0, scale=0.5, size=20)),
        },
        n=1,
    )
    assert len(samples.index) == 1
    assert len(samples.columns) == 7


def test_skewnorm_sample() -> None:
    # FIXME: implement this
    assert True


def test_uniform_sample() -> None:
    # FIXME: implement this
    assert True


###############################################################################
# Priors and their penalties
###############################################################################

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import integrate, stats  # noqa: E402

from mxlpy.distributions import Laplace, SoftBounds, prior_penalty  # noqa: E402

XS = [0.05, 0.2, 0.7, 1.0, 2.9, 4.0]


@pytest.mark.parametrize(
    ("dist", "reference"),
    [
        (Normal(1.0, 2.0), lambda x: stats.norm.logpdf(x, 1.0, 2.0)),
        (
            LogNormal(0.3, 0.5),
            lambda x: stats.lognorm.logpdf(x, 0.5, scale=np.exp(0.3)),
        ),
        (Uniform(0.5, 3.0), lambda x: stats.uniform.logpdf(x, 0.5, 2.5)),
        (LogUniform(0.1, 10.0), lambda x: stats.loguniform.logpdf(x, 0.1, 10.0)),
        (Laplace(1.0, 0.5), lambda x: stats.laplace.logpdf(x, 1.0, 0.5)),
    ],
)
def test_log_pdf_matches_scipy(dist: Normal, reference: object) -> None:
    for x in [*XS, -1.0, 20.0]:
        expected = float(reference(x))  # type: ignore[operator]
        assert dist.log_pdf(x) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "dist", [SoftBounds(1.0, 3.0, 0.5), SoftBounds(0.01, 100.0, 0.3, log=True)]
)
def test_soft_bounds_density_and_samples(dist: SoftBounds) -> None:
    if dist.log:
        total = integrate.quad(
            lambda u: math.exp(dist.log_pdf(10**u)) * 10**u * math.log(10),
            -6,
            6,
            limit=200,
        )[0]
        width = 4.0
    else:
        total = integrate.quad(lambda x: math.exp(dist.log_pdf(x)), -10, 15, limit=200)[
            0
        ]
        width = 2.0
    assert total == pytest.approx(1.0, rel=1e-6)
    samples = dist.sample(100_000, np.random.default_rng(1))
    inside = np.mean((samples >= dist.lower) & (samples <= dist.upper))
    assert inside == pytest.approx(
        width / (width + dist.scale * math.sqrt(2 * math.pi)), abs=0.01
    )


def test_soft_bounds_checks_parameters() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        SoftBounds(2.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="positive scale"):
        SoftBounds(1.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="positive bounds"):
        SoftBounds(0.0, 2.0, 0.1, log=True)


@pytest.mark.parametrize(
    "dist",
    [
        Normal(1.0, 2.0),
        LogNormal(0.3, 0.5),
        Laplace(1.0, 0.5),
        SoftBounds(1.0, 3.0, 0.5),
        SoftBounds(0.01, 100.0, 0.3, log=True),
        Beta(2.0, 5.0),
        Skewnorm(1.0, 1.0, 4.0),
        GaussianKde.from_data(np.random.default_rng(1).normal(2.0, 1.0, 300)),
    ],
)
def test_penalty_residual_squares_to_shifted_penalty(dist: Normal) -> None:
    """r² / 2 is the negative log density on the distribution's own scale."""
    own = np.array(
        [
            -dist.log_pdf(x) - (math.log(x) if getattr(dist, "log_scale", False) else 0)
            for x in XS
        ]
    )
    half_squares = np.array([0.5 * dist.penalty_residual(x) ** 2 for x in XS])
    finite = np.isfinite(own)
    offset = own[finite] - half_squares[finite]
    np.testing.assert_allclose(offset, offset[0], atol=1e-9)
    # ... and zero at its minimum, so never negative
    assert (half_squares >= 0).all()
    assert prior_penalty(dist, 1.0) == pytest.approx(
        0.5 * dist.penalty_residual(1.0) ** 2
    )


def test_log_scale_penalties() -> None:
    # LogNormal pulls towards exp(mean), LogUniform is flat
    assert prior_penalty(LogNormal(math.log(0.5), 0.2), 0.5) == 0.0
    assert prior_penalty(LogUniform(0.1, 10.0), 0.2) == 0.0
    assert prior_penalty(LogUniform(0.1, 10.0), 20.0) == math.inf


def test_penalty_of_prior_without_residual() -> None:
    class Custom:
        positive_support = False

        def log_pdf(self, x: float) -> float:
            return -abs(x)

    assert prior_penalty(Custom(), 2.0) == 2.0  # noqa: PLR2004


def test_beta_with_unbounded_density_has_no_penalty() -> None:
    with pytest.raises(ValueError, match="unbounded density"):
        Beta(0.5, 2.0).penalty_residual(0.3)
