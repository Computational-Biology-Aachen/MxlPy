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
