"""Probability Distribution Classes for Parameter Sampling.

This module provides a collection of probability distributions used for parameter sampling
in metabolic modeling and Monte Carlo simulations.

Classes:
    Distribution (Protocol): Base protocol for all distribution classes
    Beta: Beta distribution for parameters bounded between 0 and 1
    Uniform: Uniform distribution for parameters with simple bounds
    LogUniform: LogUniform distribution for parameters with simple bounds
    Normal: Normal (Gaussian) distribution for unbounded parameters
    LogNormal: Log-normal distribution for strictly positive parameters
    Skewnorm: Skewed normal distribution for asymmetric parameter distributions

Each distribution class provides:
    - Consistent interface through the sample() method
    - Optional random number generator (RNG) control
    - Reproducible results via seed parameter

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy.plot import Axes, FigAx, _default_fig_ax
from mxlpy.types import Array

__all__ = [
    "Beta",
    "Distribution",
    "GaussianKde",
    "LogNormal",
    "LogUniform",
    "Normal",
    "RNG",
    "Skewnorm",
    "Uniform",
    "sample",
]

RNG = np.random.default_rng(seed=42)


class Distribution(Protocol):
    """Protocol defining interface for distribution classes.

    All distribution classes must implement the sample() method.
    """

    def sample(
        self,
        num: int,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Generate random samples from the distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        Returns
        -------
            Array of random samples

        """
        ...


@dataclass
class Beta:
    """Beta distribution for parameters bounded between 0 and 1.

    Parameters
    ----------
    a
        Alpha shape parameter (>0)
    b
        Beta shape parameter (>0)

    """

    a: float
    b: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the beta distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.beta(self.a, self.b, num)


@dataclass
class Uniform:
    """Uniform distribution for parameters with simple bounds.

    Parameters
    ----------
    lower_bound
        Minimum value
    upper_bound
        Maximum value

    """

    lower_bound: float
    upper_bound: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the uniform distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.uniform(self.lower_bound, self.upper_bound, num)


@dataclass
class LogUniform:
    """LogUniform distribution for parameters with simple bounds.

    Parameters
    ----------
    lower_bound
        Minimum value
    upper_bound
        Maximum value

    """

    lower_bound: float
    upper_bound: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the loguniform distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return cast(
            Array,
            stats.loguniform.rvs(
                self.lower_bound, self.upper_bound, size=num, random_state=rng
            ),
        )


@dataclass
class Normal:
    """Normal (Gaussian) distribution for unbounded parameters.

    Parameters
    ----------
    loc
        Mean of the distribution
    scale
        Standard deviation

    """

    loc: float
    scale: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the normal distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.normal(self.loc, self.scale, num)


@dataclass
class LogNormal:
    """Log-normal distribution for strictly positive parameters.

    Parameters
    ----------
    mean
        Mean of the underlying normal distribution
    sigma
        Standard deviation of the underlying normal distribution
    seed
        Random seed for reproducibility

    """

    mean: float
    sigma: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the log-normal distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.lognormal(self.mean, self.sigma, num)


@dataclass
class Skewnorm:
    """Skewed normal distribution for asymmetric parameter distributions.

    Parameters
    ----------
    loc
        Mean of the distribution
    scale
        Standard deviation
    a
        Skewness parameter

    """

    loc: float
    scale: float
    a: float

    def sample(
        self,
        num: int,
        rng: np.random.Generator | None = None,  # noqa: ARG002
    ) -> Array:
        """Generate random samples from the skewed normal distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            The random generator argument is unused but required for API compatibility

        """
        return cast(
            Array, stats.skewnorm(self.a, loc=self.loc, scale=self.scale).rvs(num)
        )


@dataclass
class GaussianKde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    mean
        Mean of the underlying normal distribution
    sigma
        Standard deviation of the underlying normal distribution
    seed
        Random seed for reproducibility

    """

    kde: stats.gaussian_kde

    @classmethod
    def from_data(cls, data: Array | pd.Series) -> GaussianKde:
        """Create a GaussianKde object from a data array.

        Parameters
        ----------
        data
            Array of data points

        """
        return cls(stats.gaussian_kde(data))

    def plot(
        self,
        xmin: float,
        xmax: float,
        n: int = 1000,
        ax: Axes | None = None,
    ) -> FigAx:
        """Plot the kernel-density estimate.

        Parameters
        ----------
        xmin
            Minimum x value for the plot range.
        xmax
            Maximum x value for the plot range.
        n
            Number of points to evaluate the KDE at.
        ax
            Matplotlib axes to plot on. If None, creates a new figure.

        """
        fig, ax = _default_fig_ax(ax=ax, grid=True, figsize=(5, 3))

        x = np.geomspace(xmin, xmax, n)
        y = self.kde(x)
        ax.set_xlim(xmin, xmax)
        ax.set_xscale("log")
        ax.fill_between(x, y, alpha=0.2)
        ax.plot(x, y)
        ax.grid(visible=True)
        ax.set_frame_on(False)
        return fig, ax

    def sample(
        self,
        num: int,
        rng: np.random.Generator | None = None,  # noqa: ARG002
    ) -> Array:
        """Generate random samples from the kde.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator. Unused but required for API compatibility

        """
        return cast(Array, self.kde.resample(num)[0])


def sample(
    parameters: dict[str, Distribution],
    n: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate samples from the specified distributions.

    Examples
    --------
        >>> sample({"beta": Beta(a=1.0, b=1.0),
        ...         "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
        ...         "normal": Normal(loc=1.0, scale=0.1),
        ...         "log_normal": LogNormal(mean=1.0, sigma=0.1),
        ...         "skewnorm": Skewnorm(loc=1.0, scale=0.1, a=5.0),},
        ...         n=2,)
                   beta   uniform    normal  log_normal  skewnorm
            0  0.253043  0.682496  1.067891    2.798020  1.216259
            1  0.573357  0.139752  1.006758    2.895416  1.129373

    Parameters
    ----------
    parameters
        Dictionary mapping parameter names to distribution objects.
    n
        Number of samples to generate.
    rng
        Random number generator.

    Returns: DataFrame containing the generated samples.

    """
    return pd.DataFrame({k: v.sample(n, rng) for k, v in parameters.items()})


if __name__ == "__main__":
    _ = sample(
        {
            "beta": Beta(a=1.0, b=1.0),
            "uniform": Uniform(lower_bound=0.0, upper_bound=1.0),
            "normal": Normal(loc=1.0, scale=0.1),
            "log_normal": LogNormal(mean=1.0, sigma=0.1),
            "skewnorm": Skewnorm(loc=1.0, scale=0.1, a=5.0),
        },
        n=1,
    )
