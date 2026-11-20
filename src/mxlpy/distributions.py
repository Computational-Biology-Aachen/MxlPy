# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

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
    Laplace: Laplace distribution, an L1 (sparsity) penalty as a prior
    SoftBounds: Flat inside bounds with Gaussian tails outside, soft box constraints
    Skewnorm: Skewed normal distribution for asymmetric parameter distributions

Each distribution class provides:
    - Consistent interface through the sample() method
    - Optional random number generator (RNG) control
    - Reproducible results via seed parameter
    - `log_pdf`, so it can be used as a prior (`mxlpy.fit.mcmc_*`, and the
      `prior` of the point-estimate fits, see `Prior`)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy.plot import Axes, FigAx, _default_fig_ax
from mxlpy.types import Array

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Beta",
    "Distribution",
    "GaussianKde",
    "Laplace",
    "LogNormal",
    "LogUniform",
    "Normal",
    "Prior",
    "RNG",
    "Skewnorm",
    "SoftBounds",
    "Uniform",
    "prior_penalty",
    "sample",
]

RNG = np.random.default_rng(seed=42)


_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


class Prior(Protocol):
    """A distribution usable as a Bayesian prior.

    Used by `mxlpy.fit.mcmc_*` and, as a penalty, by the point-estimate fits
    (`prior=` of e.g. `mxlpy.fit.time_course`, see `prior_penalty`).

    Optional members refine the penalty of a point-estimate fit:

    - `log_scale: bool`: the distribution is defined on `log(x)` (e.g.
      `LogNormal`, `LogUniform`), so its penalty is the negative log density
      of `log(x)`, without the `1/x` of the change of variables.
    - `penalty_residual(x) -> float`: the signed pseudo-residual `r` with
      `r² / 2` = penalty - its minimum, used by least-squares minimizers.
      Without it, least-squares minimizers can't use the prior.
    """

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        ...

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf)."""
        ...


def prior_penalty(prior: Prior, x: float) -> float:
    """Penalty of a prior in a point-estimate (MAP) fit.

    The negative log density on the distribution's own scale: of `x`, or of
    `log(x)` for distributions defined on a log scale (`log_scale`), so that
    e.g. `LogNormal(log(0.5), s)` pulls towards 0.5 and `LogUniform` is flat.
    For the built-in log-scale priors this is the mode of the posterior in the
    log space `mxlpy.fit.mcmc_*` samples them in (MCMC also samples other
    priors with `positive_support`, like `Uniform(0, ...)`, in log space, but
    they are penalised on `x`). Shifted to be zero at its minimum if the prior
    provides a `penalty_residual`, else up to a constant.

    Parameters
    ----------
    prior
        The prior distribution
    x
        Parameter value

    """
    if callable(residual := getattr(prior, "penalty_residual", None)):
        return 0.5 * float(cast(float, residual(x))) ** 2
    value = -prior.log_pdf(x)
    if getattr(prior, "log_scale", False):
        value -= math.log(x) if x > 0 else -math.inf
    return value


def _sqrt_residual(log_pdf: float, log_pdf_mode: float, x: float, mode: float) -> float:
    """Signed pseudo-residual `sign(x - mode) sqrt(2 (log p(mode) - log p(x)))`."""
    if not math.isfinite(log_pdf):
        return math.inf if x >= mode else -math.inf
    return math.copysign(math.sqrt(max(2 * (log_pdf_mode - log_pdf), 0.0)), x - mode)


def _box_residual(x: float, lower: float, upper: float) -> float:
    """Zero inside the bounds, infinite outside."""
    if lower <= x <= upper:
        return 0.0
    return -math.inf if x < lower else math.inf


def _soft_residual(u: float, lower: float, upper: float, scale: float) -> float:
    """Zero inside the bounds, the distance to them in units of `scale` outside."""
    if u > upper:
        return (u - upper) / scale
    if u < lower:
        return (u - lower) / scale
    return 0.0


def _numeric_mode(
    log_pdf: Callable[[float], float], start: float, width: float
) -> float:
    """Maximum of a univariate log density close to `start`."""
    from scipy.optimize import minimize_scalar  # noqa: PLC0415

    res = minimize_scalar(lambda x: -log_pdf(x), bracket=(start - width, start + width))
    return float(res.x)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        return float(stats.beta.logpdf(x, self.a, self.b))

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return True

    @property
    def mode(self) -> float:
        """Most likely value. Undefined if a or b is below 1 (unbounded density)."""
        a, b = self.a, self.b
        if a < 1 or b < 1:
            msg = f"Beta(a={a}, b={b}) has an unbounded density and no mode, so it can't be a penalty"
            raise ValueError(msg)
        if a == 1 and b == 1:
            return 0.5
        return (a - 1) / (a + b - 2)

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual of the penalty, see `Prior`."""
        mode = self.mode
        return _sqrt_residual(self.log_pdf(x), self.log_pdf(mode), x, mode)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        if self.lower_bound <= x <= self.upper_bound:
            return -math.log(self.upper_bound - self.lower_bound)
        return -math.inf

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return self.lower_bound >= 0

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual of the penalty (zero inside the bounds), see `Prior`."""
        return _box_residual(x, self.lower_bound, self.upper_bound)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        if self.lower_bound <= x <= self.upper_bound:
            return -math.log(x) - math.log(
                math.log(self.upper_bound / self.lower_bound)
            )
        return -math.inf

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return True

    @property
    def log_scale(self) -> bool:
        """Defined on log(x), so its penalty is flat inside the bounds."""
        return True

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual of the penalty (zero inside the bounds), see `Prior`."""
        return _box_residual(x, self.lower_bound, self.upper_bound)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        z = (x - self.loc) / self.scale
        return -0.5 * z * z - math.log(self.scale) - _LOG_SQRT_2PI

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return False

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual `(x - loc) / scale` of the penalty, see `Prior`."""
        return (x - self.loc) / self.scale


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        if x <= 0:
            return -math.inf
        log_x = math.log(x)
        z = (log_x - self.mean) / self.sigma
        return -0.5 * z * z - log_x - math.log(self.sigma) - _LOG_SQRT_2PI

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return True

    @property
    def log_scale(self) -> bool:
        """Defined on log(x), so its penalty pulls towards exp(mean)."""
        return True

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual `(log(x) - mean) / sigma` of the penalty, see `Prior`."""
        if x <= 0:
            return -math.inf
        return (math.log(x) - self.mean) / self.sigma


@dataclass
class Laplace:
    """Laplace (double exponential) distribution.

    As a prior of a point-estimate fit, its penalty `|x - loc| / scale` is an
    L1 (lasso) penalty, pulling parameters exactly onto `loc`. It isn't
    differentiable there, so gradient-based minimizers only get close to it.

    Parameters
    ----------
    loc
        Location (mean and median)
    scale
        Scale, the standard deviation is `sqrt(2) * scale`

    """

    loc: float
    scale: float

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples from the Laplace distribution.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        return rng.laplace(self.loc, self.scale, num)

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x`."""
        return -abs(x - self.loc) / self.scale - math.log(2 * self.scale)

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return False

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual `±sqrt(2 |x - loc| / scale)`, see `Prior`."""
        d = x - self.loc
        return math.copysign(math.sqrt(2 * abs(d) / self.scale), d)


@dataclass
class SoftBounds:
    """Flat inside `[lower, upper]`, with Gaussian tails of width `scale` outside.

    As a prior of a point-estimate fit, it is a soft box constraint: no
    penalty inside the bounds and `((x - bound) / scale)² / 2` outside. Unlike
    hard bounds, it doesn't leave solutions stuck on a boundary with poor
    gradients.

    Parameters
    ----------
    lower
        Lower bound of the flat region
    upper
        Upper bound of the flat region
    scale
        Standard deviation of the tails outside the bounds, in log10 units
        (decades) if `log`
    log
        Whether the distribution is defined on log10(x), so flat in log space.
        Needs positive bounds

    """

    lower: float
    upper: float
    scale: float
    log: bool = False

    def __post_init__(self) -> None:
        """Check the parameters."""
        if not self.lower < self.upper:
            msg = f"SoftBounds needs lower < upper, got {self.lower} and {self.upper}"
            raise ValueError(msg)
        if not self.scale > 0:
            msg = f"SoftBounds needs a positive scale, got {self.scale}"
            raise ValueError(msg)
        if self.log and self.lower <= 0:
            msg = "SoftBounds(log=True) needs positive bounds"
            raise ValueError(msg)

    def _to_u(self, x: float) -> float:
        if not self.log:
            return x
        return math.log10(x) if x > 0 else -math.inf

    @property
    def _bounds_u(self) -> tuple[float, float]:
        if self.log:
            return math.log10(self.lower), math.log10(self.upper)
        return self.lower, self.upper

    @property
    def _log_norm(self) -> float:
        lo, hi = self._bounds_u
        return math.log((hi - lo) + self.scale * math.sqrt(2 * math.pi))

    def sample(self, num: int, rng: np.random.Generator | None = None) -> Array:
        """Generate random samples.

        Parameters
        ----------
        num
            Number of samples to generate
        rng
            Random number generator

        """
        if rng is None:
            rng = RNG
        lo, hi = self._bounds_u
        p_inside = (hi - lo) / math.exp(self._log_norm)
        inside = rng.random(num) < p_inside
        tail = np.abs(rng.normal(0.0, self.scale, num))
        above = rng.random(num) < 0.5  # noqa: PLR2004
        u = np.where(
            inside, rng.uniform(lo, hi, num), np.where(above, hi + tail, lo - tail)
        )
        return np.power(10.0, u) if self.log else u

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        u = self._to_u(x)
        if not math.isfinite(u):
            return -math.inf
        r = _soft_residual(u, *self._bounds_u, self.scale)
        value = -0.5 * r * r - self._log_norm
        if self.log:
            value -= math.log(x * math.log(10))
        return value

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return self.log

    @property
    def log_scale(self) -> bool:
        """Whether it is defined on log10(x), so its penalty is flat inside the bounds."""
        return self.log

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual (zero inside the bounds), see `Prior`."""
        u = self._to_u(x)
        if not math.isfinite(u):
            return -math.inf
        return _soft_residual(u, *self._bounds_u, self.scale)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        return float(stats.skewnorm.logpdf(x, self.a, loc=self.loc, scale=self.scale))

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return False

    @cached_property
    def mode(self) -> float:
        """Most likely value, found numerically."""
        return _numeric_mode(self.log_pdf, self.loc, self.scale)

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual of the penalty, see `Prior`."""
        mode = self.mode
        return _sqrt_residual(self.log_pdf(x), self.log_pdf(mode), x, mode)


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

    def log_pdf(self, x: float) -> float:
        """Log probability density at `x` (-inf outside the support)."""
        return float(self.kde.logpdf(x)[0])

    @property
    def positive_support(self) -> bool:
        """Whether the support lies within (0, inf), e.g. to sample in log space."""
        return False

    @cached_property
    def mode(self) -> float:
        """Most likely value, found numerically from the best data point."""
        points = np.asarray(self.kde.dataset[0], dtype=float)
        start = float(points[np.argmax(self.kde.logpdf(points))])
        width = float(np.sqrt(self.kde.covariance[0, 0]))
        return _numeric_mode(self.log_pdf, start, width)

    def penalty_residual(self, x: float) -> float:
        """Signed pseudo-residual of the penalty, see `Prior`."""
        mode = self.mode
        return _sqrt_residual(self.log_pdf(x), self.log_pdf(mode), x, mode)


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
