# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Information criteria, model weights and likelihood-ratio tests.

Shared by the structure search in `mxlpy.select` and any other code that
compares fitted models, e.g. selecting condition-specific parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "Criterion",
    "FitCriteria",
    "InformationCriterion",
    "LikelihoodRatio",
    "aic",
    "aicc",
    "bic",
    "information_criterion",
    "lrt",
    "model_weights",
]

type InformationCriterion = Literal["aic", "aicc", "bic"]
"""An information criterion, lower is better."""

type Criterion = Literal["aic", "aicc", "bic", "lrt"]
"""How models are compared: an information criterion or likelihood-ratio tests."""


def _check(n_parameters: int, n_data: int | None = None) -> None:
    if n_parameters < 0:
        msg = f"n_parameters can't be negative, got {n_parameters}"
        raise ValueError(msg)
    if n_data is not None and n_data < 1:
        msg = f"n_data has to be positive, got {n_data}"
        raise ValueError(msg)


def aic(log_likelihood: float, n_parameters: int) -> float:
    """Akaike information criterion `2k - 2 LL`.

    Parameters
    ----------
    log_likelihood
        Maximised log-likelihood. Constants that are the same for all
        compared models (e.g. the Gaussian normalisation) cancel in
        differences, so the negative of a `fit.GaussianNll` loss works
    n_parameters
        Number of estimated parameters `k`

    """
    _check(n_parameters)
    return 2 * n_parameters - 2 * log_likelihood


def aicc(log_likelihood: float, n_parameters: int, n_data: int) -> float:
    """Small-sample corrected AIC, `AIC + 2k(k + 1) / (n - k - 1)`.

    Infinite if there are fewer than `k + 2` data points.

    Parameters
    ----------
    log_likelihood
        Maximised log-likelihood, see `aic`
    n_parameters
        Number of estimated parameters `k`
    n_data
        Number of data points `n`

    """
    _check(n_parameters, n_data)
    k = n_parameters
    if n_data - k - 1 <= 0:
        return math.inf
    return aic(log_likelihood, k) + 2 * k * (k + 1) / (n_data - k - 1)


def bic(log_likelihood: float, n_parameters: int, n_data: int) -> float:
    """Bayesian information criterion `k ln(n) - 2 LL`.

    Parameters
    ----------
    log_likelihood
        Maximised log-likelihood, see `aic`
    n_parameters
        Number of estimated parameters `k`
    n_data
        Number of data points `n`

    """
    _check(n_parameters, n_data)
    return n_parameters * math.log(n_data) - 2 * log_likelihood


def information_criterion(
    name: InformationCriterion,
    log_likelihood: float,
    n_parameters: int,
    n_data: int,
) -> float:
    """Information criterion by name, see `aic`, `aicc` and `bic`."""
    match name:
        case "aic":
            return aic(log_likelihood, n_parameters)
        case "aicc":
            return aicc(log_likelihood, n_parameters, n_data)
        case "bic":
            return bic(log_likelihood, n_parameters, n_data)
    msg = f"Unknown information criterion {name!r}, use 'aic', 'aicc' or 'bic'"
    raise ValueError(msg)


def model_weights(values: pd.Series) -> pd.Series:
    """Akaike (or BIC) weights `exp(-Δ/2) / Σ exp(-Δ/2)` of criterion values.

    The relative likelihood of each model given the data, among the models
    in `values`. Models with infinite or NaN values (e.g. failed fits) get
    weight 0.
    """
    x = values.to_numpy(dtype=float)
    finite = np.isfinite(x)
    w = np.zeros_like(x)
    if finite.any():
        w[finite] = np.exp(-(x[finite] - x[finite].min()) / 2)
        w /= w.sum()
    return pd.Series(w, index=values.index, name="weight")


@dataclass(frozen=True)
class LikelihoodRatio:
    """Result of a likelihood-ratio test of nested models, see `lrt`.

    Attributes
    ----------
    statistic
        `2 (LL_alt - LL_null)`, clipped at 0
    df
        Degrees of freedom, the number of extra parameters of the alternative
    p_value
        Probability of a statistic at least this large if the null model is
        true, from the asymptotic chi-square distribution

    """

    statistic: float
    df: int
    p_value: float

    def rejects(self, significance: float = 0.05) -> bool:
        """Whether the null model is rejected at the significance level."""
        return self.p_value < significance


def lrt(
    log_likelihood_null: float, log_likelihood_alt: float, df: int
) -> LikelihoodRatio:
    """Likelihood-ratio test of a null model nested in an alternative one.

    The alternative has `df` more estimated parameters and contains the null
    model as a special case (e.g. a parameter fixed to 0). A negative
    statistic, where the larger model fits worse, means its fit missed the
    optimum; it is clipped to 0.

    Parameters
    ----------
    log_likelihood_null
        Maximised log-likelihood of the smaller (null) model
    log_likelihood_alt
        Maximised log-likelihood of the larger (alternative) model
    df
        Difference of the numbers of estimated parameters, positive

    """
    if df < 1:
        msg = f"A likelihood-ratio test needs df >= 1 extra parameters, got {df}"
        raise ValueError(msg)
    statistic = max(2 * (log_likelihood_alt - log_likelihood_null), 0.0)
    return LikelihoodRatio(
        statistic=statistic,
        df=df,
        p_value=float(stats.chi2.sf(statistic, df)),
    )


class FitCriteria:
    """Criteria of a maximum-likelihood fit, shared by the results of `mxlpy.select`.

    Needs the attributes `best_pars` (estimated values), `loss` (negative
    log-likelihood, up to a constant), `penalty` (priors and penalties, not
    part of the criteria) and `n_data` (number of data points).
    """

    best_pars: dict[str, float]
    loss: float
    penalty: float
    n_data: int

    @property
    def n_parameters(self) -> int:
        """Number of estimated parameters `k`."""
        return len(self.best_pars)

    @property
    def log_likelihood(self) -> float:
        """Maximised log-likelihood, `-loss`."""
        return -self.loss

    @property
    def objective(self) -> float:
        """The minimised objective, `loss + penalty`."""
        return self.loss + self.penalty

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return aic(self.log_likelihood, self.n_parameters)

    @property
    def aicc(self) -> float:
        """Small-sample corrected AIC."""
        return aicc(self.log_likelihood, self.n_parameters, self.n_data)

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return bic(self.log_likelihood, self.n_parameters, self.n_data)

    def criterion(self, name: InformationCriterion) -> float:
        """Information criterion by name."""
        return information_criterion(
            name, self.log_likelihood, self.n_parameters, self.n_data
        )
