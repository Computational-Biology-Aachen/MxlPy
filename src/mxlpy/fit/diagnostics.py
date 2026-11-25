# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fit diagnostics: measurement noise, goodness of fit and residual whiteness.

- `estimate_sigma`: noise per observable from replicates or neighbouring points
- `chi2_test`: χ² goodness of fit of a Gaussian negative log-likelihood
- `check_residuals`: runs and Ljung-Box tests per data set and observable
  (structured residuals point to missing dynamics)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from mxlpy.types import Array

__all__ = [
    "GoodnessOfFit",
    "ResidualCheck",
    "chi2_test",
    "check_residuals",
    "estimate_sigma",
    "ljung_box",
    "runs_test",
]


###############################################################################
# Noise
###############################################################################


def _replicate_sigma(column: pd.Series) -> float | None:
    """Pooled standard deviation of replicates (repeated time points), if any."""
    groups = column.dropna().groupby(level=0)
    counts = cast(pd.Series, groups.count())
    if not (counts > 1).any():
        return None
    variances = cast(pd.Series, groups.var(ddof=1)).fillna(0.0)
    ss = float((variances * (counts - 1)).sum())
    dof = int((counts - 1).clip(lower=0).sum())
    return math.sqrt(ss / dof) if dof > 0 and ss > 0 else None


def _pseudo_residuals(t: Array, y: Array, offsets: tuple[int, ...]) -> Array:
    """Normalised differences of each point to the polynomial through its neighbours.

    The Lagrange polynomial through the points at `offsets` (relative
    positions, e.g. -2, -1, 1, 2) is evaluated at the centre point. Smooth
    trends up to the polynomial's degree cancel; for independent noise of
    standard deviation sigma each pseudo-residual has variance sigma².
    """
    lo, hi = -min(offsets), max(offsets)
    centre = np.arange(lo, len(y) - hi)
    weights = []
    for j in offsets:
        w = np.ones(len(centre))
        for m in offsets:
            if m != j:
                w *= (t[centre] - t[centre + m]) / (t[centre + j] - t[centre + m])
        weights.append(w)
    fitted = sum(w * y[centre + j] for w, j in zip(weights, offsets, strict=True))
    norm = np.sqrt(1 + sum(w**2 for w in weights))
    return (y[centre] - fitted) / norm


def _difference_sigma(column: pd.Series) -> float | None:
    """Noise from pseudo-residuals of neighbouring points (no smoother needed).

    Each point is compared with the cubic through its two neighbours on
    either side (Gasser, Sroka & Jennen-Steinmetz use the line through one
    on either side), which removes smooth trends up to third order, so curved
    time courses sampled sparsely don't inflate the estimate. Falls back to
    the line for fewer than five time points.
    """
    s = cast(pd.Series, column.dropna().groupby(level=0).mean()).sort_index()
    t = np.asarray(s.index, dtype=float)
    y = s.to_numpy(dtype=float)
    if len(y) < 3:  # noqa: PLR2004
        return None
    offsets = (-2, -1, 1, 2) if len(y) >= 5 else (-1, 1)  # noqa: PLR2004
    pseudo = _pseudo_residuals(t, y, offsets)
    value = math.sqrt(float(np.mean(np.square(pseudo))))
    return value if value > 0 else None


def estimate_sigma(data: pd.DataFrame, *, floor: float = 0.01) -> pd.Series:
    """Measurement noise per observable of a time course.

    From replicates (repeated time points) if there are any, else from
    pseudo-residuals of neighbouring points. Observables without either (fewer than three time
    points, or exactly smooth data) get `floor` times their RMS, as a likelihood
    needs a positive sigma.

    Returns
    -------
    pd.Series
        Sigma by observable

    """
    out = {}
    for name in data.columns:
        column = cast(pd.Series, data[name])
        sigma = _replicate_sigma(column)
        if sigma is None:
            sigma = _difference_sigma(column)
        if sigma is None:
            values = column.dropna().to_numpy(dtype=float)
            rms = math.sqrt(float(np.mean(np.square(values)))) if values.size else 0.0
            sigma = floor * rms if rms > 0 else 1.0
        out[name] = float(sigma)
    return pd.Series(out, dtype=float)


###############################################################################
# Goodness of fit
###############################################################################


@dataclass(frozen=True)
class GoodnessOfFit:
    """χ² goodness-of-fit test of a maximum-likelihood fit.

    Attributes
    ----------
    chi2
        Sum of squared noise-weighted residuals, twice the negative
        log-likelihood
    dof
        Degrees of freedom, data points minus estimated parameters
    p_value
        Probability of a χ² at least this large if the model is right

    """

    chi2: float
    dof: int
    p_value: float

    def passed(self, significance: float) -> bool:
        """Whether the fit isn't rejected at `significance`."""
        return self.p_value >= significance


def chi2_test(nll: float, n_data: int, n_parameters: int) -> GoodnessOfFit:
    """χ² test of a Gaussian negative log-likelihood `nll` (half the χ²)."""
    chi2 = 2 * nll
    dof = max(n_data - n_parameters, 1)
    p_value = float(stats.chi2.sf(chi2, dof)) if math.isfinite(chi2) else 0.0
    return GoodnessOfFit(chi2=float(chi2), dof=dof, p_value=p_value)


###############################################################################
# Residual whiteness
###############################################################################


def runs_test(residuals: Array) -> float:
    """Two-sided p-value of the Wald-Wolfowitz runs test of residual signs.

    Too few runs (long stretches of one sign) mean structured residuals.
    Exact zeros are dropped.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r) & (r != 0)]
    n_pos = int((r > 0).sum())
    n_neg = int((r < 0).sum())
    n = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        # Every residual has the same sign: maximally structured if long enough
        return 1.0 if n < 6 else 2 * 0.5**n  # noqa: PLR2004
    runs = 1 + int(np.sum(np.sign(r[1:]) != np.sign(r[:-1])))
    mean = 2 * n_pos * n_neg / n + 1
    var = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
    if var <= 0:
        return 1.0
    z = (runs - mean) / math.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def ljung_box(residuals: Array, lags: int | None = None) -> float:
    """p-value of the Ljung-Box test for autocorrelation up to `lags`.

    Defaults to min(5, n / 4) lags.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if lags is None:
        lags = min(5, n // 4)
    if lags < 1 or n < 4:  # noqa: PLR2004
        return 1.0
    r = r - r.mean()
    denom = float(np.sum(r**2))
    if denom == 0:
        return 1.0
    q = 0.0
    for k in range(1, lags + 1):
        rho = float(np.sum(r[k:] * r[:-k])) / denom
        q += rho**2 / (n - k)
    q *= n * (n + 2)
    return float(stats.chi2.sf(q, lags))


@dataclass(frozen=True)
class ResidualCheck:
    """Whiteness of the residuals of every observable in every data set.

    Attributes
    ----------
    table
        Per data set and observable: number of residuals, lag-1
        autocorrelation and the p-values of the runs and Ljung-Box tests
    significance
        Family-wise significance level, Bonferroni-corrected over all tests

    """

    table: pd.DataFrame
    significance: float

    @property
    def threshold(self) -> float:
        """Per-test significance level."""
        return self.significance / max(2 * len(self.table), 1)

    @property
    def flagged(self) -> pd.DataFrame:
        """Rows (data set, observable) with structured residuals."""
        t = self.table
        return t.loc[
            (t["runs_p"] < self.threshold) | (t["ljung_box_p"] < self.threshold)
        ]

    @property
    def passed(self) -> bool:
        """Whether all residuals look white."""
        return self.flagged.empty


def check_residuals(
    residuals: dict[str, pd.DataFrame], significance: float = 0.05
) -> ResidualCheck:
    """Runs and Ljung-Box tests of residual time courses by data set."""
    rows = {}
    for name, df in residuals.items():
        df = df.sort_index()  # noqa: PLW2901
        for col in df.columns:
            r = df[col].to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            lag1 = (
                float(np.corrcoef(r[1:], r[:-1])[0, 1])
                if len(r) > 2 and np.std(r) > 0  # noqa: PLR2004
                else math.nan
            )
            rows[(name, col)] = {
                "n": len(r),
                "lag1": lag1,
                "runs_p": runs_test(r),
                "ljung_box_p": ljung_box(r),
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["data", "observable"])
    return ResidualCheck(table=table, significance=significance)
