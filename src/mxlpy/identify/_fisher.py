# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fisher information and asymptotic parameter uncertainty."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy import fit
from mxlpy.fit._gradient import GradientResidual, _log_names
from mxlpy.fit.abstract import _Settings, _split_p0
from mxlpy.fit.losses import _sigma_array
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.runners import default_runner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy._steady_state_builder import SteadyStateModelBuilder
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.losses import Sigma
    from mxlpy.types import Array

__all__ = [
    "FisherInformation",
    "fisher_information",
]


###############################################################################
# Fisher information
###############################################################################


@dataclass(kw_only=True)
class FisherInformation:
    """Fisher information of a fit and the asymptotic parameter uncertainty.

    Built from the noise-weighted residual Jacobian `S` (the design matrix),
    with the Fisher information `F = SᵀS` and the asymptotic covariance
    `F⁻¹`. Rank deficiency is handled with a singular value decomposition:
    directions with singular values below `rtol` times the largest are
    non-identifiable, and the covariance is the pseudo-inverse.

    Attributes
    ----------
    parameters
        Parameter values the information was evaluated at
    design
        Weighted residual Jacobian `∂r/∂θ / sigma`, one row per data point
    residuals
        Weighted residuals `(prediction - data) / sigma`
    residual_variance
        Noise variance the residuals were weighted with if it was estimated
        (`Σr² / (n - p)` of the unweighted residuals), else 1
    sigma_estimated
        Whether the noise level was estimated from the residuals
    log_parameters
        Parameters whose uncertainty is expressed in log10 units
    rtol
        Relative singular value tolerance of the rank decision

    """

    parameters: pd.Series
    design: pd.DataFrame
    residuals: pd.Series
    residual_variance: float
    sigma_estimated: bool
    log_parameters: set[str] = field(default_factory=set)
    rtol: float

    @cached_property
    def _svd(self) -> tuple[Array, Array]:
        """Singular values (descending, padded to p) and right vectors (p x p)."""
        s = self.design.to_numpy()
        p = s.shape[1]
        # QR first, so the right singular vectors are complete even if there
        # are fewer data points than parameters, without an n x n matrix
        r = np.linalg.qr(s, mode="r") if s.shape[0] > p else s
        _, sv, vt = np.linalg.svd(r, full_matrices=True)
        return np.pad(sv, (0, p - len(sv))), vt

    @property
    def rank(self) -> int:
        """Numerical rank of the design matrix."""
        sv, _ = self._svd
        return int((sv > self.rtol * sv.max(initial=0.0)).sum())

    @property
    def matrix(self) -> pd.DataFrame:
        """Fisher information matrix `F = SᵀS`."""
        s = self.design.to_numpy()
        names = self.design.columns
        return pd.DataFrame(s.T @ s, index=names, columns=names)

    @property
    def chi2(self) -> float:
        """Sum of the squared weighted residuals."""
        return float(np.square(self.residuals.to_numpy()).sum())

    @property
    def dof(self) -> int:
        """Degrees of freedom, data points minus parameters."""
        return len(self.residuals) - len(self.parameters)

    def eigen(self) -> tuple[pd.Series, pd.DataFrame]:
        """Eigenvalues (descending) and eigenvectors (columns) of `F`.

        Small eigenvalues belong to sloppy parameter combinations, whose
        eigenvectors show which parameters are involved.
        """
        sv, vt = self._svd
        names = self.design.columns
        return (
            pd.Series(sv**2, name="eigenvalue"),
            pd.DataFrame(vt.T, index=names),
        )

    def non_identifiable(self, rtol: float | None = None) -> pd.DataFrame:
        """Parameter combinations the data doesn't determine (null space of `F`).

        One row per combination, columns are the parameters. Empty if all
        parameters are locally identifiable.
        """
        rtol = self.rtol if rtol is None else rtol
        sv, vt = self._svd
        null = sv <= rtol * sv.max(initial=0.0)
        return pd.DataFrame(vt[null], columns=self.design.columns)

    @property
    def covariance(self) -> pd.DataFrame:
        """Asymptotic covariance, the pseudo-inverse of `F`."""
        sv, vt = self._svd
        keep = sv > self.rtol * sv.max(initial=0.0)
        v = vt[keep].T
        cov = (v / sv[keep] ** 2) @ v.T
        names = self.design.columns
        return pd.DataFrame(cov, index=names, columns=names)

    @property
    def std_errors(self) -> pd.Series:
        """Standard errors, `inf` for parameters in a non-identifiable combination."""
        se = pd.Series(
            np.sqrt(np.clip(np.diag(self.covariance), 0, None)),
            index=self.design.columns,
            name="std_error",
        )
        involved = (np.abs(self.non_identifiable().to_numpy()) > 1e-6).any(axis=0)  # noqa: PLR2004
        se[involved] = np.inf
        return se

    @property
    def correlation(self) -> pd.DataFrame:
        """Correlation matrix, `nan` for non-identifiable parameters."""
        cov = self.covariance
        se = np.sqrt(np.diag(cov.to_numpy()))
        corr = cov / np.outer(se, se)
        inf = np.isinf(self.std_errors.to_numpy())
        corr.iloc[inf, :] = np.nan
        corr.iloc[:, inf] = np.nan
        return corr

    def wald_ci(self, alpha: float = 0.95) -> pd.DataFrame:
        """Wald confidence intervals at confidence level `alpha`.

        Uses the t distribution if the noise level was estimated, else the
        normal distribution. Intervals of log parameters are symmetric in
        log10 space. Returns value, std_error, lower and upper per parameter.
        """
        tail = 1 - (1 - alpha) / 2
        q = (
            stats.t.ppf(tail, self.dof)
            if self.sigma_estimated
            else stats.norm.ppf(tail)
        )
        se = self.std_errors
        out = pd.DataFrame(
            {"value": self.parameters, "std_error": se},
            index=self.design.columns,
        )
        for name in out.index:
            value, err = self.parameters[name], se[name]
            if name in self.log_parameters:
                centre = np.log10(value)
                lo, hi = 10 ** (centre - q * err), 10 ** (centre + q * err)
            else:
                lo, hi = value - q * err, value + q * err
            out.loc[name, ["lower", "upper"]] = [lo, hi]
        return out


def residual_fn(data: pd.Series | pd.DataFrame, protocol: pd.DataFrame | None) -> Any:
    """Residual function of the data: steady state, time course or protocol."""
    if protocol is not None:
        return protocol_time_course_residual
    if isinstance(data, pd.Series):
        return steady_state_residual
    return time_course_residual


def _sigma_vector(
    sigma: Sigma,
    data: pd.Series | pd.DataFrame,
    mask: NDArray[np.bool_],
) -> Array:
    return _sigma_array(sigma, data).ravel()[mask]


def fisher_information(
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
    data: pd.Series | pd.DataFrame,
    parameters: Mapping[str, float] | fit.Fit,
    *,
    sigma: Sigma | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
    runner: Any | None = None,
    transform: Transform | None = None,
    rtol: float = 1e-6,
) -> FisherInformation:
    """Fisher information and asymptotic uncertainty of fitted parameters.

    Costs one sensitivity evaluation, so it is a cheap first look at
    parameter uncertainty, correlations and identifiability before e.g.
    profile likelihoods. The asymptotic covariance assumes independent
    Gaussian noise and a model that is close to linear around the optimum.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        A Series fits a steady state, a DataFrame (indexed by time) a time
        course, or a protocol time course if `protocol` is given
    parameters
        Values of the fitted parameters (and initial values), e.g. a fit result
    sigma
        Standard deviation of the measurement noise: a scalar, one value per
        observable (Series) or per data point (DataFrame like `data`).
        If None, the noise variance is estimated from the residuals as
        `Σr² / (n - p)` and Wald intervals use the t distribution
    y0
        Initial conditions the fit used, if any. Variables in `parameters`
        are fitted initial values, as in the fit functions
    protocol
        Protocol of a protocol time course
    runner
        How the model is simulated and differentiated, see `mxlpy.runners`
    transform
        `log` (or per parameter) to express uncertainties of these
        parameters in log10 units, i.e. relative uncertainties
    rtol
        Singular values of the design matrix below `rtol` times the largest
        count as zero. Sensitivities are only accurate to roughly the
        integration tolerance (or the finite-difference error), so much
        smaller values can't be told apart from zero

    Examples
    --------
        >>> res = fit.time_course(model, p0=p0, data=data).unwrap_or_err()
        >>> fim = identify.fisher_information(model, data, res, sigma=0.1)
        >>> fim.wald_ci()
        >>> fim.non_identifiable()

    """
    if isinstance(parameters, fit.Fit):
        parameters = parameters.best_pars
    values = {k: float(v) for k, v in parameters.items()}
    model = deepcopy(model)
    p_names, v_names = _split_p0(model, values)
    if unknown := sorted(set(values) - set(p_names) - set(v_names)):
        msg = f"Unknown parameters or variables: {unknown}"
        raise ValueError(msg)
    settings = _Settings(
        model=model,
        data=data,
        y0=y0,
        runner=default_runner(model) if runner is None else runner,
        loss_fn=fit.rmse,
        p_names=p_names,
        v_names=v_names,
        standard_scale=False,
        protocol=protocol,
    )
    residual = GradientResidual(residual_fn(data, protocol), settings)
    r = residual.residuals(values)
    if not np.isfinite(r).all():
        msg = "Simulation failed at the given parameters"
        raise ValueError(msg)
    jac = residual.jacobian(values)

    log = _log_names(values, transform)
    jac = jac * np.array([values[k] * np.log(10) if k in log else 1.0 for k in values])

    mask = residual.mask
    if isinstance(data, pd.DataFrame):
        rows = pd.MultiIndex.from_product(
            [data.index, data.columns], names=["time", "observable"]
        )[mask]
    else:
        rows = pd.Index(data.index, name="observable")[mask]

    if sigma is None:
        n, p = jac.shape
        if n <= p:
            msg = "Estimating sigma needs more data points than parameters, pass sigma"
            raise ValueError(msg)
        residual_variance = float(np.square(r).sum() / (n - p))
        if residual_variance == 0:
            msg = "The residuals are zero, so the noise can't be estimated, pass sigma"
            raise ValueError(msg)
        # Weighting with the estimated noise makes F⁻¹ the covariance as well
        weights = np.full(n, 1 / np.sqrt(residual_variance))
    else:
        weights = 1 / _sigma_vector(sigma, data, mask)
        residual_variance = 1.0

    return FisherInformation(
        parameters=pd.Series(values, name="value"),
        design=pd.DataFrame(jac * weights[:, None], index=rows, columns=list(values)),
        residuals=pd.Series(r * weights, index=rows, name="residual"),
        residual_variance=residual_variance,
        sigma_estimated=sigma is None,
        log_parameters=log,
        rtol=rtol,
    )
