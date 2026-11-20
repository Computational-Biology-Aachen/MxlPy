# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Loss functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mxlpy.types import Array

__all__ = [
    "GaussianNll",
    "Sigma",
    "cosine_similarity",
    "mae",
    "mean",
    "mean_absolute_percentage",
    "mean_squared",
    "mean_squared_logarithmic",
    "rmse",
]

# Standard deviation of the measurement noise: one value, one per observable
# (Series) or one per data point (DataFrame shaped like the data)
type Sigma = float | pd.Series | pd.DataFrame


def _sigma_array(sigma: Sigma, data: pd.Series | pd.DataFrame) -> Array:
    """`sigma` broadcast to the shape of `data`."""
    if isinstance(sigma, pd.DataFrame):
        if isinstance(data, pd.Series):
            msg = "sigma per data point (a DataFrame) needs time course data"
            raise ValueError(msg)  # noqa: TRY004, invalid input like the others
        values = sigma.reindex(index=data.index, columns=data.columns).to_numpy()
    elif isinstance(sigma, pd.Series):
        keys = data.index if isinstance(data, pd.Series) else data.columns
        values = np.broadcast_to(sigma.reindex(keys).to_numpy(), data.shape)
    else:
        values = np.full(data.shape, float(sigma))
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(data.to_numpy(dtype=float))
    if not (np.isfinite(values[observed]).all() and (values[observed] > 0).all()):
        msg = "sigma has to be positive and finite for every observed data point"
        raise ValueError(msg)
    return values


@dataclass(frozen=True)
class GaussianNll:
    """Negative log-likelihood of independent Gaussian noise with known `sigma`.

    `Σ ((prediction - data) / sigma)² / 2`, i.e. half the chi-square, the
    negative log-likelihood up to a constant. Unlike the heuristic losses,
    this makes a fit a maximum-likelihood estimate, and with a `prior` a
    maximum-a-posteriori one, and differences in it are comparable between
    models and parameter values (likelihood ratios).

    Data and prediction are always compared unscaled (`standard_scale` is
    ignored), as `sigma` is in data units. Missing data is skipped.
    Least-squares minimizers minimize it exactly, using the residuals
    weighted by `1 / sigma`.

    Parameters
    ----------
    sigma
        Standard deviation of the measurement noise: a scalar, one value per
        observable (Series) or per data point (DataFrame like the data)

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data,
        ...                 loss_fn=fit.GaussianNll(sigma=0.05),
        ...                 minimizer=fit.LeastSquaresMinimizer())

    """

    sigma: Sigma = 1.0

    def __post_init__(self) -> None:
        """Check that sigma is positive."""
        values = (
            np.asarray(self.sigma, dtype=float)
            if not isinstance(self.sigma, (pd.Series, pd.DataFrame))
            else self.sigma.to_numpy(dtype=float)
        )
        given = ~np.isnan(values)
        if not (np.isfinite(values[given]).all() and (values[given] > 0).all()):
            msg = "sigma has to be positive and finite"
            raise ValueError(msg)

    def __call__(
        self,
        y_pred: pd.DataFrame | pd.Series,
        y_true: pd.DataFrame | pd.Series,
    ) -> float:
        """Half the chi-square of the prediction.

        Parameters
        ----------
        y_pred
            Predicted values.
        y_true
            True (observed) values.

        Returns
        -------
        float
            Negative log-likelihood up to a constant.

        """
        diff = y_pred - y_true
        z = np.asarray(diff / self.sigma, dtype=float)
        compared = np.isfinite(np.asarray(diff, dtype=float))
        if not np.isfinite(z[compared]).all():
            msg = "sigma has to be given for every observed data point"
            raise ValueError(msg)
        return 0.5 * float(np.sum(np.square(z[compared])))


def mean(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate mean error between model and data.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Mean error.

    """
    return cast(float, np.mean(y_pred - y_true))


def mean_squared(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate mean squared error between model and data.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Mean squared error.

    """
    return cast(float, np.mean(np.square(y_pred - y_true)))


def rmse(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate root mean square error between model and data.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Root mean square error.

    """
    return cast(float, np.sqrt(np.mean(np.square(y_pred - y_true))))


def mae(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate mean absolute error.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Mean absolute error.

    """
    return cast(float, np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate mean absolute percentage error.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Mean absolute percentage error.

    """
    return cast(float, 100 * np.mean(np.abs((y_true - y_pred) / y_pred)))


def mean_squared_logarithmic(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate mean squared logarithmic error between model and data.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Mean squared logarithmic error.

    """
    return cast(float, np.mean(np.square(np.log(y_pred + 1) - np.log(y_true + 1))))


def cosine_similarity(
    y_pred: pd.DataFrame | pd.Series,
    y_true: pd.DataFrame | pd.Series,
) -> float:
    """Calculate negative cosine similarity between model and data.

    Parameters
    ----------
    y_pred
        Predicted values.
    y_true
        True (observed) values.

    Returns
    -------
    float
        Negative cosine similarity.

    """
    norm = np.linalg.norm
    return cast(float, -np.sum(norm(y_pred, 2) * norm(y_true, 2)))
