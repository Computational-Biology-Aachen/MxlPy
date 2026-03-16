"""Loss functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

__all__ = [
    "cosine_similarity",
    "mae",
    "mean",
    "mean_absolute_percentage",
    "mean_squared",
    "mean_squared_logarithmic",
    "rmse",
]

if TYPE_CHECKING:
    import pandas as pd


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
