# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy import polynomial

from mxlpy.surrogates._poly import (
    Surrogate,
    train,
)


@pytest.fixture
def linear_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3 + np.random.normal(0, 0.5, size=100)  # noqa: NPY002
    return x, y


@pytest.fixture
def quadratic_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-5, 5, 100)
    y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 0.5, size=100)  # noqa: NPY002
    return x, y


def test_poly_surrogate_predict_raw() -> None:
    # Create a simple polynomial: y = 2x + 3
    poly = polynomial.polynomial.Polynomial([3, 2])
    surrogate = Surrogate(
        model=poly,
        args=["x"],
        outputs=["y"],
        stoichiometries={},
    )

    # Test with a single input
    result_single = surrogate.predict_raw(np.array([2]))
    assert np.isclose(result_single, 7)  # 2*2 + 3 = 7

    # Test with multiple inputs
    inputs = np.array([0, 1, 2, 3])
    results = surrogate.predict_raw(inputs)
    expected = np.array([3, 5, 7, 9])  # [3 + 2*0, 3 + 2*1, 3 + 2*2, 3 + 2*3]
    assert np.allclose(results, expected)


def test_train_polynomial_surrogate_power_series(
    linear_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = linear_data

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Power",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.polynomial.Polynomial)

    # Check stats dataframe
    assert isinstance(stats, pd.DataFrame)
    assert list(stats.index) == [1, 2, 3]  # degrees
    assert list(stats.columns) == ["models", "error", "score"]

    # Test predictions
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape

    # Since data is linear with noise, linear model should be close
    assert np.corrcoef(predictions, y)[0, 1] > 0.9


def test_train_polynomial_surrogate_chebyshev_series(
    quadratic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = quadratic_data

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Chebyshev",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.chebyshev.Chebyshev)

    # Check stats dataframe
    assert isinstance(stats, pd.DataFrame)
    assert list(stats.index) == [1, 2, 3]  # degrees
    assert list(stats.columns) == ["models", "error", "score"]

    # Test predictions with the best model
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape

    # Since data is quadratic with noise, model should be close
    assert np.corrcoef(predictions, y)[0, 1] > 0.9


def test_train_polynomial_surrogate_legendre_series(
    quadratic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = quadratic_data

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Legendre",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.legendre.Legendre)

    # Check stats dataframe
    assert isinstance(stats, pd.DataFrame)
    assert list(stats.columns) == ["models", "error", "score"]

    # Test predictions with the best model
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape


def test_train_polynomial_surrogate_laguerre_series(
    quadratic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = quadratic_data
    x = np.abs(x)  # Laguerre polynomials best for x >= 0

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Laguerre",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.laguerre.Laguerre)

    # Test predictions with the best model
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape


def test_train_polynomial_surrogate_hermite_series(
    quadratic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = quadratic_data

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Hermite",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.hermite.Hermite)

    # Test predictions with the best model
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape


def test_train_polynomial_surrogate_hermitee_series(
    quadratic_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = quadratic_data

    surrogate, stats = train(
        feature=x,
        target=y,
        series="HermiteE",
        degrees=[1, 2, 3],
    )

    assert isinstance(surrogate, Surrogate)
    assert isinstance(surrogate.model, polynomial.hermite_e.HermiteE)

    # Test predictions with the best model
    predictions = surrogate.predict_raw(x)
    assert predictions.shape == y.shape


def test_train_polynomial_surrogate_with_args_and_stoichiometries(
    linear_data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = linear_data
    surrogate_args = ["x"]
    surrogate_stoichiometries = {"r1": {"x": -1.0, "y": 1.0}}

    surrogate, stats = train(
        feature=x,
        target=y,
        series="Power",
        degrees=[1, 2],
        surrogate_args=surrogate_args,
        surrogate_stoichiometries=surrogate_stoichiometries,  # type: ignore
    )

    assert surrogate.args == surrogate_args
    assert surrogate.stoichiometries == surrogate_stoichiometries
