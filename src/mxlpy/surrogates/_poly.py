from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy import polynomial

from mxlpy.types import AbstractSurrogate, ArrayLike, Derived

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "PolynomialExpansion",
    "Surrogate",
    "train",
]

# define custom type
PolynomialExpansion = (
    polynomial.polynomial.Polynomial
    | polynomial.chebyshev.Chebyshev
    | polynomial.legendre.Legendre
    | polynomial.laguerre.Laguerre
    | polynomial.hermite.Hermite
    | polynomial.hermite_e.HermiteE
)


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    model: PolynomialExpansion

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        return self.model(y)

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        return dict(
            zip(
                self.outputs,
                self.model(np.array([args[arg] for arg in self.args])),
                strict=True,
            )
        )


def train(
    feature: ArrayLike | pd.Series,
    target: ArrayLike | pd.Series,
    series: Literal[
        "Power", "Chebyshev", "Legendre", "Laguerre", "Hermite", "HermiteE"
    ] = "Power",
    degrees: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
    surrogate_args: list[str] | None = None,
    surrogate_outputs: list[str] | None = None,
    surrogate_stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
) -> tuple[Surrogate, pd.DataFrame]:
    """Train a surrogate model based on function series expansion.

    Args:
        feature: Input data as a numpy array.
        target: Output data as a numpy array.
        series: Base functions for the surrogate model
        degrees: Degrees of the polynomial to fit to the data.
        surrogate_args: Additional arguments for the surrogate model.
        surrogate_outputs: Names of the surrogate model outputs.
        surrogate_stoichiometries: Mapping of variables to their stoichiometries

    Returns:
        PolySurrogate: Polynomial surrogate model.

    """
    feature = np.array(feature, dtype=float)
    target = np.array(target, dtype=float)

    # Choose numpy polynomial convenience classes
    series_dictionary = {
        "Power": polynomial.polynomial.Polynomial,
        "Chebyshev": polynomial.chebyshev.Chebyshev,
        "Legendre": polynomial.legendre.Legendre,
        "Laguerre": polynomial.laguerre.Laguerre,
        "Hermite": polynomial.hermite.Hermite,
        "HermiteE": polynomial.hermite_e.HermiteE,
    }

    fn_series = series_dictionary[series]

    models = [fn_series.fit(feature, target, degree) for degree in degrees]
    predictions = np.array([model(feature) for model in models], dtype=float)
    errors = np.sqrt(np.mean(np.square(predictions - target.reshape(1, -1)), axis=1))
    log_likelihood = -0.5 * np.sum(
        np.square(predictions - target.reshape(1, -1)), axis=1
    )
    score = 2 * np.array(degrees) - 2 * log_likelihood

    # Choose the model with the lowest AIC
    model = models[np.argmin(score)]
    return (
        Surrogate(
            model=model,
            args=surrogate_args if surrogate_args is not None else [],
            outputs=surrogate_outputs if surrogate_outputs is not None else [],
            stoichiometries=surrogate_stoichiometries
            if surrogate_stoichiometries is not None
            else {},
        ),
        pd.DataFrame(
            {"models": models, "error": errors, "score": score},
            index=pd.Index(np.array(degrees), name="degree"),
        ),
    )
