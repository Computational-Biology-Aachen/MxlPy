# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy import polynomial

from mxlpy.surrogates._io import read_meta, surrogate_fields, write_meta
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mxlpy.types import ArrayLike, Derived

__all__ = [
    "PolynomialExpansion",
    "Surrogate",
    "load",
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

    def save(self, path: str | Path) -> None:
        """Save into the directory ``path``: ``surrogate.json`` plus ``model.json``.

        ``model.json`` holds the series kind, coefficients, domain, window
        and symbol. Load again with :func:`load` or
        :func:`mxlpy.surrogates.load`.
        """
        path = write_meta(path, self, backend="poly")
        state = {
            "kind": type(self.model).__name__,
            "coef": self.model.coef.tolist(),
            "domain": self.model.domain.tolist(),
            "window": self.model.window.tolist(),
            "symbol": self.model.symbol,
        }
        (path / "model.json").write_text(json.dumps(state, indent=2))

    def predict_raw(self, y: np.ndarray) -> np.ndarray:
        """Predict raw output from the polynomial model.

        Parameters
        ----------
        y
            Input data as a numpy array.

        Returns
        -------
        np.ndarray
            Raw model prediction.

        """
        return self.model(y)

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        """Predict outputs based on input data.

        Parameters
        ----------
        args
            Mapping of input names to their values.

        Returns
        -------
        dict[str, float]
            Mapping of output names to predicted values.

        """
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

    Parameters
    ----------
    feature
        Input data as a numpy array.
    target
        Output data as a numpy array.
    series
        Base functions for the surrogate model
    degrees
        Degrees of the polynomial to fit to the data.
    surrogate_args
        Additional arguments for the surrogate model.
    surrogate_outputs
        Names of the surrogate model outputs.
    surrogate_stoichiometries
        Mapping of variables to their stoichiometries

    Returns
    -------
    PolySurrogate
        Polynomial surrogate model.

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


_SERIES: dict[str, type[PolynomialExpansion]] = {
    cls.__name__: cls
    for cls in (
        polynomial.Polynomial,
        polynomial.Chebyshev,
        polynomial.Legendre,
        polynomial.Laguerre,
        polynomial.Hermite,
        polynomial.HermiteE,
    )
}


def load(path: str | Path) -> Surrogate:
    """Load a polynomial surrogate saved with ``surrogate.save(path)``.

    Parameters
    ----------
    path
        Directory written by ``save``.

    Returns
    -------
    Surrogate
        The polynomial surrogate.

    Raises
    ------
    SerializationError
        If ``model.json`` names an unknown series kind.

    """
    meta = read_meta(path, backend="poly")
    state = json.loads((Path(path) / "model.json").read_text())
    if (cls := _SERIES.get(state["kind"])) is None:
        msg = f"Unknown polynomial series {state['kind']!r}."
        raise SerializationError(msg)
    model = cls(
        np.array(state["coef"], dtype=float),
        domain=state["domain"],
        window=state["window"],
        symbol=state["symbol"],
    )
    _, fields = surrogate_fields(meta)
    return Surrogate(model=model, **fields)
