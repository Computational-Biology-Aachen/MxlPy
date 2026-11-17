# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dill

from mxlpy.surrogates._io import read_meta, surrogate_fields, write_meta
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import Array

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "QSSFn",
    "Surrogate",
    "load",
]

type QSSFn = Callable[..., Iterable[float] | Array]


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    model: QSSFn

    def save(self, path: str | Path) -> None:
        """Save into the directory ``path``: ``surrogate.json`` plus ``model.dill``.

        The function is pickled with dill, see :func:`load` for the
        implications. Load again with :func:`load` or
        :func:`mxlpy.surrogates.load`.
        """
        path = write_meta(path, self, backend="qss")
        with (path / "model.dill").open("wb") as fp:
            dill.dump(self.model, fp)

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],
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
                self.model(*(args[arg] for arg in self.args)),
                strict=True,
            )
        )


def load(path: str | Path) -> Surrogate:
    """Load a qss surrogate saved with ``surrogate.save(path)``.

    Like any ``pickle``-based loader, this executes arbitrary code embedded
    in ``model.dill``; only call it on files your own code (or another
    trusted party) wrote, never on untrusted input.

    Parameters
    ----------
    path
        Directory written by ``save``.

    Returns
    -------
    Surrogate
        The qss surrogate.

    """
    meta = read_meta(path, backend="qss")
    with (Path(path) / "model.dill").open("rb") as fp:
        model = dill.load(fp)  # nosec -- trusted, caller-controlled file
    _, fields = surrogate_fields(meta)
    return Surrogate(model=model, **fields)
