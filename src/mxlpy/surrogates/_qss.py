from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import Array

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "QSSFn",
    "Surrogate",
]

type QSSFn = Callable[..., Iterable[float] | Array]


@dataclass(kw_only=True)
class Surrogate(AbstractSurrogate):
    model: QSSFn

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
