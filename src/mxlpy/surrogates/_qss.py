from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mxlpy.types import AbstractSurrogate, Array

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
        return dict(
            zip(
                self.outputs,
                self.model(*(args[arg] for arg in self.args)),
                strict=True,
            )
        )
