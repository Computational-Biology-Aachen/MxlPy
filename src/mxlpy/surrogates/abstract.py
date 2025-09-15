"""Surrogate Interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from wadler_lindig import pformat

__all__ = ["AbstractSurrogate", "MockSurrogate", "SurrogateProtocol"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd

    from mxlpy.types import Derived


class SurrogateProtocol(Protocol):
    """FIXME: Something I will fill out."""

    args: list[str]
    outputs: list[str]
    stoichiometries: dict[str, dict[str, float | Derived]]

    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        """Predict outputs based on input data."""
        ...

    def calculate_inpl(
        self,
        name: str,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> None:
        """Predict outputs based on input data."""
        ...


@dataclass(kw_only=True)
class AbstractSurrogate:
    """Abstract base class for surrogate models.

    Attributes:
        inputs: List of input variable names.
        stoichiometries: Dictionary mapping reaction names to stoichiometries.

    Methods:
        predict: Abstract method to predict outputs based on input data.

    """

    args: list[str]
    outputs: list[str]
    stoichiometries: dict[str, dict[str, float | Derived]] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @abstractmethod
    def predict(
        self, args: dict[str, float | pd.Series | pd.DataFrame]
    ) -> dict[str, float]:
        """Predict outputs based on input data."""

    def calculate_inpl(
        self,
        name: str,  # noqa: ARG002, for API compatibility
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> None:
        """Predict outputs based on input data."""
        args |= self.predict(args=args)


@dataclass(kw_only=True)
class MockSurrogate(AbstractSurrogate):
    """Mock surrogate model for testing purposes."""

    fn: Callable[..., Iterable[float]]

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],
    ) -> dict[str, float]:
        """Predict outputs based on input data."""
        return dict(
            zip(
                self.outputs,
                self.fn(*(args[i] for i in self.args)),
                strict=True,
            )
        )  # type: ignore
