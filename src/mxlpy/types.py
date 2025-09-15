"""Types Module.

This module provides type definitions and utility types for use throughout the project.
It includes type aliases for arrays, numbers, and callable functions, as well as re-exports
of common types from standard libraries.

Classes:
    DerivedFn: Callable type for derived functions.
    Array: Type alias for numpy arrays of float64.
    Number: Type alias for float, list of floats, or numpy arrays.
    Param: Type alias for parameter specifications.
    RetType: Type alias for return types.
    Axes: Type alias for numpy arrays of matplotlib axes.
    ArrayLike: Type alias for numpy arrays or lists of floats.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from wadler_lindig import pformat

__all__ = [
    "Array",
    "ArrayLike",
    "Derived",
    "InitialAssignment",
    "Param",
    "Parameter",
    "RateFn",
    "Reaction",
    "Readout",
    "RetType",
    "Rhs",
    "Variable",
    "unwrap",
    "unwrap2",
]

type RateFn = Callable[..., float]
type Array = NDArray[np.floating[Any]]
type ArrayLike = NDArray[np.floating[Any]] | pd.Index | list[float]
type Rhs = Callable[
    [
        float,  # t
        Iterable[float],  # y
    ],
    tuple[float, ...],
]

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


if TYPE_CHECKING:
    import sympy

    from mxlpy.model import Model


def unwrap[T](el: T | None) -> T:
    """Unwraps an optional value, raising an error if the value is None.

    Args:
        el: The value to unwrap. It can be of type T or None.

    Returns:
        The unwrapped value if it is not None.

    Raises:
        ValueError: If the provided value is None.

    """
    if el is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return el


def unwrap2[T1, T2](tpl: tuple[T1 | None, T2 | None]) -> tuple[T1, T2]:
    """Unwraps a tuple of optional values, raising an error if either of them is None.

    Args:
        tpl: The value to unwrap.

    Returns:
        The unwrapped values if it is not None.

    Raises:
        ValueError: If the provided value is None.

    """
    a, b = tpl
    if a is None or b is None:
        msg = "Unexpected None"
        raise ValueError(msg)
    return a, b


@dataclass
class Variable:
    """Container for variable meta information."""

    initial_value: float | InitialAssignment
    unit: sympy.Expr | None = None
    source: str | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class Parameter:
    """Container for parameter meta information."""

    value: float | InitialAssignment
    unit: sympy.Expr | None = None
    source: str | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass(kw_only=True, slots=True)
class Derived:
    """Container for a derived value."""

    fn: RateFn
    args: list[str]
    unit: sympy.Expr | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def calculate(self, args: dict[str, Any]) -> float:
        """Calculate the derived value.

        Args:
            args: Dictionary of args variables.

        Returns:
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the derived value in place.

        Args:
            name: Name of the derived variable.
            args: Dictionary of args variables.

        """
        args[name] = cast(float, self.fn(*(args[arg] for arg in self.args)))


@dataclass(kw_only=True, slots=True)
class InitialAssignment:
    """Container for a derived value."""

    fn: RateFn
    args: list[str]
    unit: sympy.Expr | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def calculate(self, args: dict[str, Any]) -> float:
        """Calculate the derived value.

        Args:
            args: Dictionary of args variables.

        Returns:
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the derived value in place.

        Args:
            name: Name of the derived variable.
            args: Dictionary of args variables.

        """
        args[name] = cast(float, self.fn(*(args[arg] for arg in self.args)))


@dataclass(kw_only=True, slots=True)
class Readout:
    """Container for a readout."""

    fn: RateFn
    args: list[str]
    unit: sympy.Expr | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def calculate(self, args: dict[str, Any]) -> float:
        """Calculate the derived value.

        Args:
            args: Dictionary of args variables.

        Returns:
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the reaction in place.

        Args:
            name: Name of the derived variable.
            args: Dictionary of args variables.

        """
        args[name] = cast(float, self.fn(*(args[arg] for arg in self.args)))


@dataclass(kw_only=True, slots=True)
class Reaction:
    """Container for a reaction."""

    fn: RateFn
    stoichiometry: Mapping[str, float | Derived]
    args: list[str]
    unit: sympy.Expr | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_modifiers(self, model: Model) -> list[str]:
        """Get the modifiers of the reaction."""
        include = set(model.get_variable_names())
        exclude = set(self.stoichiometry)

        return [k for k in self.args if k in include and k not in exclude]

    def calculate(self, args: dict[str, Any]) -> float:
        """Calculate the derived value.

        Args:
            args: Dictionary of args variables.

        Returns:
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the reaction in place.

        Args:
            name: Name of the derived variable.
            args: Dictionary of args variables.

        """
        args[name] = cast(float, self.fn(*(args[arg] for arg in self.args)))
