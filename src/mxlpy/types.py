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
    "FitFailure",
    "InitialAssignment",
    "IntegrationFailure",
    "NoSteadyState",
    "Option",
    "Param",
    "Parameter",
    "RateFn",
    "Reaction",
    "Readout",
    "Result",
    "RetType",
    "Rhs",
    "Variable",
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


class IntegrationFailure(Exception):
    """Custom exception."""

    message: str = "Simulation failed because of integration problems."

    def __init__(self) -> None:
        """Initialise."""
        super().__init__(self.message)


class NoSteadyState(Exception):
    """Custom exception."""

    message: str = "Could not find a steady-state."

    def __init__(self) -> None:
        """Initialise."""
        super().__init__(self.message)


class FitFailure(Exception):
    """Custom exception."""

    message: str = "Could not find a good fit."
    extra_info: list[str]

    def __init__(self, extra_info: list[str] | None) -> None:
        """Initialise."""
        super().__init__(self.message)
        self.extra_info = [] if extra_info is None else extra_info


@dataclass(slots=True)
class Option[T]:
    """Generic Option type."""

    value: T | None

    def unwrap(self) -> T:
        """Obtain value if Ok, else raise exception."""
        if (value := self.value) is None:
            msg = "Unexpected `None`"
            raise ValueError(msg)
        return value

    def default(self, fn: Callable[[], T]) -> T:
        """Obtain value if Ok, else create default one."""
        if (value := self.value) is None:
            return fn()
        return value


@dataclass(slots=True)
class Result[T]:
    """Generic Result type."""

    value: T | Exception

    def unwrap(self) -> T:
        """Obtain value if Ok, else raise exception."""
        if isinstance(value := self.value, Exception):
            raise value
        return value

    def default(self, fn: Callable[[], T]) -> T:
        """Obtain value if Ok, else create default one."""
        if isinstance(value := self.value, Exception):
            return fn()
        return value


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
