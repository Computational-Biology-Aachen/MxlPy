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
    Literal,
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
    "Event",
    "FitFailure",
    "InitialAssignment",
    "IntegrationFailure",
    "NoSteadyState",
    "Option",
    "OscillationDetected",
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


class OscillationDetected(Exception):
    """Raised when oscillatory behaviour is detected during steady-state search.

    Attributes
    ----------
    oscillating_species
        Names of the variables exhibiting oscillatory behaviour.
    period
        Estimated oscillation period in simulation time units, or ``None``
        if it could not be determined.

    """

    message: str = "Oscillatory behaviour detected; no steady state exists."
    oscillating_species: list[str]
    period: float | None

    def __init__(
        self,
        oscillating_species: list[str],
        period: float | None = None,
    ) -> None:
        """Initialise.

        Parameters
        ----------
        oscillating_species
            Names (or index strings) of the variables identified as oscillating.
        period
            Estimated oscillation period, or ``None`` if undetermined.

        """
        super().__init__(self.message)
        self.oscillating_species = oscillating_species
        self.period = period


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

    def unwrap_or_err(self) -> T:
        """Obtain value if Ok, else raise exception."""
        if (value := self.value) is None:
            msg = "Option contains None - use .default(fn) to provide a fallback value instead of .unwrap_or_err()"
            raise ValueError(msg)
        return value

    def default(self, fn: Callable[[], T]) -> T:
        """Obtain value if Ok, else create default one.

        Parameters
        ----------
        fn
            Factory function called to produce a default value when the option
            is None.

        Returns
        -------
        T
            The contained value if present, otherwise the result of calling *fn*.

        """
        if (value := self.value) is None:
            return fn()
        return value


@dataclass(slots=True)
class Result[T]:
    """Generic Result type."""

    value: T | Exception

    def unwrap_or_err(self) -> T:
        """Obtain value if Ok, else raise exception."""
        if isinstance(value := self.value, Exception):
            raise value
        return value

    def default(self, fn: Callable[[], T]) -> T:
        """Obtain value if Ok, else create default one.

        Parameters
        ----------
        fn
            Factory function called to produce a default value when the result
            contains an exception.

        Returns
        -------
        T
            The contained value if Ok, otherwise the result of calling *fn*.

        """
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

        Parameters
        ----------
        args
            Dictionary of args variables.

        Returns
        -------
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the derived value in place.

        Parameters
        ----------
        name
            Name of the derived variable.
        args
            Dictionary of args variables.

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

        Parameters
        ----------
        args
            Dictionary of args variables.

        Returns
        -------
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the derived value in place.

        Parameters
        ----------
        name
            Name of the derived variable.
        args
            Dictionary of args variables.

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

        Parameters
        ----------
        args
            Dictionary of args variables.

        Returns
        -------
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the reaction in place.

        Parameters
        ----------
        name
            Name of the derived variable.
        args
            Dictionary of args variables.

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
        """Get the modifiers of the reaction.

        Parameters
        ----------
        model
            Model instance used to determine which args are variables.

        Returns
        -------
        list[str]
            Variable names that appear in args but not in the stoichiometry.

        """
        include = set(model.get_variable_names())
        exclude = set(self.stoichiometry)

        return [k for k in self.args if k in include and k not in exclude]

    def calculate(self, args: dict[str, Any]) -> float:
        """Calculate the derived value.

        Parameters
        ----------
        args
            Dictionary of args variables.

        Returns
        -------
            The calculated derived value.

        """
        return cast(float, self.fn(*(args[arg] for arg in self.args)))

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Calculate the reaction in place.

        Parameters
        ----------
        name
            Name of the derived variable.
        args
            Dictionary of args variables.

        """
        args[name] = cast(float, self.fn(*(args[arg] for arg in self.args)))


@dataclass(kw_only=True, slots=True)
class Event:
    """Zero-crossing trigger with state/parameter assignments.

    Parameters
    ----------
    trigger_fn
        Returns a float; the event fires when this value crosses zero.
    trigger_args
        Model names passed as positional args to *trigger_fn*.
    assignments
        Map from target name to a :class:`Derived` that computes the new value.
        Use a zero-arg :class:`Derived` for static assignments.
    direction
        Which crossing fires the event: ``"rising"`` (negative→positive),
        ``"falling"`` (positive→negative), or ``"both"``.
    persistent
        If ``True`` the event fires every time the trigger crosses zero.
        If ``False`` it fires once and is disabled for the rest of the simulation.

    Examples
    --------
    >>> def at_t5(t: float) -> float:
    ...     return t - 5.0
    >>> def set_zero() -> float:
    ...     return 0.0
    >>> event = Event(
    ...     trigger_fn=at_t5,
    ...     trigger_args=["time"],
    ...     assignments={"k1": Derived(fn=set_zero, args=[])},
    ... )

    """

    trigger_fn: RateFn
    trigger_args: list[str]
    assignments: dict[str, Derived]
    direction: Literal["rising", "falling", "both"] = "both"
    persistent: bool = True

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def evaluate_trigger(self, args: dict[str, Any]) -> float:
        """Evaluate the trigger function with named args.

        Parameters
        ----------
        args
            Dict containing all named model quantities (variables, parameters, time).

        Returns
        -------
        float
            Sign change means the event fires.

        """
        return cast(float, self.trigger_fn(*(args[k] for k in self.trigger_args)))

    def apply_assignments(self, args: dict[str, Any]) -> dict[str, float]:
        """Compute new values for all assignment targets.

        Parameters
        ----------
        args
            Dict containing all named model quantities at the event time.

        Returns
        -------
        dict[str, float]
            Map from target name to its new value.

        """
        return {
            name: derived.calculate(args) for name, derived in self.assignments.items()
        }
