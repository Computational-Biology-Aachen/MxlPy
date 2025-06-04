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

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "AbstractEstimator",
    "AbstractSurrogate",
    "Array",
    "ArrayLike",
    "Derived",
    "IntegratorProtocol",
    "IntegratorType",
    "McSteadyStates",
    "MockSurrogate",
    "Param",
    "Parameter",
    "ProtocolByPars",
    "RateFn",
    "Reaction",
    "Readout",
    "ResponseCoefficients",
    "ResponseCoefficientsByPars",
    "RetType",
    "SteadyStates",
    "TimeCourseByPars",
    "Variable",
    "unwrap",
    "unwrap2",
]

type RateFn = Callable[..., float]
type Array = NDArray[np.floating[Any]]
type ArrayLike = NDArray[np.floating[Any]] | pd.Index | list[float]


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


class IntegratorProtocol(Protocol):
    """Protocol for numerical integrators."""

    def __init__(
        self,
        rhs: Callable,
        y0: ArrayLike,
        jacobian: Callable | None = None,
    ) -> None:
        """Initialise the integrator."""
        ...

    def reset(self) -> None:
        """Reset the integrator."""
        ...

    def integrate(
        self,
        *,
        t_end: float,
        steps: int | None = None,
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the system."""
        ...

    def integrate_time_course(
        self, *, time_points: ArrayLike
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the system over a time course."""
        ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the system to steady state."""
        ...


type IntegratorType = Callable[
    [Callable, ArrayLike, Callable | None], IntegratorProtocol
]


@dataclass
class Parameter:
    """Container for parameter meta information."""

    value: float
    unit: sympy.Expr | None = None
    source: str | None = None


@dataclass
class Variable:
    """Container for variable meta information."""

    initial_value: float | Derived
    unit: sympy.Expr | None = None
    source: str | None = None


@dataclass(kw_only=True, slots=True)
class Derived:
    """Container for a derived value."""

    fn: RateFn
    args: list[str]
    unit: sympy.Expr | None = None

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


@dataclass(kw_only=True, slots=True)
class ResponseCoefficients:
    """Container for response coefficients."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)


@dataclass(kw_only=True, slots=True)
class ResponseCoefficientsByPars:
    """Container for response coefficients by parameter."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)


@dataclass(kw_only=True, slots=True)
class SteadyStates:
    """Container for steady states."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the steady states as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)


@dataclass(kw_only=True, slots=True)
class McSteadyStates:
    """Container for Monte Carlo steady states."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    mc_to_scan: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the steady states as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)


@dataclass(kw_only=True, slots=True)
class TimeCourseByPars:
    """Container for time courses by parameter."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux time courses."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get time courses by name."""
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)


@dataclass(kw_only=True, slots=True)
class ProtocolByPars:
    """Container for protocols by parameter."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    protocol: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux protocols."""
        return iter((self.variables, self.fluxes))

    @property
    def results(self) -> pd.DataFrame:
        """Return the protocols as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get concentration or flux by name."""
        return self.results[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.results.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.results.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)


@dataclass(kw_only=True)
class AbstractEstimator:
    """Abstract class for parameter estimation using neural networks."""

    parameter_names: list[str]

    @abstractmethod
    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""
