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
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from wadler_lindig import pformat

__all__ = [
    "AbstractEstimator",
    "AbstractSurrogate",
    "Array",
    "ArrayLike",
    "Derived",
    "InitialAssignment",
    "IntegratorProtocol",
    "IntegratorType",
    "McSteadyStates",
    "MockSurrogate",
    "Param",
    "Parameter",
    "ProtocolScan",
    "RateFn",
    "Reaction",
    "Readout",
    "ResponseCoefficients",
    "ResponseCoefficientsByPars",
    "Result",
    "RetType",
    "Rhs",
    "SteadyStateScan",
    "TimeCourseScan",
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


class IntegratorProtocol(Protocol):
    """Protocol for numerical integrators."""

    def __init__(
        self,
        rhs: Rhs,
        y0: tuple[float, ...],
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
    [
        Rhs,  # model
        tuple[float, ...],  # y0
        Callable | None,  # jacobian
    ],
    IntegratorProtocol,
]


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


@dataclass(kw_only=True)
class AbstractEstimator:
    """Abstract class for parameter estimation using neural networks."""

    parameter_names: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @abstractmethod
    def predict(self, features: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Predict the target values for the given features."""


###############################################################################
# Simulation results
###############################################################################


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    """Normalize split results by a given factor or array.

    Args:
        results: List of DataFrames containing the results to normalize.
        normalise: Normalization factor or array.

    Returns:
        list[pd.DataFrame]: List of normalized DataFrames.

    """
    if isinstance(normalise, int | float):
        return [i / normalise for i in results]
    if len(normalise) == len(results):
        return [(i.T / j).T for i, j in zip(results, normalise, strict=True)]

    results = []
    start = 0
    end = 0
    for i in results:
        end += len(i)
        results.append(i / np.reshape(normalise[start:end], (len(i), 1)))  # type: ignore
        start += end
    return results


@dataclass(kw_only=True, slots=True)
class Result:
    """Simulation results."""

    model: Model
    raw_variables: list[pd.DataFrame]
    raw_parameters: list[dict[str, float]]
    raw_args: list[pd.DataFrame] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @classmethod
    def default(cls, model: Model, time_points: Array) -> Result:
        """Get result filled with NaNs."""
        return Result(
            model=model,
            raw_variables=[
                pd.DataFrame(
                    data=np.full(
                        shape=(len(time_points), len(model.get_variable_names())),
                        fill_value=np.nan,
                    ),
                    index=time_points,
                    columns=model.get_variable_names(),
                )
            ],
            raw_parameters=[model.get_parameter_values()],
        )

    @property
    def variables(self) -> pd.DataFrame:
        """Simulation variables."""
        return self.get_variables(
            include_derived_variables=True,
            include_surrogate_variables=True,
            include_readouts=True,
            concatenated=True,
            normalise=None,
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Simulation fluxes."""
        return self.get_fluxes(
            include_surrogates=True,
        )

    def _compute_args(self) -> list[pd.DataFrame]:
        # Already computed
        if len(self.raw_args) > 0:
            return self.raw_args

        # Compute new otherwise
        for res, p in zip(self.raw_variables, self.raw_parameters, strict=True):
            self.model.update_parameters(p)
            self.raw_args.append(
                self.model.get_args_time_course(
                    variables=res,
                    include_variables=True,
                    include_parameters=True,
                    include_derived_parameters=True,
                    include_derived_variables=True,
                    include_reactions=True,
                    include_surrogate_variables=True,
                    include_surrogate_fluxes=True,
                    include_readouts=True,
                )
            )
        return self.raw_args

    def _select_data(
        self,
        dependent: list[pd.DataFrame],
        *,
        include_variables: bool = False,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = False,
        include_reactions: bool = False,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> list[pd.DataFrame]:
        names = self.model.get_arg_names(
            include_time=False,
            include_variables=include_variables,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_reactions=include_reactions,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_fluxes=include_surrogate_fluxes,
            include_readouts=include_readouts,
        )
        return [i.loc[:, names] for i in dependent]

    def _adjust_data(
        self,
        data: list[pd.DataFrame],
        normalise: float | ArrayLike | None = None,
        *,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        if normalise is not None:
            data = _normalise_split_results(data, normalise=normalise)
        if concatenated:
            return pd.concat(data, axis=0)
        return data

    @overload
    def get_args(  # type: ignore
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the variables over time.

        Examples:
            >>> Result().get_variables()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        """
        variables = self._select_data(
            self._compute_args(),
            include_variables=include_variables,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_reactions=include_reactions,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_fluxes=include_surrogate_fluxes,
            include_readouts=include_readouts,
        )
        return self._adjust_data(
            variables, normalise=normalise, concatenated=concatenated
        )

    @overload
    def get_variables(  # type: ignore
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the variables over time.

        Examples:
            >>> Result().get_variables()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        """
        if not (
            include_derived_variables or include_readouts or include_surrogate_variables
        ):
            return self._adjust_data(
                self.raw_variables,
                normalise=normalise,
                concatenated=concatenated,
            )

        variables = self._select_data(
            self._compute_args(),
            include_variables=True,
            include_derived_variables=include_derived_variables,
            include_surrogate_variables=include_surrogate_variables,
            include_readouts=include_readouts,
        )
        return self._adjust_data(
            variables, normalise=normalise, concatenated=concatenated
        )

    @overload
    def get_fluxes(  # type: ignore
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the flux results.

        Examples:
            >>> Result.get_fluxes()
            Time             v1         v2
            0.000000   1.000000   10.00000
            0.000100   0.999900   9.999000
            0.000200   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

        """
        fluxes = self._select_data(
            self._compute_args(),
            include_reactions=True,
            include_surrogate_fluxes=include_surrogates,
        )
        return self._adjust_data(
            fluxes,
            normalise=normalise,
            concatenated=concatenated,
        )

    def get_combined(self) -> pd.DataFrame:
        """Get the variables and fluxes as a single pandas.DataFrame.

        Examples:
            >>> Result.get_combined()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   10.00000
            0.000100   0.999900   0.999900   0.999900   9.999000
            0.000200   0.999800   0.999800   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

        """
        return pd.concat((self.variables, self.fluxes), axis=1)

    @overload
    def get_right_hand_side(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get right hand side over time."""
        args_by_simulation = self._compute_args()
        return self._adjust_data(
            [
                self.model.update_parameters(p).get_right_hand_side_time_course(
                    args=args
                )
                for args, p in zip(args_by_simulation, self.raw_parameters, strict=True)
            ],
            normalise=normalise,
            concatenated=concatenated,
        )

    @overload
    def get_producers(  # type: ignore
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get fluxes of variable with positive stoichiometry."""
        self.model.update_parameters(self.raw_parameters[0])
        names = [
            k
            for k, v in self.model.get_stoichiometries_of_variable(variable).items()
            if v > 0
        ]

        fluxes: list[pd.DataFrame] = [
            i.loc[:, names]
            for i in self.get_fluxes(normalise=normalise, concatenated=False)
        ]

        if scaled:
            fluxes = [i.copy() for i in fluxes]
            for v, p in zip(fluxes, self.raw_parameters, strict=True):
                self.model.update_parameters(p)
                stoichs = self.model.get_stoichiometries_of_variable(variable)
                for k in names:
                    v.loc[:, k] *= stoichs[k]

        self.model.update_parameters(self.raw_parameters[-1])
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    @overload
    def get_consumers(  # type: ignore
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get fluxes of variable with negative stoichiometry."""
        self.model.update_parameters(self.raw_parameters[0])
        names = [
            k
            for k, v in self.model.get_stoichiometries_of_variable(variable).items()
            if v < 0
        ]

        fluxes: list[pd.DataFrame] = [
            i.loc[:, names]
            for i in self.get_fluxes(normalise=normalise, concatenated=False)
        ]

        if scaled:
            fluxes = [i.copy() for i in fluxes]
            for v, p in zip(fluxes, self.raw_parameters, strict=True):
                self.model.update_parameters(p)
                stoichs = self.model.get_stoichiometries_of_variable(variable)
                for k in names:
                    v.loc[:, k] *= -stoichs[k]

        self.model.update_parameters(self.raw_parameters[-1])
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    def get_new_y0(self) -> dict[str, float]:
        """Get the new initial conditions after the simulation.

        Examples:
            >>> Simulator(model).simulate_to_steady_state().get_new_y0()
            {"ATP": 1.0, "NADPH": 1.0}

        """
        return dict(
            self.get_variables(
                include_derived_variables=False,
                include_readouts=False,
                include_surrogate_variables=False,
            ).iloc[-1]
        )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class ResponseCoefficients:
    """Container for response coefficients."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class ResponseCoefficientsByPars:
    """Container for response coefficients by parameter."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return the response coefficients as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class SteadyStateScan:
    """Container for steady states by scanned values."""

    to_scan: pd.DataFrame
    raw_index: pd.Index | pd.MultiIndex
    raw_results: list[Result]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(self) -> pd.DataFrame:
        """Return steady-state variables by scan."""
        return pd.DataFrame(
            [i.variables.iloc[-1].T for i in self.raw_results], index=self.raw_index
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return steady-state fluxes by scan."""
        return pd.DataFrame(
            [i.fluxes.iloc[-1].T for i in self.raw_results], index=self.raw_index
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return steady-state args by scan."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return steady-state args by scan."""
        return pd.DataFrame(
            [
                i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                .iloc[-1]
                .T
                for i in self.raw_results
            ],
            index=self.raw_index,
        )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class TimeCourseScan:
    """Container for time courses by scanned values."""

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    to_scan: pd.DataFrame
    raw_results: dict[Hashable, Result]

    @property
    def variables(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.variables for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()}, names=["n", "time"]
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {
                k: i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                for k, i in self.raw_results.items()
            },
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get time courses by name."""
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated time courses."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux time courses."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class ProtocolScan:
    """Container for protocols by scanned values."""

    to_scan: pd.DataFrame
    protocol: pd.DataFrame
    raw_results: dict[Hashable, Result]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.variables for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {k: i.fluxes for k, i in self.raw_results.items()},
            names=["n", "time"],
        )

    @property
    def combined(self) -> pd.DataFrame:
        """Return the time courses as a DataFrame."""
        return self.get_args()

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Return all args of the time courses."""
        return pd.concat(
            {
                k: i.get_args(
                    include_variables=include_variables,
                    include_parameters=include_parameters,
                    include_derived_parameters=include_derived_parameters,
                    include_derived_variables=include_derived_variables,
                    include_reactions=include_reactions,
                    include_surrogate_variables=include_surrogate_variables,
                    include_surrogate_fluxes=include_surrogate_fluxes,
                    include_readouts=include_readouts,
                )
                for k, i in self.raw_results.items()
            },
            names=["n", "time"],
        )

    def get_by_name(self, name: str) -> pd.DataFrame:
        """Get concentration or flux by name."""
        return self.combined[name].unstack().T

    def get_agg_per_time(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=1).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def get_agg_per_run(self, agg: str | Callable) -> pd.DataFrame:
        """Get aggregated concentration or flux."""
        mean = cast(pd.DataFrame, self.combined.unstack(level=0).agg(agg, axis=0))
        return cast(pd.DataFrame, mean.unstack().T)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux protocols."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class McSteadyStates:
    """Container for Monte Carlo steady states."""

    variables: pd.DataFrame
    fluxes: pd.DataFrame
    parameters: pd.DataFrame
    mc_to_scan: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def combined(self) -> pd.DataFrame:
        """Return the steady states as a DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux steady states."""
        return iter((self.variables, self.fluxes))
