"""Model for Metabolic System Representation.

This module provides the core Model class and supporting functionality for representing
metabolic models, including reactions, variables, parameters and derived quantities.

"""

from __future__ import annotations

import copy
import inspect
import itertools as it
import logging
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pandas as pd
import sympy
from wadler_lindig import pformat

from mxlpy import fns
from mxlpy.meta.source_tools import fn_to_sympy
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
    stoichiometries_to_sympy,
)
from mxlpy.types import (
    AbstractSurrogate,
    Derived,
    InitialAssignment,
    Parameter,
    Reaction,
    Readout,
    Variable,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from inspect import FullArgSpec

    from sympy.physics.units.quantities import Quantity

    from mxlpy.types import Callable, Param, RateFn, RetType

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ArityMismatchError",
    "CircularDependencyError",
    "Dependency",
    "Failure",
    "LOGGER",
    "MdText",
    "MissingDependenciesError",
    "Model",
    "ModelCache",
    "TableView",
    "UnitCheck",
    "unit_of",
]


def _latex_view(expr: sympy.Expr | None) -> str:
    if expr is None:
        return "PARSE-ERROR"
    return f"${sympy.latex(expr)}$"


def unit_of(expr: sympy.Expr) -> sympy.Expr:
    """Get unit of sympy expr."""
    return expr.as_coeff_Mul()[1]


@dataclass
class Failure:
    """Unit test failure."""

    expected: sympy.Expr
    obtained: sympy.Expr

    @property
    def difference(self) -> sympy.Expr:
        """Difference between expected and obtained unit."""
        return self.expected / self.obtained  # type: ignore


@dataclass
class MdText:
    """Generic markdown text."""

    content: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def _repr_markdown_(self) -> str:
        return "\n".join(self.content)


@dataclass
class UnitCheck:
    """Container for unit check."""

    per_variable: dict[str, dict[str, bool | Failure | None]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @staticmethod
    def _fmt_success(s: str) -> str:
        return f"<span style='color: green'>{s}</span>"

    @staticmethod
    def _fmt_failed(s: str) -> str:
        return f"<span style='color: red'>{s}</span>"

    def correct_diff_eqs(self) -> dict[str, bool]:
        """Get all correctly annotated reactions by variable."""
        return {
            var: all(isinstance(i, bool) for i in checks.values())
            for var, checks in self.per_variable.items()
        }

    def report(self) -> MdText:
        """Export check as markdown report."""
        report = ["## Type check"]
        for diff_eq, res in self.correct_diff_eqs().items():
            txt = self._fmt_success("Correct") if res else self._fmt_failed("Failed")
            report.append(f"\n### d{diff_eq}dt: {txt}")

            if res:
                continue
            for k, v in self.per_variable[diff_eq].items():
                match v:
                    case bool():
                        continue
                    case None:
                        report.append(f"\n- {k}")
                        report.append("  - Failed to parse")
                    case Failure(expected, obtained):
                        report.append(f"\n- {k}")
                        report.append(f"  - expected: {_latex_view(expected)}")
                        report.append(f"  - obtained: {_latex_view(obtained)}")
                        report.append(f"  - difference: {_latex_view(v.difference)}")

        return MdText(report)


@dataclass(kw_only=True, slots=True)
class TableView:
    """Markdown view of pandas Dataframe.

    Mostly used to get nice LaTeX rendering of sympy expressions.
    """

    data: pd.DataFrame

    def __repr__(self) -> str:
        """Normal Python shell output."""
        return self.data.to_markdown()

    def _repr_markdown_(self) -> str:
        """Fancy IPython shell output.

        Looks the same as __repr__, but is handled by IPython to output
        `IPython.display.Markdown`, so looks nice
        """
        return self.data.to_markdown()


@dataclass
class Dependency:
    """Container class for building dependency tree."""

    name: str
    required: set[str]
    provided: set[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


class MissingDependenciesError(Exception):
    """Raised when dependencies cannot be sorted topologically.

    This typically indicates circular dependencies in model components.
    """

    def __init__(self, not_solvable: dict[str, list[str]]) -> None:
        """Initialise exception."""
        missing_by_module = "\n".join(f"\t{k}: {v}" for k, v in not_solvable.items())
        msg = (
            f"Dependencies cannot be solved. Missing dependencies:\n{missing_by_module}"
        )
        super().__init__(msg)


class CircularDependencyError(Exception):
    """Raised when dependencies cannot be sorted topologically.

    This typically indicates circular dependencies in model components.
    """

    def __init__(
        self,
        missing: dict[str, set[str]],
    ) -> None:
        """Initialise exception."""
        missing_by_module = "\n".join(f"\t{k}: {v}" for k, v in missing.items())
        msg = (
            f"Exceeded max iterations on sorting dependencies.\n"
            "Check if there are circular references. "
            "Missing dependencies:\n"
            f"{missing_by_module}"
        )
        super().__init__(msg)


def _get_all_args(argspec: FullArgSpec) -> list[str]:
    kwonly = [] if argspec.kwonlyargs is None else argspec.kwonlyargs
    return argspec.args + kwonly


def _check_function_arity(function: Callable, arity: int) -> bool:
    """Check if the amount of arguments given fits the argument count of the function."""
    argspec = inspect.getfullargspec(function)
    # Give up on *args functions
    if argspec.varargs is not None:
        return True

    # The sane case
    if len(argspec.args) == arity:
        return True

    # It might be that the user has set some args to default values,
    # in which case they are also ok (might be kwonly as well)
    defaults = argspec.defaults
    if defaults is not None and len(argspec.args) + len(defaults) == arity:
        return True
    kwonly = argspec.kwonlyargs
    return bool(defaults is not None and len(argspec.args) + len(kwonly) == arity)


class ArityMismatchError(Exception):
    """Mismatch between python function and model arguments."""

    def __init__(self, name: str, fn: Callable, args: list[str]) -> None:
        """Format message."""
        argspec = inspect.getfullargspec(fn)

        message = f"Function arity mismatch for {name}.\n"
        message += "\n".join(
            (
                f"{i:<8.8} | {j:<10.10}"
                for i, j in [
                    ("Fn args", "Model args"),
                    ("-------", "----------"),
                    *it.zip_longest(argspec.args, args, fillvalue="---"),
                ]
            )
        )
        super().__init__(message)


def _invalidate_cache(method: Callable[Param, RetType]) -> Callable[Param, RetType]:
    """Decorator that invalidates model cache when decorated method is called.

    Args:
        method: Method to wrap with cache invalidation

    Returns:
        Wrapped method that clears cache before execution

    """

    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(Model, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


def _check_if_is_sortable(
    available: set[str],
    elements: list[Dependency],
) -> None:
    all_available = available.copy()
    for dependency in elements:
        all_available.update(dependency.provided)

    # Check if it can be sorted in the first place
    not_solvable = {}
    for dependency in elements:
        if not dependency.required.issubset(all_available):
            not_solvable[dependency.name] = sorted(
                dependency.required.difference(all_available)
            )

    if not_solvable:
        raise MissingDependenciesError(not_solvable=not_solvable)


def _sort_dependencies(
    available: set[str],
    elements: list[Dependency],
) -> list[str]:
    """Sort model elements topologically based on their dependencies.

    Args:
        available: Set of available component names
        elements: List of (name, dependencies, supplier) tuples to sort

    Returns:
        List of element names in dependency order

    Raises:
        SortError: If circular dependencies are detected

    """
    _check_if_is_sortable(available, elements)

    order = []
    # FIXME: what is the worst case here?
    max_iterations = len(elements) ** 2
    queue: SimpleQueue[Dependency] = SimpleQueue()
    for dependency in elements:
        queue.put(dependency)

    last_name = None
    i = 0
    while True:
        try:
            dependency = queue.get_nowait()
        except Empty:
            break
        if dependency.required.issubset(available):
            available.update(dependency.provided)
            order.append(dependency.name)

        else:
            if last_name == dependency.name:
                order.append(last_name)
                break
            queue.put(dependency)
            last_name = dependency.name
        i += 1

        # Failure case
        if i > max_iterations:
            unsorted = []
            while True:
                try:
                    unsorted.append(queue.get_nowait().name)
                except Empty:
                    break

            mod_to_args: dict[str, set[str]] = {
                dependency.name: dependency.required for dependency in elements
            }
            missing = {k: mod_to_args[k].difference(available) for k in unsorted}
            raise CircularDependencyError(missing=missing)
    return order


@dataclass(slots=True)
class ModelCache:
    """ModelCache is a class that stores various model-related data structures.

    Attributes:
        var_names: A list of variable names.
        parameter_values: A dictionary mapping parameter names to their values.
        derived_parameters: A dictionary mapping parameter names to their derived parameter objects.
        derived_variables: A dictionary mapping variable names to their derived variable objects.
        stoich_by_cpds: A dictionary mapping compound names to their stoichiometric coefficients.
        dyn_stoich_by_cpds: A dictionary mapping compound names to their dynamic stoichiometric coefficients.
        dxdt: A pandas Series representing the rate of change of variables.
        initial_conditions: calculated initial conditions

    """

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    order: list[str]  # mostly for debug purposes
    var_names: list[str]
    dyn_order: list[str]
    base_parameter_values: dict[str, float]
    all_parameter_values: dict[str, float]
    stoich_by_cpds: dict[str, dict[str, float]]
    dyn_stoich_by_cpds: dict[str, dict[str, Derived]]
    initial_conditions: dict[str, float]


@dataclass(slots=True)
class Model:
    """Represents a metabolic model.

    Attributes:
        _ids: Dictionary mapping internal IDs to names.
        _variables: Dictionary of model variables and their initial values.
        _parameters: Dictionary of model parameters and their values.
        _derived: Dictionary of derived quantities.
        _readouts: Dictionary of readout functions.
        _reactions: Dictionary of reactions in the model.
        _surrogates: Dictionary of surrogate models.
        _cache: Cache for storing model-related data structures.
        _data: Named references to data sets

    """

    _ids: dict[str, str] = field(default_factory=dict, repr=False)
    _cache: ModelCache | None = field(default=None, repr=False)
    _variables: dict[str, Variable] = field(default_factory=dict)
    _parameters: dict[str, Parameter] = field(default_factory=dict)
    _derived: dict[str, Derived] = field(default_factory=dict)
    _readouts: dict[str, Readout] = field(default_factory=dict)
    _reactions: dict[str, Reaction] = field(default_factory=dict)
    _surrogates: dict[str, AbstractSurrogate] = field(default_factory=dict)
    _data: dict[str, pd.Series | pd.DataFrame] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    ###########################################################################
    # Cache
    ###########################################################################

    def _create_cache(self) -> ModelCache:
        """Creates and initializes the model cache.

        This method constructs a cache that includes parameter values, stoichiometry
        by compounds, dynamic stoichiometry by compounds, derived variables, and
        derived parameters. It processes the model's parameters, variables, derived
        elements, reactions, and surrogates to populate the cache.

        Returns:
            ModelCache: An instance of ModelCache containing the initialized cache data.

        """
        parameter_names = set(self._parameters)
        all_parameter_names = set(parameter_names)  # later include static derived

        base_parameter_values: dict[str, float] = {
            k: val
            for k, v in self._parameters.items()
            if not isinstance(val := v.value, InitialAssignment)
        }
        base_variable_values: dict[str, float] = {
            k: init
            for k, v in self._variables.items()
            if not isinstance(init := v.initial_value, InitialAssignment)
        }
        initial_assignments: dict[str, InitialAssignment] = {
            k: init
            for k, v in self._variables.items()
            if isinstance(init := v.initial_value, InitialAssignment)
        } | {
            k: init
            for k, v in self._parameters.items()
            if isinstance(init := v.value, InitialAssignment)
        }

        # Sanity checks
        for name, el in it.chain(
            initial_assignments.items(),
            self._derived.items(),
            self._reactions.items(),
            self._readouts.items(),
        ):
            if not _check_function_arity(el.fn, len(el.args)):
                raise ArityMismatchError(name, el.fn, el.args)

        # Sort derived & reactions
        available = (
            set(base_parameter_values)
            | set(base_variable_values)
            | set(self._data)
            | {"time"}
        )
        to_sort = (
            initial_assignments | self._derived | self._reactions | self._surrogates
        )
        order = _sort_dependencies(
            available=available,
            elements=[
                Dependency(name=k, required=set(v.args), provided={k})
                if not isinstance(v, AbstractSurrogate)
                else Dependency(name=k, required=set(v.args), provided=set(v.outputs))
                for k, v in to_sort.items()
            ],
        )

        # Calculate all values once, including dynamic ones
        # That way, we can make initial conditions dependent on e.g. rates
        dependent = (
            base_parameter_values | base_variable_values | self._data | {"time": 0.0}
        )
        for name in order:
            to_sort[name].calculate_inpl(name, dependent)

        # Split derived into static and dynamic variables
        static_order = []
        dyn_order = []
        for name in order:
            if name in self._reactions or name in self._surrogates:
                dyn_order.append(name)
            elif name in self._variables or name in self._parameters:
                static_order.append(name)
            else:
                derived = self._derived[name]
                if all(i in all_parameter_names for i in derived.args):
                    static_order.append(name)
                    all_parameter_names.add(name)
                else:
                    dyn_order.append(name)

        # Calculate dynamic and static stochiometries
        stoich_by_compounds: dict[str, dict[str, float]] = {}
        dyn_stoich_by_compounds: dict[str, dict[str, Derived]] = {}
        for rxn_name, rxn in self._reactions.items():
            for cpd_name, factor in rxn.stoichiometry.items():
                d_static = stoich_by_compounds.setdefault(cpd_name, {})
                if isinstance(factor, Derived):
                    if all(i in all_parameter_names for i in factor.args):
                        d_static[rxn_name] = factor.calculate(dependent)
                    else:
                        dyn_stoich_by_compounds.setdefault(cpd_name, {})[rxn_name] = (
                            factor
                        )
                else:
                    d_static[rxn_name] = factor

        for surrogate in self._surrogates.values():
            for rxn_name, rxn in surrogate.stoichiometries.items():
                for cpd_name, factor in rxn.items():
                    d_static = stoich_by_compounds.setdefault(cpd_name, {})
                    if isinstance(factor, Derived):
                        if all(i in all_parameter_names for i in factor.args):
                            d_static[rxn_name] = factor.calculate(dependent)
                        else:
                            dyn_stoich_by_compounds.setdefault(cpd_name, {})[
                                rxn_name
                            ] = factor
                    else:
                        d_static[rxn_name] = factor

        var_names = self.get_variable_names()
        initial_conditions: dict[str, float] = {
            k: cast(float, dependent[k]) for k in self._variables
        }
        all_parameter_values = dict(base_parameter_values)
        for name in static_order:
            if name in self._variables:
                continue  # handled in initial_conditions above
            if name in self._parameters or name in self._derived:
                all_parameter_values[name] = cast(float, dependent[name])
            else:
                msg = "Unknown target for static derived variable."
                raise KeyError(msg)

        self._cache = ModelCache(
            order=order,
            var_names=var_names,
            dyn_order=dyn_order,
            base_parameter_values=base_parameter_values,
            all_parameter_values=all_parameter_values,
            stoich_by_cpds=stoich_by_compounds,
            dyn_stoich_by_cpds=dyn_stoich_by_compounds,
            initial_conditions=initial_conditions,
        )
        return self._cache

    ###########################################################################
    # Ids
    ###########################################################################

    @property
    def ids(self) -> dict[str, str]:
        """Returns a copy of the _ids dictionary.

        The _ids dictionary contains key-value pairs where both keys and values are strings.

        Returns:
            dict[str, str]: A copy of the _ids dictionary.

        """
        return self._ids.copy()

    def _insert_id(self, *, name: str, ctx: str) -> None:
        """Inserts an identifier into the model's internal ID dictionary.

        Args:
            name: The name of the identifier to insert.
            ctx: The context associated with the identifier.

        Raises:
            KeyError: If the name is "time", which is a protected variable.
            NameError: If the name already exists in the model's ID dictionary.

        """
        if name == "time":
            msg = "time is a protected variable for time"
            raise KeyError(msg)

        if name in self._ids:
            msg = f"Model already contains {ctx} called '{name}'"
            raise NameError(msg)
        self._ids[name] = ctx

    def _remove_id(self, *, name: str) -> None:
        """Remove an ID from the internal dictionary.

        Args:
            name (str): The name of the ID to be removed.

        Raises:
            KeyError: If the specified name does not exist in the dictionary.

        """
        del self._ids[name]

    ##########################################################################
    # Parameters - views
    ##########################################################################

    @property
    def parameters(self) -> TableView:
        """Return view of parameters."""
        index = list(self._parameters.keys())
        data = []
        for name, el in self._parameters.items():
            if isinstance(init := el.value, InitialAssignment):
                value_str = _latex_view(
                    fn_to_sympy(
                        init.fn,
                        origin=name,
                        model_args=list_of_symbols(init.args),
                    )
                )
            else:
                value_str = str(init)
            data.append(
                {
                    "value": value_str,
                    "unit": _latex_view(unit) if (unit := el.unit) is not None else "",
                    # "source": ...,
                }
            )
        return TableView(data=pd.DataFrame(data, index=index))

    def get_raw_parameters(self, *, as_copy: bool = True) -> dict[str, Parameter]:
        """Returns the parameters of the model."""
        if as_copy:
            return copy.deepcopy(self._parameters)
        return self._parameters

    def get_parameter_values(self) -> dict[str, float]:
        """Returns the parameters of the model.

        Examples:
            >>> model.parameters
                {"k1": 0.1, "k2": 0.2}

        Returns:
            parameters: A dictionary where the keys are parameter names (as strings)
                  and the values are parameter values (as floats).

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.base_parameter_values

    def get_parameter_names(self) -> list[str]:
        """Retrieve the names of the parameters.

        Examples:
            >>> model.get_parameter_names()
                ['k1', 'k2']

        Returns:
            parametes: A list containing the names of the parameters.

        """
        return list(self._parameters)

    #####################################
    # Parameters - create
    #####################################

    @_invalidate_cache
    def add_parameter(
        self,
        name: str,
        value: float | InitialAssignment,
        unit: sympy.Expr | None = None,
        source: str | None = None,
    ) -> Self:
        """Adds a parameter to the model.

        Examples:
            >>> model.add_parameter("k1", 0.1)

        Args:
            name: The name of the parameter.
            value: The value of the parameter.
            unit: unit of the parameter
            source: source of the information given

        Returns:
            Self: The instance of the model with the added parameter.

        """
        self._insert_id(name=name, ctx="parameter")
        self._parameters[name] = Parameter(value=value, unit=unit, source=source)
        return self

    def add_parameters(
        self, parameters: Mapping[str, float | Parameter | InitialAssignment]
    ) -> Self:
        """Adds multiple parameters to the model.

        Examples:
            >>> model.add_parameters({"k1": 0.1, "k2": 0.2})

        Args:
            parameters (dict[str, float]): A dictionary where the keys are parameter names
                                           and the values are the corresponding parameter values.

        Returns:
            Self: The instance of the model with the added parameters.

        """
        for k, v in parameters.items():
            if isinstance(v, Parameter):
                self.add_parameter(k, v.value, unit=v.unit, source=v.source)
            else:
                self.add_parameter(k, v)
        return self

    #####################################
    # Parameters - delete
    #####################################

    @_invalidate_cache
    def remove_parameter(self, name: str) -> Self:
        """Remove a parameter from the model.

        Examples:
            >>> model.remove_parameter("k1")

        Args:
            name: The name of the parameter to remove.

        Returns:
            Self: The instance of the model with the parameter removed.

        """
        self._remove_id(name=name)
        self._parameters.pop(name)
        return self

    def remove_parameters(self, names: list[str]) -> Self:
        """Remove multiple parameters from the model.

        Examples:
            >>> model.remove_parameters(["k1", "k2"])

        Args:
            names: A list of parameter names to be removed.

        Returns:
            Self: The instance of the model with the specified parameters removed.

        """
        for name in names:
            self.remove_parameter(name)
        return self

    #####################################
    # Parameters - update
    #####################################

    @_invalidate_cache
    def update_parameter(
        self,
        name: str,
        value: float | InitialAssignment | None = None,
        *,
        unit: sympy.Expr | None = None,
        source: str | None = None,
    ) -> Self:
        """Update the value of a parameter.

        Examples:
            >>> model.update_parameter("k1", 0.2)

        Args:
            name: The name of the parameter to update.
            value: The new value for the parameter.
            unit: Unit of the parameter
            source: Source of the information

        Returns:
            Self: The instance of the class with the updated parameter.

        Raises:
            NameError: If the parameter name is not found in the parameters.

        """
        if name not in self._parameters:
            msg = f"{name!r} not found in parameters"
            raise KeyError(msg)

        parameter = self._parameters[name]
        if value is not None:
            parameter.value = value
        if unit is not None:
            parameter.unit = unit
        if source is not None:
            parameter.source = source
        return self

    def update_parameters(
        self, parameters: Mapping[str, float | Parameter | InitialAssignment]
    ) -> Self:
        """Update multiple parameters of the model.

        Examples:
            >>> model.update_parameters({"k1": 0.2, "k2": 0.3})

        Args:
            parameters: A dictionary where keys are parameter names and values are the new parameter values.

        Returns:
            Self: The instance of the model with updated parameters.

        """
        for k, v in parameters.items():
            if isinstance(v, Parameter):
                self.update_parameter(k, value=v.value, unit=v.unit, source=v.source)
            else:
                self.update_parameter(k, v)
        return self

    def scale_parameter(self, name: str, factor: float) -> Self:
        """Scales the value of a specified parameter by a given factor.

        Examples:
            >>> model.scale_parameter("k1", 2.0)

        Args:
            name: The name of the parameter to be scaled.
            factor: The factor by which to scale the parameter's value.

        Returns:
            Self: The instance of the class with the updated parameter.

        """
        old = self._parameters[name].value
        if isinstance(old, InitialAssignment):
            LOGGER.warning("Overwriting initial assignment %s", name)
            if (cache := self._cache) is None:
                cache = self._create_cache()

            return self.update_parameter(
                name, cache.all_parameter_values[name] * factor
            )

        return self.update_parameter(name, old * factor)

    def scale_parameters(self, parameters: dict[str, float]) -> Self:
        """Scales the parameters of the model.

        Examples:
            >>> model.scale_parameters({"k1": 2.0, "k2": 0.5})

        Args:
            parameters: A dictionary where the keys are parameter names
                        and the values are the scaling factors.

        Returns:
            Self: The instance of the model with scaled parameters.

        """
        for k, v in parameters.items():
            self.scale_parameter(k, v)
        return self

    @_invalidate_cache
    def make_parameter_dynamic(
        self,
        name: str,
        initial_value: float | None = None,
        stoichiometries: dict[str, float] | None = None,
    ) -> Self:
        """Converts a parameter to a dynamic variable in the model.

        Examples:
            >>> model.make_parameter_dynamic("k1")
            >>> model.make_parameter_dynamic("k2", initial_value=0.5)

        This method removes the specified parameter from the model and adds it as a variable with an optional initial value.

        Args:
            name: The name of the parameter to be converted.
            initial_value: The initial value for the new variable. If None, the current value of the parameter is used. Defaults to None.
            stoichiometries: A dictionary mapping reaction names to stoichiometries for the new variable. Defaults to None.

        Returns:
            Self: The instance of the model with the parameter converted to a variable.

        """
        value = self._parameters[name].value if initial_value is None else initial_value
        self.remove_parameter(name)
        self.add_variable(name, value)

        if stoichiometries is not None:
            for rxn_name, value in stoichiometries.items():
                target = False
                if (rxn := self._reactions.get(rxn_name)) is not None:
                    target = True
                    cast(dict, rxn.stoichiometry)[name] = value
                else:
                    for surrogate in self._surrogates.values():
                        if stoich := surrogate.stoichiometries.get(rxn_name):
                            target = True
                            stoich[name] = value
                if not target:
                    msg = f"Reaction '{rxn_name}' not found in reactions or surrogates"
                    raise KeyError(msg)

        return self

    def get_unused_parameters(self) -> set[str]:
        """Get parameters which aren't used in the model."""
        args = set()
        for variable in self._variables.values():
            if isinstance(variable, Derived):
                args.update(variable.args)
        for derived in self._derived.values():
            args.update(derived.args)
        for reaction in self._reactions.values():
            args.update(reaction.args)
        for surrogate in self._surrogates.values():
            args.update(surrogate.args)

        return set(self._parameters).difference(args)

    ##########################################################################
    # Variables
    ##########################################################################

    @property
    def variables(self) -> TableView:
        """Returns a copy of the variables dictionary.

        Examples:
            >>> model.variables
                {"x1": 1.0, "x2": 2.0}

        This method returns a copy of the internal dictionary that maps variable
        names to their corresponding float values.

        Returns:
            dict[str, float]: A copy of the variables dictionary.

        """
        index = list(self._variables.keys())
        data = []
        for name, el in self._variables.items():
            if isinstance(init := el.initial_value, InitialAssignment):
                value_str = _latex_view(
                    fn_to_sympy(
                        init.fn,
                        origin=name,
                        model_args=list_of_symbols(init.args),
                    )
                )
            else:
                value_str = str(init)
            data.append(
                {
                    "value": value_str,
                    "unit": _latex_view(unit) if (unit := el.unit) is not None else "",
                    # "source"
                }
            )
        return TableView(data=pd.DataFrame(data, index=index))

    def get_raw_variables(self, *, as_copy: bool = True) -> dict[str, Variable]:
        """Retrieve the initial conditions of the model.

        Examples:
            >>> model.get_initial_conditions()
                {"x1": 1.0, "x2": 2.0}

        Returns:
            initial_conditions: A dictionary where the keys are variable names and the values are their initial conditions.

        """
        if as_copy:
            return copy.deepcopy(self._variables)
        return self._variables

    def get_initial_conditions(self) -> dict[str, float]:
        """Retrieve the initial conditions of the model.

        Examples:
            >>> model.get_initial_conditions()
                {"x1": 1.0, "x2": 2.0}

        Returns:
            initial_conditions: A dictionary where the keys are variable names and the values are their initial conditions.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.initial_conditions

    def get_variable_names(self) -> list[str]:
        """Retrieve the names of all variables.

        Examples:
            >>> model.get_variable_names()
                ["x1", "x2"]

        Returns:
            variable_names: A list containing the names of all variables.

        """
        return list(self._variables)

    @_invalidate_cache
    def add_variable(
        self,
        name: str,
        initial_value: float | InitialAssignment,
        unit: sympy.Expr | None = None,
        source: str | None = None,
    ) -> Self:
        """Adds a variable to the model with the given name and initial condition.

        Examples:
            >>> model.add_variable("x1", 1.0)

        Args:
            name: The name of the variable to add.
            initial_value: The initial condition value for the variable.
            unit: unit of the variable
            source: source of the information given

        Returns:
            Self: The instance of the model with the added variable.

        """
        self._insert_id(name=name, ctx="variable")
        self._variables[name] = Variable(
            initial_value=initial_value, unit=unit, source=source
        )
        return self

    def add_variables(
        self, variables: Mapping[str, float | Variable | InitialAssignment]
    ) -> Self:
        """Adds multiple variables to the model with their initial conditions.

        Examples:
            >>> model.add_variables({"x1": 1.0, "x2": 2.0})

        Args:
            variables: A dictionary where the keys are variable names (str)
                       and the values are their initial conditions (float).

        Returns:
            Self: The instance of the model with the added variables.

        """
        for name, v in variables.items():
            if isinstance(v, Variable):
                self.add_variable(
                    name=name,
                    initial_value=v.initial_value,
                    unit=v.unit,
                    source=v.source,
                )
            else:
                self.add_variable(name=name, initial_value=v)
        return self

    @_invalidate_cache
    def remove_variable(
        self,
        name: str,
        *,
        remove_stoichiometries: bool = True,
    ) -> Self:
        """Remove a variable from the model.

        Examples:
            >>> model.remove_variable("x1")

        Args:
            name: The name of the variable to remove.
            remove_stoichiometries: whether to remove the variable from all reactions

        Returns:
            Self: The instance of the model with the variable removed.

        """
        if remove_stoichiometries:
            for rxn in self._reactions.values():
                if name in rxn.stoichiometry:
                    cast(dict, rxn.stoichiometry).pop(name)
            for surrogate in self._surrogates.values():
                for stoich in surrogate.stoichiometries.values():
                    if name in stoich:
                        cast(dict, stoich).pop(name)

        self._remove_id(name=name)
        del self._variables[name]
        return self

    def remove_variables(
        self,
        variables: Iterable[str],
        *,
        remove_stoichiometries: bool = True,
    ) -> Self:
        """Remove multiple variables from the model.

        Examples:
            >>> model.remove_variables(["x1", "x2"])

        Args:
            variables: An iterable of variable names to be removed.
            remove_stoichiometries: whether to remove the variables from all reactions

        Returns:
            Self: The instance of the model with the specified variables removed.

        """
        for variable in variables:
            self.remove_variable(
                name=variable, remove_stoichiometries=remove_stoichiometries
            )
        return self

    @_invalidate_cache
    def update_variable(
        self,
        name: str,
        initial_value: float | InitialAssignment,
        unit: sympy.Expr | None = None,
        source: str | None = None,
    ) -> Self:
        """Updates the value of a variable in the model.

        Examples:
            >>> model.update_variable("x1", 2.0)

        Args:
            name: The name of the variable to update.
            initial_value: The initial condition or value to set for the variable.
            unit: Unit of the variable
            source: Source of the information

        Returns:
            Self: The instance of the model with the updated variable.

        """
        if name not in self._variables:
            msg = f"'{name}' not found in variables"
            raise KeyError(msg)

        variable = self._variables[name]

        if initial_value is not None:
            variable.initial_value = initial_value
        if unit is not None:
            variable.unit = unit
        if source is not None:
            variable.source = source
        return self

    def update_variables(
        self, variables: Mapping[str, float | Variable | InitialAssignment]
    ) -> Self:
        """Updates multiple variables in the model.

        Examples:
            >>> model.update_variables({"x1": 2.0, "x2": 3.0})

        Args:
            variables: A dictionary where the keys are variable names and the values are their new initial conditions.

        Returns:
            Self: The instance of the model with updated variables.

        """
        for k, v in variables.items():
            if isinstance(v, Variable):
                self.update_variable(
                    k,
                    initial_value=v.initial_value,
                    unit=v.unit,
                    source=v.source,
                )
            else:
                self.update_variable(k, v)
        return self

    def make_variable_static(self, name: str, value: float | None = None) -> Self:
        """Converts a variable to a static parameter.

        This removes the variable from the stoichiometries of all reactions and surrogates.
        It is not re-inserted if `Model.make_parameter_dynamic` is called.

        Examples:
            >>> model.make_variable_static("x1")
            >>> model.make_variable_static("x2", value=2.0)

        Args:
            name: The name of the variable to be made static.
            value: The value to assign to the parameter.
                   If None, the current value of the variable is used. Defaults to None.

        Returns:
            Self: The instance of the class for method chaining.

        """
        value_or_derived = (
            self._variables[name].initial_value if value is None else value
        )
        self.remove_variable(name, remove_stoichiometries=True)

        if isinstance(der := value_or_derived, Derived):
            self.add_derived(
                name,
                der.fn,
                args=der.args,
                unit=der.unit,
            )
        else:
            self.add_parameter(name, value_or_derived)
        return self

    ##########################################################################
    # Derived
    ##########################################################################

    @property
    def derived(self) -> TableView:
        """Returns a view of the derived quantities.

        Examples:
            >>> model.derived
                {"d1": Derived(fn1, ["x1", "x2"]),
                 "d2": Derived(fn2, ["x1", "d1"])}

        Returns:
            dict[str, Derived]: A copy of the derived dictionary.

        """
        index = list(self._derived.keys())
        data = [
            {
                "value": _latex_view(
                    fn_to_sympy(
                        el.fn,
                        origin=name,
                        model_args=list_of_symbols(el.args),
                    )
                ),
                "unit": _latex_view(unit) if (unit := el.unit) is not None else "",
            }
            for name, el in self._derived.items()
        ]

        return TableView(data=pd.DataFrame(data, index=index))

    def get_raw_derived(self, *, as_copy: bool = True) -> dict[str, Derived]:
        """Get copy of derived values."""
        if as_copy:
            return copy.deepcopy(self._derived)
        return self._derived

    def get_derived_variables(self) -> dict[str, Derived]:
        """Returns a dictionary of derived variables.

        Examples:
            >>> model.derived_variables()
                {"d1": Derived(fn1, ["x1", "x2"]),
                 "d2": Derived(fn2, ["x1", "d1"])}

        Returns:
            derived_variables: A dictionary where the keys are strings
            representing the names of the derived variables and the values are
            instances of DerivedVariable.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        derived = self._derived

        return {k: v for k, v in derived.items() if k not in cache.all_parameter_values}

    def get_derived_parameters(self) -> dict[str, Derived]:
        """Returns a dictionary of derived parameters.

        Examples:
            >>> model.derived_parameters()
                {"kd1": Derived(fn1, ["k1", "k2"]),
                 "kd2": Derived(fn2, ["k1", "kd1"])}

        Returns:
            A dictionary where the keys are
            parameter names and the values are Derived.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        derived = self._derived
        return {k: v for k, v in derived.items() if k in cache.all_parameter_values}

    @_invalidate_cache
    def add_derived(
        self,
        name: str,
        fn: RateFn,
        *,
        args: list[str],
        unit: sympy.Expr | None = None,
    ) -> Self:
        """Adds a derived attribute to the model.

        Examples:
            >>> model.add_derived("d1", add, args=["x1", "x2"])

        Args:
            name: The name of the derived attribute.
            fn: The function used to compute the derived attribute.
            args: The list of arguments to be passed to the function.
            unit: Unit of the derived value

        Returns:
            Self: The instance of the model with the added derived attribute.

        """
        self._insert_id(name=name, ctx="derived")
        self._derived[name] = Derived(fn=fn, args=args, unit=unit)
        return self

    def get_derived_parameter_names(self) -> list[str]:
        """Retrieve the names of derived parameters.

        Examples:
            >>> model.get_derived_parameter_names()
                ["kd1", "kd2"]

        Returns:
            A list of names of the derived parameters.

        """
        return list(self.get_derived_parameters())

    def get_derived_variable_names(self) -> list[str]:
        """Retrieve the names of derived variables.

        Examples:
            >>> model.get_derived_variable_names()
                ["d1", "d2"]

        Returns:
            A list of names of derived variables.

        """
        return list(self.get_derived_variables())

    @_invalidate_cache
    def update_derived(
        self,
        name: str,
        fn: RateFn | None = None,
        *,
        args: list[str] | None = None,
        unit: sympy.Expr | None = None,
    ) -> Self:
        """Updates the derived function and its arguments for a given name.

        Examples:
            >>> model.update_derived("d1", add, ["x1", "x2"])

        Args:
            name: The name of the derived function to update.
            fn: The new derived function. If None, the existing function is retained.
            args: The new arguments for the derived function. If None, the existing arguments are retained.
            unit: Unit of the derived value

        Returns:
            Self: The instance of the class with the updated derived function and arguments.

        """
        der = self._derived[name]
        if fn is not None:
            der.fn = fn
        if args is not None:
            der.args = args
        if unit is not None:
            der.unit = unit
        return self

    @_invalidate_cache
    def remove_derived(self, name: str) -> Self:
        """Remove a derived attribute from the model.

        Examples:
            >>> model.remove_derived("d1")

        Args:
            name: The name of the derived attribute to remove.

        Returns:
            Self: The instance of the model with the derived attribute removed.

        """
        self._remove_id(name=name)
        self._derived.pop(name)
        return self

    ###########################################################################
    # Reactions
    ###########################################################################

    @property
    def reactions(self) -> TableView:
        """Get view of reactions."""
        index = list(self._reactions.keys())
        data = [
            {
                "value": _latex_view(
                    fn_to_sympy(
                        rxn.fn,
                        origin=name,
                        model_args=list_of_symbols(rxn.args),
                    )
                ),
                "stoichiometry": stoichiometries_to_sympy(name, rxn.stoichiometry),
                "unit": _latex_view(unit) if (unit := rxn.unit) is not None else "",
                # "source"
            }
            for name, rxn in self._reactions.items()
        ]
        return TableView(data=pd.DataFrame(data, index=index))

    def get_raw_reactions(self, *, as_copy: bool = True) -> dict[str, Reaction]:
        """Retrieve the reactions in the model.

        Examples:
            >>> model.reactions
                {"r1": Reaction(fn1, {"x1": -1, "x2": 1}, ["k1"]),

        Returns:
            dict[str, Reaction]: A deep copy of the reactions dictionary.

        """
        if as_copy:
            return copy.deepcopy(self._reactions)
        return self._reactions

    def get_stoichiometries(
        self, variables: dict[str, float] | None = None, time: float = 0.0
    ) -> pd.DataFrame:
        """Retrieve the stoichiometries of the model.

        Examples:
            >>> model.stoichiometries()
                v1  v2
            x1 -1   1
            x2  1  -1

        Returns:
            pd.DataFrame: A DataFrame containing the stoichiometries of the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        args = self.get_args(variables=variables, time=time)

        stoich_by_cpds = copy.deepcopy(cache.stoich_by_cpds)
        for cpd, stoich in cache.dyn_stoich_by_cpds.items():
            for rxn, derived in stoich.items():
                stoich_by_cpds[cpd][rxn] = float(
                    derived.fn(*(args[i] for i in derived.args))
                )
        return pd.DataFrame(stoich_by_cpds).T.fillna(0)

    def get_stoichiometries_of_variable(
        self,
        variable: str,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> dict[str, float]:
        """Retrieve the stoichiometry of a specific variable.

        Examples:
            >>> model.get_stoichiometries_of_variable("x1")
                {"v1": -1, "v2": 1}

        Args:
            variable: The name of the variable for which to retrieve the stoichiometry.
            variables: A dictionary of variable names and their values.
            time: The time point at which to evaluate the stoichiometry.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        args = self.get_args(variables=variables, time=time)

        stoich = copy.deepcopy(cache.stoich_by_cpds[variable])
        for rxn, derived in cache.dyn_stoich_by_cpds.get(variable, {}).items():
            stoich[rxn] = float(derived.fn(*(args[i] for i in derived.args)))
        return stoich

    def get_raw_stoichiometries_of_variable(
        self, variable: str
    ) -> dict[str, float | Derived]:
        """Retrieve the raw stoichiometry of a specific variable.

        Examples:
            >>> model.get_stoichiometries_of_variable("x1")
                {"v1": -1, "v2": Derived(...)}

        Args:
            variable: The name of the variable for which to retrieve the stoichiometry.

        """
        stoichs: dict[str, dict[str, float | Derived]] = {}
        for rxn_name, rxn in self._reactions.items():
            for cpd_name, factor in rxn.stoichiometry.items():
                stoichs.setdefault(cpd_name, {})[rxn_name] = factor
        return stoichs[variable]

    @_invalidate_cache
    def add_reaction(
        self,
        name: str,
        fn: RateFn,
        *,
        args: list[str],
        stoichiometry: Mapping[str, float | str | Derived],
        unit: sympy.Expr | None = None,
        # source: str | None = None,
    ) -> Self:
        """Adds a reaction to the model.

        Examples:
            >>> model.add_reaction("v1",
            ...     fn=mass_action,
            ...     args=["x1", "kf1"],
            ...     stoichiometry={"x1": -1, "x2": 1},
            ... )

        Args:
            name: The name of the reaction.
            fn: The function representing the reaction.
            args: A list of arguments for the reaction function.
            stoichiometry: The stoichiometry of the reaction, mapping species to their coefficients.
            unit: Unit of the rate

        Returns:
            Self: The instance of the model with the added reaction.

        """
        self._insert_id(name=name, ctx="reaction")

        stoich: dict[str, Derived | float] = {
            k: Derived(fn=fns.constant, args=[v]) if isinstance(v, str) else v
            for k, v in stoichiometry.items()
        }
        self._reactions[name] = Reaction(
            fn=fn,
            stoichiometry=stoich,
            args=args,
            unit=unit,
        )
        return self

    def get_reaction_names(self) -> list[str]:
        """Retrieve the names of all reactions.

        Examples:
            >>> model.get_reaction_names()
                ["v1", "v2"]

        Returns:
            list[str]: A list containing the names of the reactions.

        """
        return list(self._reactions)

    @_invalidate_cache
    def update_reaction(
        self,
        name: str,
        fn: RateFn | None = None,
        *,
        args: list[str] | None = None,
        stoichiometry: Mapping[str, float | Derived | str] | None = None,
        unit: sympy.Expr | None = None,
    ) -> Self:
        """Updates the properties of an existing reaction in the model.

        Examples:
            >>> model.update_reaction("v1",
            ...     fn=mass_action,
            ...     args=["x1", "kf1"],
            ...     stoichiometry={"x1": -1, "x2": 1},
            ... )

        Args:
            name: The name of the reaction to update.
            fn: The new function for the reaction. If None, the existing function is retained.
            args: The new arguments for the reaction. If None, the existing arguments are retained.
            stoichiometry: The new stoichiometry for the reaction. If None, the existing stoichiometry is retained.
            unit: Unit of the reaction

        Returns:
            Self: The instance of the model with the updated reaction.

        """
        rxn = self._reactions[name]
        rxn.fn = rxn.fn if fn is None else fn

        if stoichiometry is not None:
            stoich = {
                k: Derived(fn=fns.constant, args=[v]) if isinstance(v, str) else v
                for k, v in stoichiometry.items()
            }
            rxn.stoichiometry = stoich
        rxn.args = rxn.args if args is None else args
        rxn.unit = rxn.unit if unit is None else unit
        return self

    @_invalidate_cache
    def remove_reaction(self, name: str) -> Self:
        """Remove a reaction from the model by its name.

        Examples:
            >>> model.remove_reaction("v1")

        Args:
            name: The name of the reaction to be removed.

        Returns:
            Self: The instance of the model with the reaction removed.

        """
        self._remove_id(name=name)
        self._reactions.pop(name)
        return self

    # def update_stoichiometry_of_cpd(
    #     self,
    #     rate_name: str,
    #     compound: str,
    #     value: float,
    # ) -> Model:
    #     self.update_stoichiometry(
    #         rate_name=rate_name,
    #         stoichiometry=self.stoichiometries[rate_name] | {compound: value},
    #     )
    #     return self

    # def scale_stoichiometry_of_cpd(
    #     self,
    #     rate_name: str,
    #     compound: str,
    #     scale: float,
    # ) -> Model:
    #     return self.update_stoichiometry_of_cpd(
    #         rate_name=rate_name,
    #         compound=compound,
    #         value=self.stoichiometries[rate_name][compound] * scale,
    #     )

    ##########################################################################
    # Readouts
    # They are like derived variables, but only calculated on demand, e.g. after
    # a simulation
    # Think of something like NADPH / (NADP + NADPH) as a proxy for energy state
    ##########################################################################

    def add_readout(
        self,
        name: str,
        fn: RateFn,
        *,
        args: list[str],
        unit: sympy.Expr | None = None,
    ) -> Self:
        """Adds a readout to the model.

        Examples:
            >>> model.add_readout("energy_state",
            ...     fn=div,
            ...     args=["NADPH", "NADP*_total"]
            ... )

        Args:
            name: The name of the readout.
            fn: The function to be used for the readout.
            args: The list of arguments for the function.
            unit: Unit of the readout

        Returns:
            Self: The instance of the model with the added readout.

        """
        self._insert_id(name=name, ctx="readout")
        self._readouts[name] = Readout(fn=fn, args=args, unit=unit)
        return self

    def get_readout_names(self) -> list[str]:
        """Retrieve the names of all readouts.

        Examples:
            >>> model.get_readout_names()
                ["energy_state", "redox_state"]

        Returns:
            list[str]: A list containing the names of the readouts.

        """
        return list(self._readouts)

    def get_raw_readouts(self, *, as_copy: bool = True) -> dict[str, Readout]:
        """Get copy of readouts in the model."""
        if as_copy:
            return copy.deepcopy(self._readouts)
        return self._readouts

    def remove_readout(self, name: str) -> Self:
        """Remove a readout by its name.

        Examples:
            >>> model.remove_readout("energy_state")

        Args:
            name (str): The name of the readout to remove.

        Returns:
            Self: The instance of the class after the readout has been removed.

        """
        self._remove_id(name=name)
        del self._readouts[name]
        return self

    ##########################################################################
    # Surrogates
    ##########################################################################

    @_invalidate_cache
    def add_surrogate(
        self,
        name: str,
        surrogate: AbstractSurrogate,
        args: list[str] | None = None,
        outputs: list[str] | None = None,
        stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
    ) -> Self:
        """Adds a surrogate model to the current instance.

        Examples:
            >>> model.add_surrogate("name", surrogate)

        Args:
            name (str): The name of the surrogate model.
            surrogate (AbstractSurrogate): The surrogate model instance to be added.
            args: Names of the values passed for the surrogate model.
            outputs: Names of values produced by the surrogate model.
            stoichiometries: A dictionary mapping reaction names to stoichiometries.

        Returns:
            Self: The current instance with the added surrogate model.

        """
        self._insert_id(name=name, ctx="surrogate")

        if args is not None:
            surrogate.args = args
        if outputs is not None:
            surrogate.outputs = outputs
        if stoichiometries is not None:
            surrogate.stoichiometries = stoichiometries

        self._surrogates[name] = surrogate
        return self

    def update_surrogate(
        self,
        name: str,
        surrogate: AbstractSurrogate | None = None,
        args: list[str] | None = None,
        stoichiometries: dict[str, dict[str, float | Derived]] | None = None,
    ) -> Self:
        """Update a surrogate model in the model.

        Examples:
            >>> model.update_surrogate("name", surrogate)

        Args:
            name (str): The name of the surrogate model to update.
            surrogate (AbstractSurrogate): The new surrogate model instance.
            args: A list of arguments for the surrogate model.
            stoichiometries: A dictionary mapping reaction names to stoichiometries.

        Returns:
            Self: The instance of the model with the updated surrogate model.

        """
        if name not in self._surrogates:
            msg = f"Surrogate '{name}' not found in model"
            raise KeyError(msg)

        if surrogate is None:
            surrogate = self._surrogates[name]
        if args is not None:
            surrogate.args = args
        if stoichiometries is not None:
            surrogate.stoichiometries = stoichiometries

        self._surrogates[name] = surrogate
        return self

    def remove_surrogate(self, name: str) -> Self:
        """Remove a surrogate model from the model.

        Examples:
            >>> model.remove_surrogate("name")

        Returns:
            Self: The instance of the model with the specified surrogate model removed.

        """
        self._remove_id(name=name)
        self._surrogates.pop(name)
        return self

    def get_raw_surrogates(
        self, *, as_copy: bool = True
    ) -> dict[str, AbstractSurrogate]:
        """Get direct copies of model surrogates."""
        if as_copy:
            return copy.deepcopy(self._surrogates)
        return self._surrogates

    def get_surrogate_output_names(
        self,
        *,
        include_fluxes: bool = True,
    ) -> list[str]:
        """Return output names by surrogates.

        Optionally filter out the names of which surrogate outfluxes are actually
        fluxes / reactions rather than variables.

        Args:
            include_fluxes: whether to also include outputs which are reaction
                            names

        """
        names = []
        if include_fluxes:
            for i in self._surrogates.values():
                names.extend(i.outputs)
        else:
            for i in self._surrogates.values():
                names.extend(x for x in i.outputs if x not in i.stoichiometries)
        return names

    def get_surrogate_reaction_names(self) -> list[str]:
        """Return reaction names by surrogates."""
        names = []
        for i in self._surrogates.values():
            names.extend(i.stoichiometries)
        return names

    ##########################################################################
    # Datasets
    ##########################################################################

    def add_data(self, name: str, data: pd.Series | pd.DataFrame) -> Self:
        """Add named data set to model."""
        self._insert_id(name=name, ctx="data")
        self._data[name] = data
        return self

    def update_data(self, name: str, data: pd.Series | pd.DataFrame) -> Self:
        """Update named data set."""
        self._data[name] = data
        return self

    def remove_data(self, name: str) -> Self:
        """Remove data set from model."""
        self._remove_id(name=name)
        self._data.pop(name)
        return self

    ##########################################################################
    # Get dependent values. This includes
    # - derived parameters
    # - derived variables
    # - fluxes
    # - readouts
    ##########################################################################

    def get_arg_names(
        self,
        *,
        include_time: bool,
        include_variables: bool,
        include_parameters: bool,
        include_derived_parameters: bool,
        include_derived_variables: bool,
        include_reactions: bool,
        include_surrogate_variables: bool,
        include_surrogate_fluxes: bool,
        include_readouts: bool,
    ) -> list[str]:
        """Get names of all kinds of model components."""
        names = []
        if include_time:
            names.append("time")
        if include_variables:
            names.extend(self.get_variable_names())
        if include_parameters:
            names.extend(self.get_parameter_names())
        if include_derived_variables:
            names.extend(self.get_derived_variable_names())
        if include_derived_parameters:
            names.extend(self.get_derived_parameter_names())
        if include_reactions:
            names.extend(self.get_reaction_names())
        if include_surrogate_variables:
            names.extend(self.get_surrogate_output_names(include_fluxes=False))
        if include_surrogate_fluxes:
            names.extend(self.get_surrogate_reaction_names())
        if include_readouts:
            names.extend(self.get_readout_names())
        return names

    def _get_args(
        self,
        variables: dict[str, float],
        time: float = 0.0,
        *,
        cache: ModelCache,
    ) -> dict[str, float]:
        """Generate a dictionary of model components dependent on other components.

        Examples:
            >>> model._get_args({"x1": 1.0, "x2": 2.0}, time=0.0)
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 0.0}

        Args:
            variables: A dictionary of concentrations with keys as the names of the substances
                   and values as their respective concentrations.
            time: The time point for the calculation
            cache: A ModelCache object containing precomputed values and dependencies.
            include_readouts: A flag indicating whether to include readout values in the returned dictionary.

        Returns:
            dict[str, float]
                A dictionary containing parameter values, derived variables, and optionally readouts,
                with their respective names as keys and their calculated values as values.

        """
        args = cache.all_parameter_values | variables | self._data
        args["time"] = time

        containers = self._derived | self._reactions | self._surrogates
        for name in cache.dyn_order:
            containers[name].calculate_inpl(name, args)

        for k in self._data:
            args.pop(k)

        return cast(dict[str, float], args)

    def get_args(
        self,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
        *,
        include_time: bool = True,
        include_variables: bool = True,
        include_parameters: bool = True,
        include_derived_parameters: bool = True,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = True,
        include_surrogate_fluxes: bool = True,
        include_readouts: bool = False,
    ) -> pd.Series:
        """Generate a pandas Series of arguments for the model.

        Examples:
            # Using initial conditions
            >>> model.get_args()
                {"x1": 1.get_args, "x2": 2.0, "k1": 0.1, "time": 0.0}

            # With custom concentrations
            >>> model.get_args({"x1": 1.0, "x2": 2.0})
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 0.0}

            # With custom concentrations and time
            >>> model.get_args({"x1": 1.0, "x2": 2.0}, time=1.0)
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 1.0}

        Args:
            variables: A dictionary where keys are the names of the concentrations and values are their respective float values.
            time: The time point at which the arguments are generated.
            include_time: Whether to include the time as an argument
            include_variables: Whether to include variables
            include_parameters: Whether to include parameters
            include_derived_parameters: Whether to include derived parameters
            include_derived_variables: Whether to include derived variables
            include_reactions: Whether to include reactions
            include_surrogate_variables: Whether to include derive variables obtained from surrogate
            include_surrogate_fluxes: Whether to include surrogate fluxes
            include_readouts: Whether to include readouts

        Returns:
            A pandas Series containing the generated arguments with float dtype.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        raw = self._get_args(
            variables=self.get_initial_conditions() if variables is None else variables,
            time=time,
            cache=cache,
        )
        if include_readouts:
            for name, ro in self._readouts.items():  # FIXME: order?
                ro.calculate_inpl(name, raw)
        args = pd.Series(raw, dtype=float)
        return args.loc[
            self.get_arg_names(
                include_time=include_time,
                include_variables=include_variables,
                include_parameters=include_parameters,
                include_derived_parameters=include_derived_parameters,
                include_derived_variables=include_derived_variables,
                include_reactions=include_reactions,
                include_surrogate_variables=include_surrogate_variables,
                include_surrogate_fluxes=include_surrogate_fluxes,
                include_readouts=include_readouts,
            )
        ]

    def _get_args_time_course(
        self,
        *,
        variables: pd.DataFrame,
        include_readouts: bool,
    ) -> dict[float, dict[str, float]]:
        if (cache := self._cache) is None:
            cache = self._create_cache()

        args_by_time = {}
        for time, values in variables.iterrows():
            args = self._get_args(
                variables=values.to_dict(),
                time=cast(float, time),
                cache=cache,
            )
            if include_readouts:
                for name, ro in self._readouts.items():  # FIXME: order?
                    ro.calculate_inpl(name, args)
            args_by_time[time] = args
        return args_by_time

    def get_args_time_course(
        self,
        variables: pd.DataFrame,
        *,
        include_variables: bool = True,
        include_parameters: bool = True,
        include_derived_parameters: bool = True,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = True,
        include_surrogate_fluxes: bool = True,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Generate a DataFrame containing time course arguments for model evaluation.

        Examples:
            >>> model.get_args_time_course(
            ...     pd.DataFrame({"x1": [1.0, 2.0], "x2": [2.0, 3.0]}
            ... )
                pd.DataFrame({
                    "x1": [1.0, 2.0],
                    "x2": [2.0, 3.0],
                    "k1": [0.1, 0.1],
                    "time": [0.0, 1.0]},
                )

        Args:
            variables: A DataFrame containing concentration data with time as the index.
            include_variables: Whether to include variables
            include_parameters: Whether to include parameters
            include_derived_parameters: Whether to include derived parameters
            include_derived_variables: Whether to include derived variables
            include_reactions: Whether to include reactions
            include_surrogate_variables: Whether to include variables derived from surrogates
            include_surrogate_fluxes: Whether to include surrogate fluxes
            include_readouts: Whether to include readouts

        Returns:
            A DataFrame containing the combined concentration data, parameter values,
            derived variables, and optionally readout variables, with time as an additional column.

        """
        args = pd.DataFrame(
            self._get_args_time_course(
                variables=variables,
                include_readouts=include_readouts,
            ),
            dtype=float,
        ).T

        return args.loc[
            :,
            self.get_arg_names(
                include_time=False,
                include_variables=include_variables,
                include_parameters=include_parameters,
                include_derived_parameters=include_derived_parameters,
                include_derived_variables=include_derived_variables,
                include_reactions=include_reactions,
                include_surrogate_variables=include_surrogate_variables,
                include_surrogate_fluxes=include_surrogate_fluxes,
                include_readouts=include_readouts,
            ),
        ]

    ##########################################################################
    # Get fluxes
    ##########################################################################

    def get_fluxes(
        self,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> pd.Series:
        """Calculate the fluxes for the given concentrations and time.

        Examples:
            # Using initial conditions as default
            >>> model.get_fluxes()
                pd.Series({"r1": 0.1, "r2": 0.2})

            # Using custom concentrations
            >>> model.get_fluxes({"x1": 1.0, "x2": 2.0})
                pd.Series({"r1": 0.1, "r2": 0.2})

            # Using custom concentrations and time
            >>> model.get_fluxes({"x1": 1.0, "x2": 2.0}, time=0.0)
                pd.Series({"r1": 0.1, "r2": 0.2})

        Args:
            variables: A dictionary where keys are species names and values are their concentrations.
            time: The time at which to calculate the fluxes. Defaults to 0.0.

        Returns:
            Fluxes: A pandas Series containing the fluxes for each reaction.

        """
        return self.get_args(
            variables=variables,
            time=time,
            include_time=False,
            include_variables=False,
            include_parameters=False,
            include_derived_parameters=False,
            include_derived_variables=False,
            include_reactions=True,
            include_surrogate_variables=False,
            include_surrogate_fluxes=True,
            include_readouts=False,
        )

    def get_fluxes_time_course(self, variables: pd.DataFrame) -> pd.DataFrame:
        """Generate a time course of fluxes for the given reactions and surrogates.

        Examples:
            >>> model.get_fluxes_time_course(args)
                pd.DataFrame({"v1": [0.1, 0.2], "v2": [0.2, 0.3]})

        This method calculates the fluxes for each reaction in the model using the provided
        arguments and combines them wit      names: list[str] = self.get_reaction_names()
        for surrogate in self._surrogates.values():
            names.extend(surrogate.stoichiometries)h the outputs from the surrogates to create a complete
        time course of fluxes.

        Args:
            variables: A DataFrame containing the input arguments for the reactions
                       and surrogates. Each column corresponds to a specific input
                       variable, and each row represents a different time point.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated fluxes for each reaction and
                          the outputs from the surrogates. The index of the DataFrame matches
                          the index of the input arguments.

        """
        return self.get_args_time_course(
            variables=variables,
            include_variables=False,
            include_parameters=False,
            include_derived_parameters=False,
            include_derived_variables=False,
            include_reactions=True,
            include_surrogate_variables=False,
            include_surrogate_fluxes=True,
            include_readouts=False,
        )

    ##########################################################################
    # Get rhs
    ##########################################################################

    def __call__(self, /, time: float, variables: Iterable[float]) -> tuple[float, ...]:
        """Simulation version of get_right_hand_side.

        Examples:
            >>> model(0.0, np.array([1.0, 2.0]))
                np.array([0.1, 0.2])

        Warning: Swaps t and y!
        This can't get kw args, as the integrators call it with pos-only

        Args:
            time: The current time point.
            variables: Array of concentrations


        Returns:
            The rate of change of each variable in the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        vars_d: dict[str, float] = dict(
            zip(
                cache.var_names,
                variables,
                strict=True,
            )
        )
        dependent: dict[str, float] = self._get_args(
            variables=vars_d,
            time=time,
            cache=cache,
        )

        dxdt = dict.fromkeys(cache.var_names, 0.0)
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * dependent[flux]
        for k, sd in cache.dyn_stoich_by_cpds.items():
            for flux, dv in sd.items():
                n = dv.calculate(dependent)
                dxdt[k] += n * dependent[flux]
        return tuple(dxdt[i] for i in cache.var_names)

    def _get_right_hand_side(
        self,
        *,
        args: dict[str, float],
        var_names: list[str],
        cache: ModelCache,
    ) -> pd.Series:
        dxdt = pd.Series(np.zeros(len(var_names), dtype=float), index=var_names)
        for k, stoc in cache.stoich_by_cpds.items():
            for flux, n in stoc.items():
                dxdt[k] += n * args[flux]

        for k, sd in cache.dyn_stoich_by_cpds.items():
            for flux, dv in sd.items():
                n = dv.fn(*(args[i] for i in dv.args))
                dxdt[k] += n * args[flux]
        return dxdt

    def get_right_hand_side(
        self,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> pd.Series:
        """Calculate the right-hand side of the differential equations for the model.

        Examples:
            # Using initial conditions as default
            >>> model.get_right_hand_side()
                pd.Series({"x1": 0.1, "x2": 0.2})

            # Using custom concentrations
            >>> model.get_right_hand_side({"x1": 1.0, "x2": 2.0})
                pd.Series({"x1": 0.1, "x2": 0.2})

            # Using custom concentrations and time
            >>> model.get_right_hand_side({"x1": 1.0, "x2": 2.0}, time=0.0)
                pd.Series({"x1": 0.1, "x2": 0.2})

        Args:
            variables: A dictionary mapping compound names to their concentrations.
            time: The current time point. Defaults to 0.0.

        Returns:
            The rate of change of each variable in the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        var_names = self.get_variable_names()
        args = self._get_args(
            variables=self.get_initial_conditions() if variables is None else variables,
            time=time,
            cache=cache,
        )
        return self._get_right_hand_side(args=args, var_names=var_names, cache=cache)

    def get_right_hand_side_time_course(self, args: pd.DataFrame) -> pd.DataFrame:
        """Calculate the right-hand side of the differential equations for the model."""
        if (cache := self._cache) is None:
            cache = self._create_cache()
        var_names = self.get_variable_names()

        rhs_by_time = {}
        for time, variables in args.iterrows():
            rhs_by_time[time] = self._get_right_hand_side(
                args=variables.to_dict(),
                var_names=var_names,
                cache=cache,
            )
        return pd.DataFrame(rhs_by_time).T

    ##########################################################################
    # Check units
    ##########################################################################

    def check_units(self, time_unit: Quantity) -> UnitCheck:
        """Check unit consistency per differential equation and reaction."""
        units_per_fn = {}
        for name, rxn in self._reactions.items():
            unit_per_arg = {}
            for arg in rxn.args:
                if (par := self._parameters.get(arg)) is not None:
                    unit_per_arg[sympy.Symbol(arg)] = par.unit
                elif (var := self._variables.get(arg)) is not None:
                    unit_per_arg[sympy.Symbol(arg)] = var.unit
                else:
                    raise NotImplementedError

            symbolic_fn = fn_to_sympy(
                rxn.fn,
                origin="unit-checking",
                model_args=list_of_symbols(rxn.args),
            )
            units_per_fn[name] = None
            if symbolic_fn is None:
                continue
            if any(i is None for i in unit_per_arg.values()):
                continue
            units_per_fn[name] = symbolic_fn.subs(unit_per_arg)

        check_per_variable = {}
        for name, var in self._variables.items():
            check_per_rxn = {}

            if (var_unit := var.unit) is None:
                break

            for rxn in self.get_stoichiometries_of_variable(name):
                if (rxn_unit := units_per_fn.get(rxn)) is None:
                    check_per_rxn[rxn] = None
                elif unit_of(rxn_unit) == var_unit / time_unit:  # type: ignore
                    check_per_rxn[rxn] = True
                else:
                    check_per_rxn[rxn] = Failure(
                        expected=unit_of(rxn_unit),
                        obtained=var_unit / time_unit,  # type: ignore
                    )
            check_per_variable[name] = check_per_rxn
        return UnitCheck(check_per_variable)
