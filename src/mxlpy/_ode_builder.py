"""Model for Metabolic System Representation.

This module provides the core Model class and supporting functionality for representing
metabolic models, including diff_eqs, variables, parameters and derived quantities.

"""

from __future__ import annotations

import copy
import inspect
import itertools as it
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast

import pandas as pd
import sympy
from wadler_lindig import pformat

from mxlpy import _topo
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
)
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import (
    Annotation,
    Derived,
    DiffEq,
    InitialAssignment,
    Parameter,
    Readout,
)
from mxlpy.unit_inference import (
    MdText,
    _fmt_failed,
    _fmt_success,
    _latex_view,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mxlpy.types import Callable, Param, RateFn, RetType

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ArityMismatchError",
    "Failure",
    "LOGGER",
    "ModelCache",
    "OdeModelBuilder",
    "TableView",
    "UnitCheck",
]


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
class UnitCheck:
    """Container for unit check."""

    per_variable: dict[str, dict[str, bool | Failure | None]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def correct_diff_eqs(self) -> dict[str, bool]:
        """Get all correctly annotated diff_eqs by variable."""
        return {
            var: all(isinstance(i, bool) for i in checks.values())
            for var, checks in self.per_variable.items()
        }

    def report(self) -> MdText:
        """Export check as markdown report."""
        report = ["## Type check"]
        for diff_eq, res in self.correct_diff_eqs().items():
            txt = _fmt_success("Correct") if res else _fmt_failed("Failed")
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

    Parameters
    ----------
    method
        Method to wrap with cache invalidation

    Returns
    -------
        Wrapped method that clears cache before execution

    """

    def wrapper(
        *args: Param.args,
        **kwargs: Param.kwargs,
    ) -> RetType:
        self = cast(OdeModelBuilder, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


def _expr_free_symbol_names(expr: sympy.Expr) -> list[str]:
    """Return an expression's free symbol names in a deterministic order.

    `Expr.free_symbols` is a `set`, whose iteration order depends on
    Python's per-process string hash seed. Sorting keeps repeated calls
    for the same expression reproducible across processes.
    """
    return sorted(i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol))


def _expr_to_fn_and_args(expr: sympy.Expr) -> tuple[RateFn, list[str]]:
    """Convert a sympy expression into a callable and its own argument names.

    The expression's free symbol names are used *literally* as the
    argument names - they must exist as parameters/variables/derived
    quantities in the model. There is no separate `args` list to remap
    them through: a sympy expression has no declared parameter order to
    bind such a list against, so mxlpy does not offer one.
    """
    args = _expr_free_symbol_names(expr)
    return sympy.lambdify(args, expr), args


def _expr_as_initial_assignment[T](
    value: T | InitialAssignment | sympy.Expr,
) -> T | InitialAssignment:
    if isinstance(value, sympy.Expr):
        fn, args = _expr_to_fn_and_args(value)
        value = InitialAssignment(fn=fn, args=args)
    return value


@dataclass(slots=True)
class ModelCache:
    """ModelCache is a class that stores various model-related data structures.

    Attributes
    ----------
    var_names
        A list of variable names.
    parameter_values
        A dictionary mapping parameter names to their values.
    derived_parameters
        A dictionary mapping parameter names to their derived parameter objects.
    derived_variables
        A dictionary mapping variable names to their derived variable objects.
    stoich_by_cpds
        A dictionary mapping compound names to their stoichiometric coefficients.
    dyn_stoich_by_cpds
        A dictionary mapping compound names to their dynamic stoichiometric coefficients.
    dxdt
        A pandas Series representing the rate of change of variables.
    initial_conditions
        calculated initial conditions

    """

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    order: list[str]  # mostly for debug purposes
    var_names: list[str]
    dyn_order: list[str]
    base_parameter_values: dict[str, float]
    all_parameter_values: dict[str, float]
    initial_conditions: dict[str, float]


def _normalize_annotations(
    annotations: Annotation | Iterable[Annotation] | None,
) -> list[Annotation]:
    """Normalise an annotation argument into a list of annotations."""
    if annotations is None:
        return []
    if isinstance(annotations, Annotation):
        return [annotations]
    return list(annotations)


@dataclass(slots=True)
class OdeModelBuilder:
    """Represents a metabolic model.

    Attributes
    ----------
    _ids
        Dictionary mapping internal IDs to names.
    _variables
        Dictionary of model variables and their initial values.
    _parameters
        Dictionary of model parameters and their values.
    _derived
        Dictionary of derived quantities.
    _readouts
        Dictionary of readout functions.
    _diff_eqs
        Dictionary of diff_eqs in the model.
    _surrogates
        Dictionary of surrogate models.
    _cache
        Cache for storing model-related data structures.
    _data
        Named references to data sets

    """

    _ids: dict[str, str] = field(default_factory=dict, repr=False)
    _cache: ModelCache | None = field(default=None, repr=False)
    _diff_eqs: dict[str, DiffEq] = field(default_factory=dict)
    _parameters: dict[str, Parameter] = field(default_factory=dict)
    _derived: dict[str, Derived] = field(default_factory=dict)
    _readouts: dict[str, Readout] = field(default_factory=dict)
    _data: dict[str, pd.Series | pd.DataFrame] = field(default_factory=dict)
    _annotations: list[Annotation] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def annotate_model(self, annotations: Annotation | Iterable[Annotation]) -> Self:
        """Attach MIRIAM model-level annotations to the model.

        Examples
        --------
            >>> model.annotate_model(
            ...     Annotation(
            ...         uri="https://identifiers.org/biomodels.db:BIOMD0000000048",
            ...         predicate="isDerivedFrom",
            ...     )
            ... )

        Parameters
        ----------
        annotations
            A single annotation or an iterable of annotations describing the
            model. Use ``bqmodel`` qualifiers as the predicate.

        Returns
        -------
        Self
            The instance of the model with the added model-level annotations.

        """
        self._annotations.extend(_normalize_annotations(annotations))
        return self

    def get_annotations(self) -> list[Annotation]:
        """Return the model-level annotations.

        Returns
        -------
        list[Annotation]
            The model-level annotations.

        """
        return self._annotations

    ###########################################################################
    # Cache
    ###########################################################################

    def _create_cache(self) -> ModelCache:
        """Creates and initializes the model cache.

        This method constructs a cache that includes parameter values, stoichiometry
        by compounds, dynamic stoichiometry by compounds, derived variables, and
        derived parameters. It processes the model's parameters, variables, derived
        elements, diff_eqs, and surrogates to populate the cache.

        Returns
        -------
        ModelCache
            An instance of ModelCache containing the initialized cache data.

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
            for k, v in self._diff_eqs.items()
            if not isinstance(init := v.initial_value, InitialAssignment)
        }
        initial_assignments: dict[str, InitialAssignment] = {
            k: init
            for k, v in self._diff_eqs.items()
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
            self._readouts.items(),
        ):
            if not _check_function_arity(el.fn, len(el.args)):
                raise ArityMismatchError(name, el.fn, el.args)

        # Sort derived & diff_eqs
        available = (
            set(base_parameter_values)
            | set(base_variable_values)
            | set(self._data)
            | {"time"}
        )
        to_sort = initial_assignments | self._derived
        order = _topo.sort_dependencies(
            available=available,
            elements=[
                _topo.Dependency(name=k, required=set(v.args), provided={k})
                if not isinstance(v, AbstractSurrogate)
                else _topo.Dependency(
                    name=k, required=set(v.args), provided=set(v.outputs)
                )
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
            if name in self._diff_eqs:
                dyn_order.append(name)
            elif name in self._parameters:
                static_order.append(name)
            else:
                derived = self._derived[name]
                if all(i in all_parameter_names for i in derived.args):
                    static_order.append(name)
                    all_parameter_names.add(name)
                else:
                    dyn_order.append(name)

        var_names = self.get_diff_eq_names()
        initial_conditions: dict[str, float] = {
            k: cast(float, dependent[k]) for k in self._diff_eqs
        }
        all_parameter_values = dict(base_parameter_values)
        for name in static_order:
            if name in self._parameters or name in self._derived:
                all_parameter_values[name] = cast(float, dependent[name])
            else:
                msg = f"Internal error: '{name}' appears in static_order but is not a parameter, variable, or derived - this is a bug in dependency sorting"
                raise KeyError(msg)

        self._cache = ModelCache(
            order=order,
            var_names=var_names,
            dyn_order=dyn_order,
            base_parameter_values=base_parameter_values,
            all_parameter_values=all_parameter_values,
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

        Returns
        -------
        dict[str, str]
            A copy of the _ids dictionary.

        """
        return self._ids.copy()

    def _insert_id(self, *, name: str, ctx: str) -> None:
        """Inserts an identifier into the model's internal ID dictionary.

        Parameters
        ----------
        name
            The name of the identifier to insert.
        ctx
            The context associated with the identifier.

        Raises
        ------
        KeyError
            If the name is "time", which is a protected variable.
        NameError
            If the name already exists in the model's ID dictionary.

        """
        if name == "time":
            msg = "'time' is a reserved identifier - it represents the simulation time and cannot be used as a parameter, variable, derived, or diff_eq name"
            raise KeyError(msg)

        if name in self._ids:
            existing_ctx = self._ids[name]
            msg = f"Name '{name}' already exists as a {existing_ctx} - cannot add it as a {ctx}. Each name must be unique across all model components."
            raise NameError(msg)
        self._ids[name] = ctx

    def _remove_id(self, *, name: str) -> None:
        """Remove an ID from the internal dictionary.

        Parameters
        ----------
        name : str
            The name of the ID to be removed.

        Raises
        ------
        KeyError
            If the specified name does not exist in the dictionary.

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
                    fn_to_sympy_expr(
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
        """Returns the parameters of the model.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the parameters dictionary.
            If False, return the internal dictionary directly.

        Returns
        -------
        dict[str, Parameter]
            Dictionary mapping parameter names to Parameter objects.

        """
        if as_copy:
            return copy.deepcopy(self._parameters)
        return self._parameters

    def get_parameter_values(self) -> dict[str, float]:
        """Returns the parameters of the model.

        Examples
        --------
            >>> model.parameters
                {"k1": 0.1, "k2": 0.2}

        Returns
        -------
        parameters
            A dictionary where the keys are parameter names (as strings)
            and the values are parameter values (as floats).

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.base_parameter_values

    def get_parameter_names(self) -> list[str]:
        """Retrieve the names of the parameters.

        Examples
        --------
            >>> model.get_parameter_names()
                ['k1', 'k2']

        Returns
        -------
        parametes
            A list containing the names of the parameters.

        """
        return list(self._parameters)

    #####################################
    # Parameters - create
    #####################################

    @_invalidate_cache
    def add_parameter(
        self,
        name: str,
        value: float | InitialAssignment | sympy.Expr,
        unit: sympy.Expr | None = None,
        source: str | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Adds a parameter to the model.

        Examples
        --------
            >>> model.add_parameter("k1", 0.1)

        Parameters
        ----------
        name
            The name of the parameter.
        value
            The value of the parameter.
        unit
            unit of the parameter
        source
            source of the information given
        annotations
            MIRIAM annotation(s) for the parameter (bqbiol qualifiers).

        Returns
        -------
        Self
            The instance of the model with the added parameter.

        """
        self._insert_id(name=name, ctx="parameter")
        self._parameters[name] = Parameter(
            value=_expr_as_initial_assignment(value),
            unit=unit,
            source=source,
            annotations=_normalize_annotations(annotations),
        )
        return self

    def add_parameters(
        self,
        parameters: Mapping[str, float | Parameter | InitialAssignment | sympy.Expr],
    ) -> Self:
        """Adds multiple parameters to the model.

        Examples
        --------
            >>> model.add_parameters({"k1": 0.1, "k2": 0.2})

        Parameters
        ----------
        parameters : dict[str, float]
            A dictionary where the keys are parameter names
            and the values are the corresponding parameter values.

        Returns
        -------
        Self
            The instance of the model with the added parameters.

        """
        for k, v in parameters.items():
            if isinstance(v, Parameter):
                self.add_parameter(
                    k,
                    v.value,
                    unit=v.unit,
                    source=v.source,
                    annotations=v.annotations,
                )
            else:
                self.add_parameter(k, v)
        return self

    #####################################
    # Parameters - delete
    #####################################

    @_invalidate_cache
    def remove_parameter(self, name: str) -> Self:
        """Remove a parameter from the model.

        Examples
        --------
            >>> model.remove_parameter("k1")

        Parameters
        ----------
        name
            The name of the parameter to remove.

        Returns
        -------
        Self
            The instance of the model with the parameter removed.

        """
        self._remove_id(name=name)
        self._parameters.pop(name)
        return self

    def remove_parameters(self, names: list[str]) -> Self:
        """Remove multiple parameters from the model.

        Examples
        --------
            >>> model.remove_parameters(["k1", "k2"])

        Parameters
        ----------
        names
            A list of parameter names to be removed.

        Returns
        -------
        Self
            The instance of the model with the specified parameters removed.

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
        value: float | InitialAssignment | sympy.Expr | None = None,
        *,
        unit: sympy.Expr | None = None,
        source: str | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Update the value of a parameter.

        Examples
        --------
            >>> model.update_parameter("k1", 0.2)

        Parameters
        ----------
        name
            The name of the parameter to update.
        value
            The new value for the parameter.
        unit
            Unit of the parameter
        source
            Source of the information
        annotations
            MIRIAM annotation(s) to replace the parameter's annotations.

        Returns
        -------
        Self
            The instance of the class with the updated parameter.

        Raises
        ------
        NameError
            If the parameter name is not found in the parameters.

        """
        if name not in self._parameters:
            msg = f"Parameter {name!r} not found. Available parameters: {sorted(self._parameters)}"
            raise KeyError(msg)

        parameter = self._parameters[name]
        if value is not None:
            parameter.value = _expr_as_initial_assignment(value)
        if unit is not None:
            parameter.unit = unit
        if source is not None:
            parameter.source = source
        if annotations is not None:
            parameter.annotations = _normalize_annotations(annotations)
        return self

    def update_parameters(
        self,
        parameters: Mapping[str, float | Parameter | InitialAssignment | sympy.Expr],
    ) -> Self:
        """Update multiple parameters of the model.

        Examples
        --------
            >>> model.update_parameters({"k1": 0.2, "k2": 0.3})

        Parameters
        ----------
        parameters
            A dictionary where keys are parameter names and values are the new parameter values.

        Returns
        -------
        Self
            The instance of the model with updated parameters.

        """
        for k, v in parameters.items():
            if isinstance(v, Parameter):
                self.update_parameter(
                    k,
                    value=v.value,
                    unit=v.unit,
                    source=v.source,
                    annotations=v.annotations,
                )
            else:
                self.update_parameter(k, v)
        return self

    def scale_parameter(self, name: str, factor: float) -> Self:
        """Scales the value of a specified parameter by a given factor.

        Examples
        --------
            >>> model.scale_parameter("k1", 2.0)

        Parameters
        ----------
        name
            The name of the parameter to be scaled.
        factor
            The factor by which to scale the parameter's value.

        Returns
        -------
        Self
            The instance of the class with the updated parameter.

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

        Examples
        --------
            >>> model.scale_parameters({"k1": 2.0, "k2": 0.5})

        Parameters
        ----------
        parameters
            A dictionary where the keys are parameter names
            and the values are the scaling factors.

        Returns
        -------
        Self
            The instance of the model with scaled parameters.

        """
        for k, v in parameters.items():
            self.scale_parameter(k, v)
        return self

    @_invalidate_cache
    def make_parameter_dynamic(
        self,
        name: str,
        fn: RateFn,
        args: list[str],
        initial_value: float | InitialAssignment | sympy.Expr | None = None,
    ) -> Self:
        """Converts a parameter to a dynamic variable in the model.

        Examples
        --------
            >>> model.make_parameter_dynamic("k1")
            >>> model.make_parameter_dynamic("k2", initial_value=0.5)

        This method removes the specified parameter from the model and adds it as a variable with an optional initial value.

        Parameters
        ----------
        name
            The name of the parameter to be converted.
        initial_value
            The initial value for the new variable. If None, the current value of the parameter is used. Defaults to None.
        stoichiometries
            A dictionary mapping diff_eq names to stoichiometries for the new variable. Defaults to None.

        Returns
        -------
        Self
            The instance of the model with the parameter converted to a variable.

        """
        value = self._parameters[name].value if initial_value is None else initial_value
        self.remove_parameter(name)
        self.add_diff_eq(
            name,
            initial_value=value,
            fn=fn,
            args=args,
        )

        return self

    def get_unused_parameters(self) -> set[str]:
        """Get parameters which aren't used in the model."""
        args = set()
        for diff_eq in self._diff_eqs.values():
            args.update(diff_eq.args)
            if isinstance(diff_eq.initial_value, Derived):
                args.update(diff_eq.initial_value.args)
        for derived in self._derived.values():
            args.update(derived.args)

        return set(self._parameters).difference(args)

    ##########################################################################
    # Derived
    ##########################################################################

    @property
    def derived(self) -> TableView:
        """Returns a view of the derived quantities.

        Examples
        --------
            >>> model.derived
                {"d1": Derived(fn1, ["x1", "x2"]),
                 "d2": Derived(fn2, ["x1", "d1"])}

        Returns
        -------
        dict[str, Derived]
            A copy of the derived dictionary.

        """
        index = list(self._derived.keys())
        data = [
            {
                "value": _latex_view(
                    fn_to_sympy_expr(
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
        """Get copy of derived values.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the derived dictionary.
            If False, return the internal dictionary directly.

        Returns
        -------
        dict[str, Derived]
            Dictionary mapping derived names to Derived objects.

        """
        if as_copy:
            return copy.deepcopy(self._derived)
        return self._derived

    def get_derived_variables(self) -> dict[str, Derived]:
        """Returns a dictionary of derived variables.

        Examples
        --------
            >>> model.derived_variables()
                {"d1": Derived(fn1, ["x1", "x2"]),
                 "d2": Derived(fn2, ["x1", "d1"])}

        Returns
        -------
        derived_variables
            A dictionary where the keys are strings
            representing the names of the derived variables and the values are
            instances of DerivedVariable.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        derived = self._derived

        return {k: v for k, v in derived.items() if k not in cache.all_parameter_values}

    def get_derived_parameters(self) -> dict[str, Derived]:
        """Returns a dictionary of derived parameters.

        Examples
        --------
            >>> model.derived_parameters()
                {"kd1": Derived(fn1, ["k1", "k2"]),
                 "kd2": Derived(fn2, ["k1", "kd1"])}

        Returns
        -------
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
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Adds a derived attribute to the model.

        Examples
        --------
            >>> model.add_derived("d1", add, args=["x1", "x2"])

        Parameters
        ----------
        name
            The name of the derived attribute.
        fn
            The function used to compute the derived attribute.
        args
            The list of arguments to be passed to the function.
        unit
            Unit of the derived value
        annotations
            MIRIAM annotation(s) for the derived value (bqbiol qualifiers).

        Returns
        -------
        Self
            The instance of the model with the added derived attribute.

        """
        self._insert_id(name=name, ctx="derived")
        self._derived[name] = Derived(
            fn=fn,
            args=args,
            unit=unit,
            annotations=_normalize_annotations(annotations),
        )
        return self

    @_invalidate_cache
    def add_derived_from_expr(
        self,
        name: str,
        expr: sympy.Expr,
        *,
        unit: sympy.Expr | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Adds a derived attribute computed from a sympy expression.

        Unlike `add_derived`, there is no separate `args` list: the
        expression's free symbols are used *literally* as argument names,
        so they must already exist as parameters/variables/derived
        quantities in the model.

        Examples
        --------
            >>> k1, km = sympy.symbols("k1 km")
            >>> model.add_derived_from_expr("keq", k1 / km)

        Parameters
        ----------
        name
            The name of the derived attribute.
        expr
            A sympy expression whose free symbols name existing model
            components.
        unit
            Unit of the derived value
        annotations
            MIRIAM annotation(s) for the derived value (bqbiol qualifiers).

        Returns
        -------
        Self
            The instance of the model with the added derived attribute.

        """
        self._insert_id(name=name, ctx="derived")
        fn, args = _expr_to_fn_and_args(expr)
        self._derived[name] = Derived(
            fn=fn,
            args=args,
            unit=unit,
            annotations=_normalize_annotations(annotations),
        )
        return self

    def get_derived_parameter_names(self) -> list[str]:
        """Retrieve the names of derived parameters.

        Examples
        --------
            >>> model.get_derived_parameter_names()
                ["kd1", "kd2"]

        Returns
        -------
            A list of names of the derived parameters.

        """
        return list(self.get_derived_parameters())

    def get_derived_variable_names(self) -> list[str]:
        """Retrieve the names of derived variables.

        Examples
        --------
            >>> model.get_derived_variable_names()
                ["d1", "d2"]

        Returns
        -------
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
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Updates the derived function and its arguments for a given name.

        Examples
        --------
            >>> model.update_derived("d1", add, ["x1", "x2"])

        Parameters
        ----------
        name
            The name of the derived function to update.
        fn
            The new derived function. If None, the existing function is retained.
        args
            The new arguments for the derived function. If None, the existing arguments are retained.
        unit
            Unit of the derived value
        annotations
            MIRIAM annotation(s) to replace the derived value's annotations.

        Returns
        -------
        Self
            The instance of the class with the updated derived function and arguments.

        """
        der = self._derived[name]
        if fn is not None:
            der.fn = fn
        if args is not None:
            der.args = args
        if unit is not None:
            der.unit = unit
        if annotations is not None:
            der.annotations = _normalize_annotations(annotations)
        return self

    @_invalidate_cache
    def update_derived_from_expr(
        self,
        name: str,
        expr: sympy.Expr,
        *,
        unit: sympy.Expr | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Updates a derived attribute from a sympy expression.

        The expression's free symbols are used *literally* as argument
        names and fully replace the previous `args`, since there is no
        way to keep an old `args` list alongside a new expression.

        Parameters
        ----------
        name
            The name of the derived function to update.
        expr
            A sympy expression whose free symbols name existing model
            components.
        unit
            Unit of the derived value
        annotations
            MIRIAM annotation(s) to replace the derived value's annotations.

        Returns
        -------
        Self
            The instance of the class with the updated derived function and arguments.

        """
        der = self._derived[name]
        der.fn, der.args = _expr_to_fn_and_args(expr)
        if unit is not None:
            der.unit = unit
        if annotations is not None:
            der.annotations = _normalize_annotations(annotations)
        return self

    @_invalidate_cache
    def remove_derived(self, name: str) -> Self:
        """Remove a derived attribute from the model.

        Examples
        --------
            >>> model.remove_derived("d1")

        Parameters
        ----------
        name
            The name of the derived attribute to remove.

        Returns
        -------
        Self
            The instance of the model with the derived attribute removed.

        """
        self._remove_id(name=name)
        self._derived.pop(name)
        return self

    ###########################################################################
    # diff_eqs
    ###########################################################################

    def get_initial_conditions(self) -> dict[str, float]:
        """Retrieve the initial conditions of the model.

        Examples
        --------
            >>> model.get_initial_conditions()
                {"x1": 1.0, "x2": 2.0}

        Returns
        -------
        initial_conditions
            A dictionary where the keys are variable names and the values are their initial conditions.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        return cache.initial_conditions

    def make_diff_eq_static(
        self,
        name: str,
        value: float | InitialAssignment | sympy.Expr | None = None,
    ) -> Self:
        """Converts a variable to a static parameter.

        This removes the variable from the stoichiometries of all diff_eqs and surrogates.
        It is not re-inserted if `Model.make_parameter_dynamic` is called.

        Examples
        --------
            >>> model.make_variable_static("x1")
            >>> model.make_variable_static("x2", value=2.0)

        Parameters
        ----------
        name
            The name of the variable to be made static.
        value
            The value to assign to the parameter.
            If None, the current value of the variable is used. Defaults to None.

        Returns
        -------
        Self
            The instance of the class for method chaining.

        """
        value_or_derived = (
            self._diff_eqs[name].initial_value
            if value is None
            else _expr_as_initial_assignment(value)
        )
        self.remove_diff_eq(name)

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

    @property
    def diff_eqs(self) -> TableView:
        """Get view of diff_eqs."""
        index = list(self._diff_eqs.keys())
        data = [
            {
                "value": _latex_view(
                    fn_to_sympy_expr(
                        rxn.fn,
                        origin=name,
                        model_args=list_of_symbols(rxn.args),
                    )
                ),
                "initial": _latex_view(
                    fn_to_sympy_expr(
                        init.fn,
                        origin=name,
                        model_args=list_of_symbols(init.args),
                    )
                )
                if isinstance(init := rxn.initial_value, InitialAssignment)
                else str(init),
                "unit": _latex_view(unit) if (unit := rxn.unit) is not None else "",
            }
            for name, rxn in self._diff_eqs.items()
        ]
        return TableView(data=pd.DataFrame(data, index=index))

    def get_raw_diff_eqs(self, *, as_copy: bool = True) -> dict[str, DiffEq]:
        """Retrieve the diff_eqs in the model.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the diff_eqs dictionary.
            If False, return the internal dictionary directly.

        Examples
        --------
            >>> model.diff_eqs
                {"r1": diff_eq(fn1, {"x1": -1, "x2": 1}, ["k1"]),

        Returns
        -------
        dict[str, diff_eq]
            A deep copy of the diff_eqs dictionary.

        """
        if as_copy:
            return copy.deepcopy(self._diff_eqs)
        return self._diff_eqs

    @_invalidate_cache
    def add_diff_eq(
        self,
        name: str,
        initial_value: float | InitialAssignment | sympy.Expr,
        fn: RateFn,
        *,
        args: list[str],
        unit: sympy.Expr | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
        # source: str | None = None,
    ) -> Self:
        """Adds a diff_eq to the model.

        Examples
        --------
            >>> model.add_diff_eq("v1",
            ...     fn=mass_action,
            ...     args=["x1", "kf1"],
            ...     stoichiometry={"x1": -1, "x2": 1},
            ... )

        Parameters
        ----------
        name
            The name of the diff_eq.
        fn
            The function representing the diff_eq.
        args
            A list of arguments for the diff_eq function.
        stoichiometry
            The stoichiometry of the diff_eq, mapping variables to their coefficients.
        unit
            Unit of the rate
        annotations
            MIRIAM annotation(s) for the diff_eq (bqbiol qualifiers).

        Returns
        -------
        Self
            The instance of the model with the added diff_eq.

        """
        self._insert_id(name=name, ctx="diff_eq")
        self._diff_eqs[name] = DiffEq(
            fn=fn,
            initial_value=_expr_as_initial_assignment(initial_value),
            args=args,
            unit=unit,
            annotations=_normalize_annotations(annotations),
        )
        return self

    def get_diff_eq_names(self) -> list[str]:
        """Retrieve the names of all diff_eqs.

        Examples
        --------
            >>> model.get_diff_eq_names()
                ["v1", "v2"]

        Returns
        -------
        list[str]
            A list containing the names of the diff_eqs.

        """
        return list(self._diff_eqs)

    @_invalidate_cache
    def update_diff_eq(
        self,
        name: str,
        *,
        initial_value: float | InitialAssignment | sympy.Expr,
        fn: RateFn | None = None,
        args: list[str] | None = None,
        unit: sympy.Expr | None = None,
        annotations: Annotation | Iterable[Annotation] | None = None,
    ) -> Self:
        """Updates the properties of an existing diff_eq in the model.

        Examples
        --------
            >>> model.update_diff_eq("v1",
            ...     fn=mass_action,
            ...     args=["x1", "kf1"],
            ...     stoichiometry={"x1": -1, "x2": 1},
            ... )

        Parameters
        ----------
        name
            The name of the diff_eq to update.
        fn
            The new function for the diff_eq. If None, the existing function is retained.
        args
            The new arguments for the diff_eq. If None, the existing arguments are retained.
        stoichiometry
            The new stoichiometry for the diff_eq. If None, the existing stoichiometry is retained.
        unit
            Unit of the diff_eq
        annotations
            MIRIAM annotation(s) to replace the diff_eq's annotations.

        Returns
        -------
        Self
            The instance of the model with the updated diff_eq.

        """
        diff_eq = self._diff_eqs[name]
        diff_eq.fn = diff_eq.fn if fn is None else fn
        diff_eq.initial_value = (
            diff_eq.initial_value
            if initial_value is None
            else _expr_as_initial_assignment(initial_value)
        )

        diff_eq.args = diff_eq.args if args is None else args
        diff_eq.unit = diff_eq.unit if unit is None else unit
        if annotations is not None:
            diff_eq.annotations = _normalize_annotations(annotations)
        return self

    @_invalidate_cache
    def remove_diff_eq(self, name: str) -> Self:
        """Remove a diff_eq from the model by its name.

        Examples
        --------
            >>> model.remove_diff_eq("v1")

        Parameters
        ----------
        name
            The name of the diff_eq to be removed.

        Returns
        -------
        Self
            The instance of the model with the diff_eq removed.

        """
        self._remove_id(name=name)
        self._diff_eqs.pop(name)
        return self

    ##########################################################################
    # Rename
    ##########################################################################

    def _rename_references(self, old_name: str, new_name: str) -> None:
        """Replace every reference to ``old_name`` with ``new_name`` in place.

        Rewrites all ``args`` lists and stoichiometry keys across the model. The
        owning slot itself is handled by :meth:`rename`.

        Parameters
        ----------
        old_name
            The name currently being referenced.
        new_name
            The name to reference instead.

        """

        def rename_args(args: list[str]) -> list[str]:
            return [new_name if arg == old_name else arg for arg in args]

        def rename_stoich(
            stoich: Mapping[str, float | Derived],
        ) -> dict[str, float | Derived]:
            result: dict[str, float | Derived] = {}
            for cpd, factor in stoich.items():
                if isinstance(factor, Derived):
                    factor.args = rename_args(factor.args)
                result[new_name if cpd == old_name else cpd] = factor
            return result

        # Initial assignments on variables and parameters
        for diff_eq in self._diff_eqs.values():
            if isinstance(init := diff_eq.initial_value, InitialAssignment):
                init.args = rename_args(init.args)
            diff_eq.args = rename_args(diff_eq.args)
        for parameter in self._parameters.values():
            if isinstance(value := parameter.value, InitialAssignment):
                value.args = rename_args(value.args)

        # Args of derived values and readouts
        for derived in self._derived.values():
            derived.args = rename_args(derived.args)
        for readout in self._readouts.values():
            readout.args = rename_args(readout.args)

    @_invalidate_cache
    def rename(self, old_name: str, new_name: str) -> Self:
        """Rename a model component and update all references to it.

        Renames any registered name - variable, parameter, derived, diff_eq,
        readout, surrogate, surrogate output or data set - and rewrites every
        reference to it in ``args`` lists and stoichiometries.

        Examples
        --------
            >>> model.rename("v1", "glucose")

        Parameters
        ----------
        old_name
            The current name of the component.
        new_name
            The new name for the component.

        Returns
        -------
        Self
            The instance of the model with the component renamed.

        Raises
        ------
        KeyError
            If ``old_name`` is not a registered name, or ``new_name`` is the
            reserved ``"time"`` identifier.
        NameError
            If ``new_name`` already exists in the model.

        """
        if old_name == new_name:
            return self

        ctx = self._ids[old_name]  # KeyError if old_name is unknown
        # Insert before remove so a collision or reserved name raises before
        # any part of the model is mutated.
        self._insert_id(name=new_name, ctx=ctx)
        self._remove_id(name=old_name)

        # Move the owning slot
        containers: list[dict[str, Any]] = [
            self._parameters,
            self._derived,
            self._readouts,
            self._diff_eqs,
            self._data,
        ]
        for container in containers:
            if old_name in container:
                container[new_name] = container.pop(old_name)
                break

        self._rename_references(old_name, new_name)
        return self

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

        Examples
        --------
            >>> model.add_readout("energy_state",
            ...     fn=div,
            ...     args=["NADPH", "NADP*_total"]
            ... )

        Parameters
        ----------
        name
            The name of the readout.
        fn
            The function to be used for the readout.
        args
            The list of arguments for the function.
        unit
            Unit of the readout

        Returns
        -------
        Self
            The instance of the model with the added readout.

        """
        self._insert_id(name=name, ctx="readout")
        self._readouts[name] = Readout(
            fn=fn,
            args=args,
            unit=unit,
        )
        return self

    def add_readout_from_expr(
        self,
        name: str,
        expr: sympy.Expr,
        *,
        unit: sympy.Expr | None = None,
    ) -> Self:
        """Adds a readout computed from a sympy expression.

        Unlike `add_readout`, there is no separate `args` list: the
        expression's free symbols are used *literally* as argument names,
        so they must already exist as parameters/variables/derived
        quantities in the model.

        Examples
        --------
            >>> nadph, nadp_total = sympy.symbols("NADPH NADP*_total")
            >>> model.add_readout_from_expr("energy_state", nadph / nadp_total)

        Parameters
        ----------
        name
            The name of the readout.
        expr
            A sympy expression whose free symbols name existing model
            components.
        unit
            Unit of the readout

        Returns
        -------
        Self
            The instance of the model with the added readout.

        """
        self._insert_id(name=name, ctx="readout")
        fn, args = _expr_to_fn_and_args(expr)
        self._readouts[name] = Readout(
            fn=fn,
            args=args,
            unit=unit,
        )
        return self

    def get_readout_names(self) -> list[str]:
        """Retrieve the names of all readouts.

        Examples
        --------
            >>> model.get_readout_names()
                ["energy_state", "redox_state"]

        Returns
        -------
        list[str]
            A list containing the names of the readouts.

        """
        return list(self._readouts)

    def get_raw_readouts(self, *, as_copy: bool = True) -> dict[str, Readout]:
        """Get copy of readouts in the model.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the readouts dictionary.
            If False, return the internal dictionary directly.

        Returns
        -------
        dict[str, Readout]
            Dictionary mapping readout names to Readout objects.

        """
        if as_copy:
            return copy.deepcopy(self._readouts)
        return self._readouts

    def remove_readout(self, name: str) -> Self:
        """Remove a readout by its name.

        Examples
        --------
            >>> model.remove_readout("energy_state")

        Parameters
        ----------
        name : str
            The name of the readout to remove.

        Returns
        -------
        Self
            The instance of the class after the readout has been removed.

        """
        self._remove_id(name=name)
        del self._readouts[name]
        return self

    ##########################################################################
    # Datasets
    ##########################################################################

    def add_data(self, name: str, data: pd.Series | pd.DataFrame) -> Self:
        """Add named data set to model.

        Parameters
        ----------
        name
            Name for the data set.
        data
            Data to attach to the model.

        Returns
        -------
        Self
            The model instance.

        """
        self._insert_id(name=name, ctx="data")
        self._data[name] = data
        return self

    def update_data(self, name: str, data: pd.Series | pd.DataFrame) -> Self:
        """Update named data set.

        Parameters
        ----------
        name
            Name of the existing data set to update.
        data
            New data to replace the existing data set.

        Returns
        -------
        Self
            The model instance.

        """
        self._data[name] = data
        return self

    def remove_data(self, name: str) -> Self:
        """Remove data set from model.

        Parameters
        ----------
        name
            Name of the data set to remove.

        Returns
        -------
        Self
            The model instance.

        """
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
        include_diff_eqs: bool,
        include_parameters: bool,
        include_derived_parameters: bool,
        include_derived_variables: bool,
        include_readouts: bool,
    ) -> list[str]:
        """Get names of all kinds of model components.

        Parameters
        ----------
        include_time
            Include "time" as a name.
        include_variables
            Include variable names.
        include_parameters
            Include parameter names.
        include_derived_parameters
            Include derived parameter names.
        include_derived_variables
            Include derived variable names.
        include_diff_eqs
            Include diff_eq names.
        include_readouts
            Include readout names.

        Returns
        -------
        list[str]
            List of selected model component names.

        """
        names = []
        if include_time:
            names.append("time")
        if include_diff_eqs:
            names.extend(self.get_diff_eq_names())
        if include_parameters:
            names.extend(self.get_parameter_names())
        if include_derived_variables:
            names.extend(self.get_derived_variable_names())
        if include_derived_parameters:
            names.extend(self.get_derived_parameter_names())
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

        Examples
        --------
            >>> model._get_args({"x1": 1.0, "x2": 2.0}, time=0.0)
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 0.0}

        Parameters
        ----------
        variables
            A dictionary of concentrations with keys as the names of the substances
            and values as their respective concentrations.
        time
            The time point for the calculation
        cache
            A ModelCache object containing precomputed values and dependencies.
        include_readouts
            A flag indicating whether to include readout values in the returned dictionary.

        Returns
        -------
            dict[str, float]
            A dictionary containing parameter values, derived variables, and optionally readouts,
            with their respective names as keys and their calculated values as values.

        """
        args = cache.all_parameter_values | variables | self._data
        args["time"] = time

        containers = self._derived | self._diff_eqs
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
        include_diff_eqs: bool = True,
        include_parameters: bool = True,
        include_derived_parameters: bool = True,
        include_derived_variables: bool = True,
        include_readouts: bool = False,
    ) -> pd.Series:
        """Generate a pandas Series of arguments for the model.

        Examples
        --------
            # Using initial conditions
            >>> model.get_args()
                {"x1": 1.get_args, "x2": 2.0, "k1": 0.1, "time": 0.0}

            # With custom concentrations
            >>> model.get_args({"x1": 1.0, "x2": 2.0})
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 0.0}

            # With custom concentrations and time
            >>> model.get_args({"x1": 1.0, "x2": 2.0}, time=1.0)
                {"x1": 1.0, "x2": 2.0, "k1": 0.1, "time": 1.0}

        Parameters
        ----------
        variables
            A dictionary where keys are the names of the concentrations and values are their respective float values.
        time
            The time point at which the arguments are generated.
        include_time
            Whether to include the time as an argument
        include_diff_eqs
            Whether to include diff_eqs
        include_parameters
            Whether to include parameters
        include_derived_parameters
            Whether to include derived parameters
        include_derived_variables
            Whether to include derived variables
        include_readouts
            Whether to include readouts

        Returns
        -------
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
                include_parameters=include_parameters,
                include_derived_parameters=include_derived_parameters,
                include_derived_variables=include_derived_variables,
                include_diff_eqs=include_diff_eqs,
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
                variables=cast(dict, values.to_dict()),
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
        include_parameters: bool = True,
        include_derived_parameters: bool = True,
        include_derived_variables: bool = True,
        include_diff_eqs: bool = True,
        include_readouts: bool = False,
    ) -> pd.DataFrame:
        """Generate a DataFrame containing time course arguments for model evaluation.

        Examples
        --------
            >>> model.get_args_time_course(
            ...     pd.DataFrame({"x1": [1.0, 2.0], "x2": [2.0, 3.0]}
            ... )
                pd.DataFrame({
                    "x1": [1.0, 2.0],
                    "x2": [2.0, 3.0],
                    "k1": [0.1, 0.1],
                    "time": [0.0, 1.0]},
                )

        Parameters
        ----------
        variables
            A DataFrame containing concentration data with time as the index.

        include_parameters
            Whether to include parameters
        include_derived_parameters
            Whether to include derived parameters
        include_derived_variables
            Whether to include derived variables
        include_diff_eqs
            Whether to include diff_eqs
        include_readouts
            Whether to include readouts

        Returns
        -------
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
                include_parameters=include_parameters,
                include_derived_parameters=include_derived_parameters,
                include_derived_variables=include_derived_variables,
                include_diff_eqs=include_diff_eqs,
                include_readouts=include_readouts,
            ),
        ]

    ##########################################################################
    # Get rhs
    ##########################################################################

    def __call__(self, /, time: float, variables: Iterable[float]) -> tuple[float, ...]:
        """Simulation version of get_right_hand_side.

        Examples
        --------
            >>> model(0.0, np.array([1.0, 2.0]))
                np.array([0.1, 0.2])

        Warning: Swaps t and y!
        This can't get kw args, as the integrators call it with pos-only

        Parameters
        ----------
        time
            The current time point.
        variables
            Array of concentrations


        Returns
        -------
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

        return tuple(dependent[i] for i in cache.var_names)

    def _get_right_hand_side(
        self,
        *,
        args: dict[str, float],
        cache: ModelCache,
    ) -> pd.Series:
        return pd.Series(
            {i: args[i] for i in cache.var_names},
            dtype=float,
        )

    def get_right_hand_side(
        self,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> pd.Series:
        """Calculate the right-hand side of the differential equations for the model.

        Examples
        --------
            # Using initial conditions as default
            >>> model.get_right_hand_side()
                pd.Series({"x1": 0.1, "x2": 0.2})

            # Using custom concentrations
            >>> model.get_right_hand_side({"x1": 1.0, "x2": 2.0})
                pd.Series({"x1": 0.1, "x2": 0.2})

            # Using custom concentrations and time
            >>> model.get_right_hand_side({"x1": 1.0, "x2": 2.0}, time=0.0)
                pd.Series({"x1": 0.1, "x2": 0.2})

        Parameters
        ----------
        variables
            A dictionary mapping compound names to their concentrations.
        time
            The current time point. Defaults to 0.0.

        Returns
        -------
            The rate of change of each variable in the model.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()
        args = self._get_args(
            variables=self.get_initial_conditions() if variables is None else variables,
            time=time,
            cache=cache,
        )
        return self._get_right_hand_side(args=args, cache=cache)

    def get_right_hand_side_time_course(self, args: pd.DataFrame) -> pd.DataFrame:
        """Calculate the right-hand side of the differential equations for the model.

        Parameters
        ----------
        args
            DataFrame where each row contains the model arguments (variables,
            parameters, derived quantities) at a given time point.

        Returns
        -------
        pd.DataFrame
            DataFrame of right-hand side values indexed by time.

        """
        if (cache := self._cache) is None:
            cache = self._create_cache()

        rhs_by_time = {}
        for time, variables in args.iterrows():
            rhs_by_time[time] = self._get_right_hand_side(
                args=cast(dict, variables.to_dict()),
                cache=cache,
            )
        return pd.DataFrame(rhs_by_time).T
