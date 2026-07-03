from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import pandas as pd
from wadler_lindig import pformat

from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
)
from mxlpy.types import (
    Annotation,
    Derived,
    InitialAssignment,
    Parameter,
)
from mxlpy.unit_inference import (
    _latex_view,
)

__all__ = ["LOGGER", "ModelCache", "SteadyStateModelBuilder", "TableView"]

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import sympy

    from mxlpy.types import Callable, Param, RateFn, RetType

LOGGER = logging.getLogger(__name__)


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
    stoich_by_cpds: dict[str, dict[str, float]]
    dyn_stoich_by_cpds: dict[str, dict[str, Derived]]
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
        self = cast(SteadyStateModelBuilder, args[0])
        self._cache = None
        return method(*args, **kwargs)

    return wrapper  # type: ignore


@dataclass(slots=True)
class SteadyStateModelBuilder:
    _ids: dict[str, str] = field(default_factory=dict, repr=False)
    _cache: ModelCache | None = field(default=None, repr=False)
    _parameters: dict[str, Parameter] = field(default_factory=dict)
    _derived: dict[str, Derived] = field(default_factory=dict)

    ###########################################################################
    # Cache
    ###########################################################################

    def _create_cache(self) -> ModelCache:
        """Creates and initializes the model cache.

        This method constructs a cache that includes parameter values, stoichiometry
        by compounds, dynamic stoichiometry by compounds, derived variables, and
        derived parameters. It processes the model's parameters, variables, derived
        elements, reactions, and surrogates to populate the cache.

        Returns
        -------
        ModelCache
            An instance of ModelCache containing the initialized cache data.

        """
        raise NotImplementedError

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
            msg = "'time' is a reserved identifier - it represents the simulation time and cannot be used as a parameter, variable, derived, or reaction name"
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
        value: float | InitialAssignment,
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
            value=value,
            unit=unit,
            source=source,
            annotations=_normalize_annotations(annotations),
        )
        return self

    def add_parameters(
        self, parameters: Mapping[str, float | Parameter | InitialAssignment]
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
        value: float | InitialAssignment | None = None,
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
            parameter.value = value
        if unit is not None:
            parameter.unit = unit
        if source is not None:
            parameter.source = source
        if annotations is not None:
            parameter.annotations = _normalize_annotations(annotations)
        return self

    def update_parameters(
        self, parameters: Mapping[str, float | Parameter | InitialAssignment]
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
