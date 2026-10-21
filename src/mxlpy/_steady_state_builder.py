# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import copy
import inspect
import itertools as it
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Self, cast

import pandas as pd
from wadler_lindig import pformat

from mxlpy import _topo
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
)
from mxlpy.parallel import Cache, parallelise
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import (
    Annotation,
    Derived,
    InitialAssignment,
    Parameter,
    Slider,
)
from mxlpy.unit_inference import (
    _latex_view,
)

__all__ = ["LOGGER", "ModelCache", "SteadyStateModelBuilder", "TableView"]

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

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


def _simulate_worker(
    params: dict[str, float], *, model: SteadyStateModelBuilder
) -> pd.Series:
    local_model = copy.deepcopy(model)
    local_model.update_parameters(params)
    return local_model.simulate()


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


@dataclass(slots=True)
class ModelCache:
    """ModelCache is a class that stores various model-related data structures.

    Attributes
    ----------
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

    """

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    order: list[str]  # mostly for debug purposes
    dyn_order: list[str]
    base_parameter_values: dict[str, float]
    all_parameter_values: dict[str, float]


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
        parameter_names = set(self._parameters)
        all_parameter_names = set(parameter_names)

        base_parameter_values: dict[str, float] = {
            k: val
            for k, v in self._parameters.items()
            if not isinstance(val := v.value, InitialAssignment)
        }

        # Sanity checks
        for name, el in it.chain(
            self._derived.items(),
        ):
            if not _check_function_arity(el.fn, len(el.args)):
                raise ArityMismatchError(name, el.fn, el.args)

        # Sort derived & reactions
        available = (
            set(base_parameter_values)
        )
        to_sort = (
            self._derived
        )
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
            base_parameter_values
        )
        for name in order:
            to_sort[name].calculate_inpl(name, dependent)

        # Split derived into static and dynamic variables
        static_order = []
        dyn_order = []
        for name in order:
            derived = self._derived[name]
            if all(i in all_parameter_names for i in derived.args):
                static_order.append(name)
                all_parameter_names.add(name)
            else:
                dyn_order.append(name)

        all_parameter_values = dict(base_parameter_values)
        for name in static_order:
            if name in self._parameters or name in self._derived:
                all_parameter_values[name] = cast(float, dependent[name])
            else:
                msg = f"Internal error: '{name}' appears in static_order but is not a parameter, variable, or derived - this is a bug in dependency sorting"
                raise KeyError(msg)

        self._cache = ModelCache(
            order=order,
            dyn_order=dyn_order,
            base_parameter_values=base_parameter_values,
            all_parameter_values=all_parameter_values,
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
        *,
        display_name: str | None = None,
        tex_name: str | None = None,
        slider: Slider | None = None,
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
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.
        slider
            Interactive-slider configuration for mxlweb.

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
            display_name=display_name,
            tex_name=tex_name,
            slider=slider,
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
                    display_name=v.display_name,
                    tex_name=v.tex_name,
                    slider=v.slider,
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
        display_name: str | None = None,
        tex_name: str | None = None,
        slider: Slider | None = None,
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
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.
        slider
            Interactive-slider configuration for mxlweb.

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
        if display_name is not None:
            parameter.display_name = display_name
        if tex_name is not None:
            parameter.tex_name = tex_name
        if slider is not None:
            parameter.slider = slider
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
                    display_name=v.display_name,
                    tex_name=v.tex_name,
                    slider=v.slider,
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
        display_name: str | None = None,
        tex_name: str | None = None,
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
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.

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
            display_name=display_name,
            tex_name=tex_name,
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
        display_name: str | None = None,
        tex_name: str | None = None,
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
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.

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
        if display_name is not None:
            der.display_name = display_name
        if tex_name is not None:
            der.tex_name = tex_name
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

    ##########################################################################
    # Simulate
    ##########################################################################

    @_invalidate_cache
    def simulate(self) -> pd.Series:
        """Simulate the model with the current parameter values and return the derived values.
        
        Examples
        --------
            >>> model.simulate()
                {"d1": 10, "d2": 20}

        Returns
        -------
            A Series of the calculated derived values.

        """
        derived = self.get_raw_derived(as_copy=False)

        res = {}

        for name, der in derived.items():
            res[name] = der.calculate(self.get_parameter_values())

        return pd.Series(res)

    @_invalidate_cache
    def simulate_scan(
        self,
        to_scan: pd.DataFrame,
        cache: Cache | None = None,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Simulate the model for each row of parameter values.
        
        Returns
        -------
        SteadyStates
            Steady-state results for each parameter set.

        Examples
        --------
            >>> model.simulate_scan(
            >>>     to_scan=pd.DataFrame({"k1": np.linspace(1, 2, 3)})
            >>> )
            k1      x      y
            1.0   0.50   1.00
            1.5   0.75   1.50
            2.0   1.00   2.00
            
            >>> model.simulate_scan(
            >>>     to_scan=cartesian_product({"k1": [1, 2], "k2": [3, 4]})
            >>> )
            | k1 | k2 |    x |   y |
            | 1  | 3  | 0.33 |   1 |
            | 1  | 4  | 0.25 |   1 |
            | 2  | 3  | 0.66 |   2 |
            | 2  | 4  | 0.5  |   2 |
    
        """
        raw_index = (
            pd.Index(to_scan.iloc[:, 0])
            if to_scan.shape[1] == 1
            else pd.MultiIndex.from_frame(to_scan)
        )

        inputs: list[tuple[Hashable, dict[str, float]]] = [
            (index, cast(dict[str, float], row.to_dict()))
            for index, row in to_scan.iterrows()
        ]
        res = parallelise(
            partial(_simulate_worker, model=self),
            inputs=inputs,
            cache=cache,
            parallel=parallel,
        )

        raw_results = [i[1] for i in res]

        return pd.DataFrame(raw_results, index=raw_index)
