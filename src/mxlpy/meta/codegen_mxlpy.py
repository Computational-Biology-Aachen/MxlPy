"""Generate mxlpy code from a model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import sympy
from wadler_lindig import pformat

from mxlpy.meta.sympy_tools import (
    fn_to_sympy,
    list_of_symbols,
    sympy_to_inline_py,
    sympy_to_python_fn,
)
from mxlpy.types import Derived, InitialAssignment
from mxlpy.units import Quantity

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model

__all__ = [
    "SymbolicFn",
    "SymbolicParameter",
    "SymbolicReaction",
    "SymbolicRepr",
    "SymbolicVariable",
    "generate_mxlpy_code",
    "generate_mxlpy_code_from_symbolic_repr",
]

_LOGGER = logging.getLogger()


@dataclass
class SymbolicFn:
    """Container for symbolic fn."""

    fn_name: str
    expr: sympy.Expr
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicVariable:
    """Container for symbolic variable."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicParameter:
    """Container for symbolic par."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicReaction:
    """Container for symbolic rxn."""

    fn: SymbolicFn
    stoichiometry: dict[str, sympy.Float | str | SymbolicFn]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicRepr:
    """Container for symbolic model."""

    variables: dict[str, SymbolicVariable] = field(default_factory=dict)
    parameters: dict[str, SymbolicParameter] = field(default_factory=dict)
    derived: dict[str, SymbolicFn] = field(default_factory=dict)
    reactions: dict[str, SymbolicReaction] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


def _fn_to_symbolic_repr(k: str, fn: Callable, model_args: list[str]) -> SymbolicFn:
    fn_name = fn.__name__
    args = cast(list, list_of_symbols(model_args))
    if (expr := fn_to_sympy(fn, origin=k, model_args=args)) is None:
        msg = f"Unable to parse fn for '{k}'"
        raise ValueError(msg)
    return SymbolicFn(fn_name=fn_name, expr=expr, args=model_args)


def _to_symbolic_repr(model: Model) -> SymbolicRepr:
    sym = SymbolicRepr()

    for k, variable in model.get_raw_variables().items():
        sym.variables[k] = SymbolicVariable(
            value=_fn_to_symbolic_repr(k, val.fn, val.args)
            if isinstance(val := variable.initial_value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, variable.unit),
        )

    for k, parameter in model.get_raw_parameters().items():
        sym.parameters[k] = SymbolicParameter(
            value=_fn_to_symbolic_repr(k, val.fn, val.args)
            if isinstance(val := parameter.value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, parameter.unit),
        )

    for k, der in model.get_raw_derived().items():
        sym.derived[k] = _fn_to_symbolic_repr(k, der.fn, der.args)

    for k, rxn in model.get_raw_reactions().items():
        sym.reactions[k] = SymbolicReaction(
            fn=_fn_to_symbolic_repr(k, rxn.fn, rxn.args),
            stoichiometry={
                k: _fn_to_symbolic_repr(k, v.fn, v.args)
                if isinstance(v, Derived)
                else sympy.Float(v)
                for k, v in rxn.stoichiometry.items()
            },
        )

    if len(model._surrogates) > 0:  # noqa: SLF001
        msg = "Generating code for Surrogates not yet supported."
        _LOGGER.warning(msg)
    return sym


def _codegen_variable(
    k: str, var: SymbolicVariable, functions: dict[str, tuple[sympy.Expr, list[str]]]
) -> str:
    if isinstance(init := var.value, SymbolicFn):
        fn_name = f"init_{init.fn_name}"
        functions[fn_name] = (init.expr, init.args)
        return f"""        .add_variable(
            {k!r},
            initial_value=InitialAssignment(fn={fn_name}, args={init.args!r}),
        )"""

    value = sympy_to_inline_py(init)
    if (unit := var.unit) is not None:
        return f"        .add_variable({k!r}, value={value}, unit={sympy_to_inline_py(unit)})"
    return f"        .add_variable({k!r}, initial_value={value})"


def _codegen_parameter(
    k: str, par: SymbolicParameter, functions: dict[str, tuple[sympy.Expr, list[str]]]
) -> str:
    if isinstance(init := par.value, SymbolicFn):
        fn_name = f"init_{init.fn_name}"
        functions[fn_name] = (init.expr, init.args)
        return f"""        .add_parameter(
            {k!r},
            value=InitialAssignment(fn={fn_name}, args={init.args!r}),
        )"""

    value = sympy_to_inline_py(init)
    if (unit := par.unit) is not None:
        return f"        .add_parameter({k!r}, value={value}, unit={sympy_to_inline_py(unit)})"
    return f"        .add_parameter({k!r}, value={value})"


def generate_mxlpy_code_from_symbolic_repr(
    model: SymbolicRepr, imports: list[str] | None = None
) -> str:
    """Generate MxlPy source code from symbolic representation.

    This is both used by MxlPy internally to codegen an existing model again and by the
    SBML import to generate the file.
    """
    imports = [] if imports is None else imports

    functions: dict[str, tuple[sympy.Expr, list[str]]] = {}

    # Variables
    variable_source = []
    for k, var in model.variables.items():
        variable_source.append(_codegen_variable(k, var, functions=functions))

    # Parameters
    parameter_source = []
    for k, par in model.parameters.items():
        parameter_source.append(_codegen_parameter(k, par, functions=functions))

    # Derived
    derived_source = []
    for k, fn in model.derived.items():
        functions[fn.fn_name] = (fn.expr, fn.args)
        derived_source.append(
            f"""        .add_derived(
                {k!r},
                fn={fn.fn_name},
                args={fn.args},
            )"""
        )

    # Reactions
    reactions_source = []
    for k, rxn in model.reactions.items():
        fn = rxn.fn
        functions[fn.fn_name] = (fn.expr, fn.args)

        stoichiometry: list[str] = []
        for var, stoich in rxn.stoichiometry.items():
            if isinstance(stoich, SymbolicFn):
                fn_name = f"{k}_stoich_{stoich.fn_name}"
                functions[fn_name] = (stoich.expr, stoich.args)
                stoichiometry.append(
                    f""""{var}": Derived(fn={fn_name}, args={stoich.args!r})"""
                )
            elif isinstance(stoich, str):
                stoichiometry.append(f""""{var}": {stoich!r}""")
            else:
                stoichiometry.append(f""""{var}": {sympy_to_inline_py(stoich)}""")
        reactions_source.append(
            f"""        .add_reaction(
                "{k}",
                fn={fn.fn_name},
                args={fn.args},
                stoichiometry={{{",".join(stoichiometry)}}},
            )"""
        )

    # Surrogates

    # Combine all the sources
    functions_source = "\n\n".join(
        sympy_to_python_fn(fn_name=name, args=args, expr=expr)
        for name, (expr, args) in functions.items()
    )
    source = [
        *imports,
        "from mxlpy import Model, Derived, InitialAssignment\n",
        functions_source,
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
    ]
    if len(variable_source) > 0:
        source.append("\n".join(variable_source))
    if len(parameter_source) > 0:
        source.append("\n".join(parameter_source))
    if len(derived_source) > 0:
        source.append("\n".join(derived_source))
    if len(reactions_source) > 0:
        source.append("\n".join(reactions_source))
    source.append("    )")
    return "\n".join(source)


def generate_mxlpy_code(model: Model) -> str:
    """Generate a mxlpy model from a model."""
    return generate_mxlpy_code_from_symbolic_repr(_to_symbolic_repr(model))
