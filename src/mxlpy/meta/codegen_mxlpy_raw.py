"""Module to export models as code without the SymPy roundtrip.

See Also
--------
mxlpy.meta.codegen_mxlpy
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mxlpy.meta.source_tools import fn_to_source
from mxlpy.types import Derived, InitialAssignment, Parameter, Reaction, Variable

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model
    from mxlpy.surrogates.abstract import SurrogateProtocol

__all__ = [
    "generate_mxlpy_code_raw",
]

_LOGGER = logging.getLogger()


def _fn_name(k: str, fn: Callable) -> str:
    return fn.__name__ if fn.__name__ != "<lambda>" else k


def _handle_fn_source(
    name: str,
    fn: Callable,
    args: list[str],
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str:
    fn_name = _fn_name(name, fn)
    extracted = fn_to_source(fn, name, args, strip_docstring=strip_docstring)
    functions.update(extracted.dependencies)
    functions[name] = extracted.main_source
    file_imports.update(extracted.imports)
    return fn_name


def _codegen_variable(
    k: str,
    variable: Variable,
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str:
    val = variable.initial_value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(
            k,
            val.fn,
            val.args,
            functions,
            file_imports,
            strip_docstring=strip_docstring,
        )
        return (
            "        .add_variable(\n"
            f"            {k!r},\n"
            f"            initial_value=InitialAssignment(fn={fn_name}, args={val.args!r}),\n"
            "        )"
        )

    # val is float | int
    return f"        .add_variable({k!r}, initial_value={val!r})"


def _codegen_parameter(
    k: str,
    parameter: Parameter,
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str:
    val = parameter.value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(
            k,
            val.fn,
            val.args,
            functions,
            file_imports,
            strip_docstring=strip_docstring,
        )
        return (
            "        .add_parameter(\n"
            f"            {k!r},\n"
            f"            initial_value=InitialAssignment(fn={fn_name}, args={val.args!r}),\n"
            "        )"
        )

    # val is float | int
    return f"        .add_parameter({k!r}, value={val!r})"


def _codegen_derived(
    k: str,
    der: Derived,
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str:
    fn_name = _handle_fn_source(
        k,
        der.fn,
        der.args,
        functions,
        file_imports,
        strip_docstring=strip_docstring,
    )
    return (
        "        .add_derived(\n"
        f"            {k!r},\n"
        f"            fn={fn_name},\n"
        f"            args={der.args!r},\n"
        "        )"
    )


def _codegen_reaction(
    k: str,
    rxn: Reaction,
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str:
    fn_name = _handle_fn_source(
        k,
        rxn.fn,
        rxn.args,
        functions,
        file_imports,
        strip_docstring=strip_docstring,
    )

    stoichiometry: list[str] = []
    for var, stoich in rxn.stoichiometry.items():
        if isinstance(stoich, Derived):
            stoich_fn_name = _handle_fn_source(
                var,
                stoich.fn,
                stoich.args,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
            stoichiometry.append(
                f'"{var}": Derived(fn={stoich_fn_name}, args={stoich.args!r})'
            )
        else:
            stoichiometry.append(f'"{var}": {stoich!r}')
    return (
        "        .add_reaction(\n"
        f"            {k!r},\n"
        f"            fn={fn_name},\n"
        f"            args={rxn.args!r},\n"
        f"            stoichiometry={{{','.join(stoichiometry)}}},\n"
        "        )"
    )


def _codegen_surrogate(
    k: str,
    surrogate: SurrogateProtocol,
    functions: dict[str, str],
    file_imports: set[str],
    *,
    strip_docstring: bool,
) -> str | None:
    stype = type(surrogate)
    module = stype.__module__

    if not module.startswith("mxlpy.surrogates."):
        _LOGGER.warning(
            "Cannot generate code for surrogate type '%s'", stype.__qualname__
        )
        return None

    # mxlpy.surrogates._qss -> 'qss', mxlpy.surrogates.abstract -> 'abstract'
    raw_submod = module.split(".")[-1]
    submod = raw_submod.lstrip("_")
    class_ref = f"{submod}.{stype.__name__}"
    file_imports.add(f"from mxlpy.surrogates import {submod}")

    # Generate source for callable model (e.g. QSS)
    surr_model_fn_name = None
    model_attr = getattr(surrogate, "model", None)
    if callable(model_attr):
        try:
            surr_model_fn_name = _handle_fn_source(
                k,
                model_attr,
                surrogate.args,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        except (ValueError, TypeError, OSError) as exc:
            _LOGGER.warning(
                "Cannot extract source for surrogate '%s' model: %s", k, exc
            )

    # Build constructor kwargs
    ctor_parts: list[str] = []
    if surr_model_fn_name is not None:
        ctor_parts.append(f"model={surr_model_fn_name}")
    ctor_parts.append(f"args={surrogate.args!r}")
    ctor_parts.append(f"outputs={surrogate.outputs!r}")

    if surrogate.stoichiometries:
        stoich_entries: list[str] = []
        for rxn_name, rxn_stoich in surrogate.stoichiometries.items():
            rxn_parts: list[str] = []
            for var, factor in rxn_stoich.items():
                if isinstance(factor, Derived):
                    sf_name = _handle_fn_source(
                        var,
                        factor.fn,
                        factor.args,
                        functions,
                        file_imports,
                        strip_docstring=strip_docstring,
                    )
                    rxn_parts.append(
                        f'"{var}": Derived(fn={sf_name}, args={factor.args!r})'
                    )
                else:
                    rxn_parts.append(f'"{var}": {factor!r}')
            stoich_entries.append(f'"{rxn_name}": {{{",".join(rxn_parts)}}}')
        ctor_parts.append(f"stoichiometries={{{','.join(stoich_entries)}}}")

    ctor_str = ",\n                ".join(ctor_parts)
    return (
        "        .add_surrogate(\n"
        f"            {k!r},\n"
        f"            {class_ref}(\n"
        f"                {ctor_str},\n"
        "            ),\n"
        "        )"
    )


def generate_mxlpy_code_raw(
    model: Model,
    *,
    model_fn_name: str = "create_model",
    imports: list[str] | None = None,
    strip_docstring: bool = True,
) -> str:
    """Generate MxlPy source code without a symbolic representation round-trip.

    Parameters
    ----------
    model
        Model
    imports
        Optional list of import statements to include at the top of the file

    Returns
    -------
    str
        Python source code that constructs the model using MxlPy API

    """
    file_imports: set[str] = set() if imports is None else set(imports)

    # Maps fn_name -> source string (last writer wins for shared functions)
    functions: dict[str, str] = {}

    # Variables
    variable_source = []
    for k, variable in model.get_raw_variables().items():
        variable_source.append(
            _codegen_variable(
                k,
                variable,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        )

    # Parameters
    parameter_source = []
    for k, parameter in model.get_raw_parameters().items():
        parameter_source.append(
            _codegen_parameter(
                k,
                parameter,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        )

    # Derived
    derived_source = []
    for k, der in model.get_raw_derived().items():
        derived_source.append(
            _codegen_derived(
                k,
                der,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        )

    # Reactions
    reactions_source = []
    for k, rxn in model.get_raw_reactions().items():
        reactions_source.append(
            _codegen_reaction(
                k,
                rxn,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        )

    # Surrogates
    surrogate_source = []
    for k, surrogate in model.get_raw_surrogates().items():
        if (
            gen := _codegen_surrogate(
                k,
                surrogate,
                functions,
                file_imports,
                strip_docstring=strip_docstring,
            )
        ) is not None:
            surrogate_source.append(gen)

    functions_source = "\n\n".join(functions.values())
    fn_block = "\n" if len(functions) == 0 else f"\n\n{functions_source}\n"

    source = [
        *sorted(file_imports),
        "from mxlpy import Model, Derived, InitialAssignment",
        fn_block,
        f"def {model_fn_name}() -> Model:",
        "    return (",
        "        Model()",
    ]
    if variable_source:
        source.append("\n".join(variable_source))
    if parameter_source:
        source.append("\n".join(parameter_source))
    if derived_source:
        source.append("\n".join(derived_source))
    if reactions_source:
        source.append("\n".join(reactions_source))
    if surrogate_source:
        source.append("\n".join(surrogate_source))
    source.append("    )")
    return "\n".join(source)
