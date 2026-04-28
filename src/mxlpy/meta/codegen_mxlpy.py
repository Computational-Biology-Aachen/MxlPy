"""Module to export models as code without the SymPy roundtrip.

See Also
--------
mxlpy.meta.codegen_mxlpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mxlpy.meta.source_tools import fn_to_source
from mxlpy.types import (
    Derived,
    InitialAssignment,
    Parameter,
    Reaction,
    Readout,
    Variable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model
    from mxlpy.surrogates.abstract import SurrogateProtocol

__all__ = ["Context", "generate_mxlpy_code"]

_LOGGER = logging.getLogger()


@dataclass
class Context:
    functions: dict[str, str]
    mxlpy_imports: set[str]
    file_imports: set[str]


def _fn_name(k: str, fn: Callable) -> str:
    return fn.__name__ if fn.__name__ != "<lambda>" else k


def _handle_fn_source(
    name: str,
    fn: Callable,
    args: list[str],
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    fn_name = _fn_name(name, fn)
    extracted = fn_to_source(fn, fn_name, args, strip_docstring=strip_docstring)
    ctx.functions.update(extracted.dependencies)
    ctx.functions[fn_name] = extracted.main_source
    ctx.file_imports.update(extracted.imports)
    return fn_name


def _codegen_variable(
    k: str,
    variable: Variable,
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    val = variable.initial_value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(
            k,
            val.fn,
            val.args,
            ctx=ctx,
            strip_docstring=strip_docstring,
        )
        ctx.mxlpy_imports.add("InitialAssignment")
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
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    val = parameter.value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(
            k,
            val.fn,
            val.args,
            ctx=ctx,
            strip_docstring=strip_docstring,
        )
        ctx.mxlpy_imports.add("InitialAssignment")
        return (
            "        .add_parameter(\n"
            f"            {k!r},\n"
            f"            value=InitialAssignment(fn={fn_name}, args={val.args!r}),\n"
            "        )"
        )

    # val is float | int
    return f"        .add_parameter({k!r}, value={val!r})"


def _codegen_derived(
    k: str,
    der: Derived,
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    fn_name = _handle_fn_source(
        k,
        der.fn,
        der.args,
        ctx=ctx,
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
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    fn_name = _handle_fn_source(
        k,
        rxn.fn,
        rxn.args,
        ctx=ctx,
        strip_docstring=strip_docstring,
    )

    stoichiometry: list[str] = []
    for var, stoich in rxn.stoichiometry.items():
        if isinstance(stoich, Derived):
            ctx.mxlpy_imports.add("Derived")
            stoich_fn_name = _handle_fn_source(
                var,
                stoich.fn,
                stoich.args,
                ctx=ctx,
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
    *,
    ctx: Context,
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
    ctx.file_imports.add(f"from mxlpy.surrogates import {submod}")

    # Generate source for callable model (e.g. QSS)
    surr_model_fn_name = None
    model_attr = getattr(surrogate, "model", None)
    if callable(model_attr):
        try:
            surr_model_fn_name = _handle_fn_source(
                k,
                model_attr,
                surrogate.args,
                ctx=ctx,
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
                        ctx=ctx,
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


def _codegen_readout(
    k: str,
    der: Readout,
    *,
    ctx: Context,
    strip_docstring: bool,
) -> str:
    fn_name = _handle_fn_source(
        k,
        der.fn,
        der.args,
        ctx=ctx,
        strip_docstring=strip_docstring,
    )
    return (
        "        .add_readout(\n"
        f"            {k!r},\n"
        f"            fn={fn_name},\n"
        f"            args={der.args!r},\n"
        "        )"
    )


def generate_mxlpy_code(
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
    ctx = Context(
        functions={},
        mxlpy_imports={"Model"},
        file_imports=set() if imports is None else set(imports),
    )

    # Variables
    variable_source = []
    for k, variable in model.get_raw_variables().items():
        variable_source.append(
            _codegen_variable(
                k,
                variable,
                ctx=ctx,
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
                ctx=ctx,
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
                ctx=ctx,
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
                ctx=ctx,
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
                ctx=ctx,
                strip_docstring=strip_docstring,
            )
        ) is not None:
            surrogate_source.append(gen)

    # Surrogates
    readout_source = []
    for k, ro in model.get_raw_readouts().items():
        readout_source.append(
            _codegen_readout(
                k,
                ro,
                ctx=ctx,
                strip_docstring=strip_docstring,
            )
        )

    functions_source = "\n\n".join(ctx.functions.values())
    fn_block = "\n" if len(ctx.functions) == 0 else f"\n\n{functions_source}\n"

    source = [
        *sorted(ctx.file_imports),
        f"from mxlpy import {','.join(sorted(ctx.mxlpy_imports))}",
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
    if readout_source:
        source.append("\n".join(readout_source))
    source.append("    )")
    return "\n".join(source)
