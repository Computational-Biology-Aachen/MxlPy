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

_LOGGER = logging.getLogger(__name__)


@dataclass
class Context:
    """Various config."""

    functions: dict[str, str]
    mxlpy_imports: set[str]
    file_imports: set[str]
    strip_docstring: bool
    first: str
    space: str


def _fn_name(k: str, fn: Callable) -> str:
    return fn.__name__ if fn.__name__ != "<lambda>" else k


def _handle_fn_source(
    name: str,
    fn: Callable,
    args: list[str],
    *,
    ctx: Context,
) -> str:
    fn_name = _fn_name(name, fn)
    extracted = fn_to_source(fn, fn_name, args, strip_docstring=ctx.strip_docstring)
    ctx.functions.update(extracted.dependencies)
    ctx.functions[fn_name] = extracted.main_source
    ctx.file_imports.update(extracted.imports)
    return fn_name


def _codegen_variable(k: str, variable: Variable, *, ctx: Context) -> str:
    val = variable.initial_value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(k, val.fn, val.args, ctx=ctx)
        ctx.mxlpy_imports.add("InitialAssignment")
        return (
            f"{ctx.first}.add_variable(\n"
            f"{ctx.space}  {k!r},\n"
            f"{ctx.space}  initial_value=InitialAssignment(fn={fn_name}, args={val.args!r}),\n"
            f"{ctx.space})"
        )

    # val is float | int
    return f"{ctx.first}.add_variable({k!r}, initial_value={val!r})"


def _codegen_parameter(
    k: str,
    parameter: Parameter,
    *,
    ctx: Context,
) -> str:
    val = parameter.value
    if isinstance(val, InitialAssignment):
        fn_name = _handle_fn_source(k, val.fn, val.args, ctx=ctx)
        ctx.mxlpy_imports.add("InitialAssignment")
        return (
            f"{ctx.first}.add_parameter(\n"
            f"{ctx.space}  {k!r},\n"
            f"{ctx.space} value=InitialAssignment(fn={fn_name}, args={val.args!r}),\n"
            f"{ctx.space})"
        )

    # val is float | int
    return f"{ctx.first}.add_parameter({k!r}, value={val!r})"


def _codegen_derived(
    k: str,
    der: Derived,
    *,
    ctx: Context,
) -> str:
    fn_name = _handle_fn_source(
        k,
        der.fn,
        der.args,
        ctx=ctx,
    )
    return (
        f"{ctx.first}.add_derived(\n"
        f"{ctx.space}  {k!r},\n"
        f"{ctx.space}  fn={fn_name},\n"
        f"{ctx.space}  args={der.args!r},\n"
        f"{ctx.space})"
    )


def _codegen_reaction(
    k: str,
    rxn: Reaction,
    *,
    ctx: Context,
) -> str:
    fn_name = _handle_fn_source(
        k,
        rxn.fn,
        rxn.args,
        ctx=ctx,
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
            )
            stoichiometry.append(
                f'"{var}": Derived(fn={stoich_fn_name}, args={stoich.args!r})'
            )
        else:
            stoichiometry.append(f'"{var}": {stoich!r}')
    return (
        f"{ctx.first}.add_reaction(\n"
        f"{ctx.space}  {k!r},\n"
        f"{ctx.space}  fn={fn_name},\n"
        f"{ctx.space}  args={rxn.args!r},\n"
        f"{ctx.space}  stoichiometry={{{','.join(stoichiometry)}}},\n"
        f"{ctx.space})"
    )


def _codegen_surrogate(
    k: str,
    surrogate: SurrogateProtocol,
    *,
    ctx: Context,
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
        f"{ctx.first}.add_surrogate(\n"
        f"{ctx.space}  {k!r},\n"
        f"{ctx.space}  {class_ref}(\n"
        f"{ctx.space}    {ctor_str},\n"
        f"{ctx.space}    ),\n"
        f"{ctx.space})"
    )


def _codegen_readout(
    k: str,
    der: Readout,
    *,
    ctx: Context,
) -> str:
    fn_name = _handle_fn_source(
        k,
        der.fn,
        der.args,
        ctx=ctx,
    )
    return (
        f"{ctx.first}.add_readout(\n"
        f"{ctx.space}  {k!r},\n"
        f"{ctx.space}  fn={fn_name},\n"
        f"{ctx.space}  args={der.args!r},\n"
        f"{ctx.space})"
    )


def generate_mxlpy_code(
    model: Model,
    *,
    model_fn_name: str = "create_model",
    imports: list[str] | None = None,
    strip_docstring: bool = True,
    as_assignment: bool = False,
    docstring: str | None = None,
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
        first="    m = m" if as_assignment else "        ",
        space="    " if as_assignment else "        ",
        strip_docstring=strip_docstring,
    )

    # Variables
    variable_source = []
    for k, variable in model.get_raw_variables().items():
        variable_source.append(_codegen_variable(k, variable, ctx=ctx))

    # Parameters
    parameter_source = []
    for k, parameter in model.get_raw_parameters().items():
        parameter_source.append(_codegen_parameter(k, parameter, ctx=ctx))

    # Derived
    derived_source = []
    for k, der in model.get_raw_derived().items():
        derived_source.append(_codegen_derived(k, der, ctx=ctx))

    # Reactions
    reactions_source = []
    for k, rxn in model.get_raw_reactions().items():
        reactions_source.append(_codegen_reaction(k, rxn, ctx=ctx))

    # Surrogates
    surrogate_source = []
    for k, surrogate in model.get_raw_surrogates().items():
        if (gen := _codegen_surrogate(k, surrogate, ctx=ctx)) is not None:
            surrogate_source.append(gen)

    # Surrogates
    readout_source = []
    for k, ro in model.get_raw_readouts().items():
        readout_source.append(_codegen_readout(k, ro, ctx=ctx))

    functions_source = "\n\n".join(ctx.functions.values())
    fn_block = "\n" if len(ctx.functions) == 0 else f"\n\n{functions_source}\n"

    source = [
        *sorted(ctx.file_imports),
        f"from mxlpy import {','.join(sorted(ctx.mxlpy_imports))}",
        fn_block,
        f"def {model_fn_name}() -> Model:",
    ]
    if docstring is not None:
        source.append(f"    {docstring}")
    if as_assignment:
        source.append("    m: Model = Model()")
    else:
        source.append("    return (")
        source.append("        Model()")
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
    if as_assignment:
        source.append("    return m  # noqa: RET504")
    else:
        source.append("    )")
    return "\n".join(source)
