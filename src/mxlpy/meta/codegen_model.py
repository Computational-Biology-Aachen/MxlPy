"""Generate mxlpy code from a model.

See Also
--------
`mxlpy.meta.codegen_mxlpy_raw` for a version that doesn't use SymPy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import sympy

from mxlpy.meta.source_tools import fn_to_sympy_exprs
from mxlpy.meta.sympy_tools import (
    fn_to_sympy_expr,
    list_of_symbols,
    stoichiometries_to_sympy,
    sympy_to_inline_c,
    sympy_to_inline_cxx,
    sympy_to_inline_js,
    sympy_to_inline_julia,
    sympy_to_inline_matlab,
    sympy_to_inline_py,
    sympy_to_inline_rust,
)
from mxlpy.meta.utils import valid_identifier
from mxlpy.types import Derived, Reaction, Readout

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model

__all__ = [
    "generate_model_code_c",
    "generate_model_code_cpp",
    "generate_model_code_jl",
    "generate_model_code_matlab",
    "generate_model_code_py",
    "generate_model_code_rs",
    "generate_model_code_ts",
    "generate_model_components_c",
    "generate_model_components_cpp",
    "generate_model_components_jl",
    "generate_model_components_matlab",
    "generate_model_components_py",
    "generate_model_components_rs",
    "generate_model_components_ts",
]

_LOGGER = logging.getLogger(__name__)


def _collect_transitive_deps(
    requested: set[str],
    derived: dict[str, Derived],
    reactions: dict[str, Reaction],
    readouts: dict[str, Readout],
    leaf_names: set[str],
) -> set[str]:
    """BFS to find all component names needed to compute requested outputs.

    Parameters
    ----------
    requested
        Names of the desired output components
    derived
        All dynamic derived values in the model
    reactions
        All reactions in the model
    readouts
        All readouts in the model
    leaf_names
        Names that are already available (variables, parameters, "time")

    Returns
    -------
    set[str]
        All component names (including intermediates) required to compute
        every name in ``requested``

    """
    all_computable: dict[str, Derived | Reaction | Readout] = {
        **derived,
        **reactions,
        **readouts,
    }
    needed: set[str] = set()
    queue = [n for n in requested if n in all_computable]
    while queue:
        name = queue.pop()
        if name in needed:
            continue
        needed.add(name)
        queue.extend(
            dep
            for dep in all_computable[name].args
            if dep not in leaf_names and dep not in needed and dep in all_computable
        )
    return needed


def _flush_ready_surrogates(
    source,
    surrogates_raw,
    emitted_surrogates,
    available,
    sur_exprs,
    sym_subs,
    name_map,
    assignment_template,
    sympy_inline_fn,
    diff_eqs,
) -> None:
    # Greedy: keep emitting surrogates whose args are all available
    progress = True
    while progress:
        progress = False
        for sur_name, surrogate in surrogates_raw.items():
            if sur_name in emitted_surrogates:
                continue
            if not all(arg in available for arg in surrogate.args):
                continue
            emitted_surrogates.add(sur_name)
            progress = True
            exprs = sur_exprs[sur_name]
            if exprs is None:
                continue
            for output_name, out_expr in zip(surrogate.outputs, exprs, strict=True):
                final_expr = (
                    cast(sympy.Expr, out_expr.subs(sym_subs)) if sym_subs else out_expr
                )
                source.append(
                    assignment_template.format(
                        k=name_map[output_name], v=sympy_inline_fn(final_expr)
                    )
                )
                available.add(output_name)
                if output_name in surrogate.stoichiometries:
                    for var_name, factor in surrogate.stoichiometries[
                        output_name
                    ].items():
                        diff_eqs.setdefault(var_name, {})[output_name] = factor


def _generate_model_code(
    model: Model,
    *,
    sized: bool,
    model_fn: str,
    assignment_template: str,
    sympy_inline_fn: Callable[[sympy.Expr], str],
    variables_formatter: Callable[[list[str]], str],
    return_formatter: Callable[[list[str]], str],
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
    imports: list[str] | None = None,
    end: str | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    source: list[str] = []
    # Model components
    variables = model.get_initial_conditions()
    parameters = model.get_parameter_values()
    derived_raw = model.get_raw_derived()
    reactions_raw = model.get_raw_reactions()
    surrogates_raw = model.get_raw_surrogates()

    # Build sanitized name map including surrogate output names so sym_subs covers them
    surrogate_output_names = [
        out for sur in surrogates_raw.values() for out in sur.outputs
    ]
    all_names = (
        list(variables)
        + list(parameters)
        + list(derived_raw)
        + list(reactions_raw)
        + surrogate_output_names
    )
    name_map = {n: valid_identifier(n) for n in all_names}
    sanitized_names = list(name_map.values())
    if len(sanitized_names) != len(set(sanitized_names)):
        _LOGGER.warning(
            "Name sanitization produced duplicate identifiers — "
            "rename model components to avoid collisions."
        )
    # Sympy substitution list: replace symbols with original names by sanitized ones
    sym_subs = [
        (sympy.Symbol(orig), sympy.Symbol(san))
        for orig, san in name_map.items()
        if orig != san
    ]

    # Pre-compute surrogate sympy exprs so we can interleave them correctly
    sur_exprs: dict[str, list[sympy.Expr] | None] = {}
    for sur_name, surrogate in surrogates_raw.items():
        sur_fn = getattr(surrogate, "model", None) or getattr(surrogate, "fn", None)
        if not callable(sur_fn):
            _LOGGER.warning("Skipping surrogate '%s': no callable model/fn", sur_name)
            sur_exprs[sur_name] = None
            continue
        exprs = fn_to_sympy_exprs(sur_fn, sur_name, list_of_symbols(surrogate.args))
        if exprs is None or len(exprs) != len(surrogate.outputs):
            _LOGGER.warning(
                "Skipping surrogate '%s': cannot convert to sympy", sur_name
            )
            sur_exprs[sur_name] = None
            continue
        sur_exprs[sur_name] = exprs

    if imports is not None:
        source.extend(imports)

    if not sized:
        source.append(model_fn)
    else:
        source.append(model_fn.format(n=len(variables)))

    if len(variables) > 0:
        source.append(variables_formatter([name_map[v] for v in variables]))

    # Parameters
    if free_parameters is not None:
        for key in free_parameters:
            parameters.pop(key)
    if len(parameters) > 0:
        source.append(
            "\n".join(
                assignment_template.format(k=name_map[k], v=v)
                for k, v in parameters.items()
            )
        )

    # Emit derived, reactions, and surrogates in dependency order.
    # Before each derived/reaction, flush any surrogate whose args are now satisfied.
    diff_eqs: dict[str, dict[str, Any]] = {}
    available: set[str] = set(variables) | set(parameters) | {"time"}
    emitted_surrogates: set[str] = set()

    for name, derived in derived_raw.items():
        _flush_ready_surrogates(
            source,
            surrogates_raw,
            emitted_surrogates,
            available,
            sur_exprs,
            sym_subs,
            name_map,
            assignment_template,
            sympy_inline_fn,
            diff_eqs,
        )
        expr = custom_fns.get(name)
        if expr is None:
            expr = fn_to_sympy_expr(
                derived.fn,
                origin=name,
                model_args=list_of_symbols(derived.args),
            )
        if expr is None:
            msg = f"Unable to parse fn for derived value '{name}'"
            raise ValueError(msg)
        if sym_subs:
            expr = cast(sympy.Expr, expr.subs(sym_subs))
        source.append(
            assignment_template.format(k=name_map[name], v=sympy_inline_fn(expr))
        )
        available.add(name)

    for name, rxn in reactions_raw.items():
        _flush_ready_surrogates(
            source,
            surrogates_raw,
            emitted_surrogates,
            available,
            sur_exprs,
            sym_subs,
            name_map,
            assignment_template,
            sympy_inline_fn,
            diff_eqs,
        )
        expr = custom_fns.get(name)
        if expr is None:
            try:
                expr = fn_to_sympy_expr(
                    rxn.fn,
                    origin=name,
                    model_args=list_of_symbols(rxn.args),
                )
            except KeyError:
                _LOGGER.warning("Failed to parse %s", name)
        if expr is None:
            msg = f"Unable to parse fn for reaction value '{name}'"
            raise ValueError(msg)
        if sym_subs:
            expr = cast(sympy.Expr, expr.subs(sym_subs))
        source.append(
            assignment_template.format(k=name_map[name], v=sympy_inline_fn(expr))
        )
        available.add(name)
        for var_name, factor in rxn.stoichiometry.items():
            diff_eqs.setdefault(var_name, {})[name] = factor

    # Emit any surrogates whose args only became satisfied after all derived+reactions
    _flush_ready_surrogates(
        source,
        surrogates_raw,
        emitted_surrogates,
        available,
        sur_exprs,
        sym_subs,
        name_map,
        assignment_template,
        sympy_inline_fn,
        diff_eqs,
    )

    for variable, stoich in diff_eqs.items():
        stoich_expr = stoichiometries_to_sympy(origin=variable, stoichs=stoich)
        if sym_subs:
            stoich_expr = cast(sympy.Expr, stoich_expr.subs(sym_subs))
        source.append(
            assignment_template.format(
                k=f"d{name_map[variable]}dt", v=sympy_inline_fn(stoich_expr)
            )
        )

    # Return
    ret_order = [i for i in variables if i in diff_eqs]
    source.append(return_formatter([name_map[v] for v in ret_order]))

    if end is not None:
        source.append(end)

    # print(source)
    return "\n".join(source)


def _generate_components_code(
    model: Model,
    *,
    sized: bool,
    model_fn: str,
    assignment_template: str,
    sympy_inline_fn: Callable[[sympy.Expr], str],
    variables_formatter: Callable[[list[str]], str],
    return_formatter: Callable[[list[str]], str],
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
    imports: list[str] | None = None,
    end: str | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    source: list[str] = []
    variables = model.get_initial_conditions()
    parameters = model.get_parameter_values()
    derived_raw = model.get_raw_derived()
    reactions_raw = model.get_raw_reactions()
    readouts_raw = model.get_raw_readouts()
    surrogates_raw = model.get_raw_surrogates()
    # _cache is populated as a side-effect of the get_raw_* calls above
    if (cache_obj := model._cache) is None:  # noqa: SLF001
        cache_obj = model._create_cache()  # noqa: SLF001

    surrogate_output_names_all = [
        out for sur in surrogates_raw.values() for out in sur.outputs
    ]
    all_component_names = (
        set(derived_raw)
        | set(reactions_raw)
        | set(readouts_raw)
        | set(surrogate_output_names_all)
    )

    if outputs is not None:
        unknown = set(outputs) - all_component_names
        if unknown:
            msg = f"Unknown output components: {sorted(unknown)}"
            raise ValueError(msg)
        leaf_names = set(variables) | set(parameters) | {"time"}
        needed = _collect_transitive_deps(
            set(outputs), derived_raw, reactions_raw, readouts_raw, leaf_names
        )
        # Also include surrogate outputs explicitly requested
        needed_surrogate_outputs = set(outputs) & set(surrogate_output_names_all)
    else:
        needed = all_component_names
        needed_surrogate_outputs = set(surrogate_output_names_all)

    # Use dyn_order from cache for correct topological ordering of derived + reactions.
    ordered_dyn = [n for n in cache_obj.dyn_order if n in needed]
    ordered_readouts = [k for k in readouts_raw if k in needed]

    # Build sanitized name map including surrogate output names for correct sym_subs
    all_names = (
        list(variables)
        + list(parameters)
        + list(derived_raw)
        + list(reactions_raw)
        + list(readouts_raw)
        + surrogate_output_names_all
    )
    name_map = {n: valid_identifier(n) for n in all_names}
    sanitized_names = list(name_map.values())
    if len(sanitized_names) != len(set(sanitized_names)):
        msg = (
            "Name sanitization produced duplicate identifiers — "
            "rename model components to avoid collisions."
        )
        raise ValueError(msg)
    sym_subs = [
        (sympy.Symbol(orig), sympy.Symbol(san))
        for orig, san in name_map.items()
        if orig != san
    ]

    # Pre-compute surrogate sympy exprs
    sur_exprs: dict[str, list[sympy.Expr] | None] = {}
    for sur_name, surrogate in surrogates_raw.items():
        sur_fn = getattr(surrogate, "model", None) or getattr(surrogate, "fn", None)
        if not callable(sur_fn):
            _LOGGER.warning("Skipping surrogate '%s': no callable model/fn", sur_name)
            sur_exprs[sur_name] = None
            continue
        exprs = fn_to_sympy_exprs(sur_fn, sur_name, list_of_symbols(surrogate.args))
        if exprs is None or len(exprs) != len(surrogate.outputs):
            _LOGGER.warning(
                "Skipping surrogate '%s': cannot convert to sympy", sur_name
            )
            sur_exprs[sur_name] = None
            continue
        sur_exprs[sur_name] = exprs

    if imports is not None:
        source.extend(imports)

    if not sized:
        source.append(model_fn)
    else:
        source.append(model_fn.format(n=len(variables)))

    if len(variables) > 0:
        source.append(variables_formatter([name_map[v] for v in variables]))

    if free_parameters is not None:
        for key in free_parameters:
            parameters.pop(key)
    if len(parameters) > 0:
        source.append(
            "\n".join(
                assignment_template.format(k=name_map[k], v=v)
                for k, v in parameters.items()
            )
        )

    # Emit derived, reactions, and surrogates interleaved in dependency order.
    # Surrogates are flushed as soon as all their args are available.
    available: set[str] = set(variables) | set(parameters) | {"time"}
    emitted_surrogates: set[str] = set()
    emitted_surrogate_outputs: list[str] = []

    def _flush_ready_surrogates() -> None:
        progress = True
        while progress:
            progress = False
            for sur_name, surrogate in surrogates_raw.items():
                if sur_name in emitted_surrogates:
                    continue
                if not all(arg in available for arg in surrogate.args):
                    continue
                emitted_surrogates.add(sur_name)
                progress = True
                exprs = sur_exprs[sur_name]
                if exprs is None:
                    continue
                for output_name, out_expr in zip(surrogate.outputs, exprs, strict=True):
                    if output_name not in needed_surrogate_outputs:
                        available.add(output_name)
                        continue
                    final_expr = (
                        cast(sympy.Expr, out_expr.subs(sym_subs))
                        if sym_subs
                        else out_expr
                    )
                    source.append(
                        assignment_template.format(
                            k=name_map[output_name], v=sympy_inline_fn(final_expr)
                        )
                    )
                    available.add(output_name)
                    emitted_surrogate_outputs.append(name_map[output_name])

    for name in ordered_dyn:
        _flush_ready_surrogates()
        if name in derived_raw:
            component = derived_raw[name]
            expr = custom_fns.get(name)
            if expr is None:
                expr = fn_to_sympy_expr(
                    component.fn,
                    origin=name,
                    model_args=list_of_symbols(component.args),
                )
            if expr is None:
                msg = f"Unable to parse fn for derived value '{name}'"
                raise ValueError(msg)
        else:
            component = reactions_raw[name]
            expr = custom_fns.get(name)
            if expr is None:
                try:
                    expr = fn_to_sympy_expr(
                        component.fn,
                        origin=name,
                        model_args=list_of_symbols(component.args),
                    )
                except KeyError:
                    _LOGGER.warning("Failed to parse %s", name)
            if expr is None:
                msg = f"Unable to parse fn for reaction value '{name}'"
                raise ValueError(msg)
        if sym_subs:
            expr = cast(sympy.Expr, expr.subs(sym_subs))
        source.append(
            assignment_template.format(k=name_map[name], v=sympy_inline_fn(expr))
        )
        available.add(name)

    # Flush remaining surrogates (those only depending on derived/reactions)
    _flush_ready_surrogates()

    # Readouts always come after derived + reactions + surrogates
    for name in ordered_readouts:
        readout = readouts_raw[name]
        expr = custom_fns.get(name)
        if expr is None:
            expr = fn_to_sympy_expr(
                readout.fn,
                origin=name,
                model_args=list_of_symbols(readout.args),
            )
        if expr is None:
            msg = f"Unable to parse fn for readout value '{name}'"
            raise ValueError(msg)
        if sym_subs:
            expr = cast(sympy.Expr, expr.subs(sym_subs))
        source.append(
            assignment_template.format(k=name_map[name], v=sympy_inline_fn(expr))
        )

    # Return only the explicitly requested outputs (in requested order),
    # or everything when outputs is None (maintaining emission order).
    if outputs is not None:
        returns = [name_map[n] for n in outputs]
    else:
        returns = [
            name_map[n] for n in ordered_dyn + ordered_readouts
        ] + emitted_surrogate_outputs

    source.append(return_formatter(returns))

    if end is not None:
        source.append(end)

    return "\n".join(source)


def generate_model_code_py(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    *,
    typed: bool = True,
) -> str:
    """Transform the model into a python function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        Python source code of the generated model function

    """
    if free_parameters is None:
        model_fn = (
            "def model(time: float, variables: Iterable[float]) -> Iterable[float]:"
        )
    else:
        args = ", ".join(f"{valid_identifier(k)}: float" for k in free_parameters)
        model_fn = f"def model(time: float, variables: Iterable[float], {args}) -> Iterable[float]:"

    return _generate_model_code(
        model,
        imports=[
            "import math\n",
            "from collections.abc import Iterable\n",
        ],
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k}: float = {v}" if typed else "    {k} = {v}",
        sympy_inline_fn=sympy_to_inline_py,
        variables_formatter=lambda vs: f"    {', '.join(vs)} = variables",
        return_formatter=lambda vs: (
            f"    return {', '.join(f'd{v}dt' for v in vs) or '()'}"
        ),
        end=None,
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_py(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
    *,
    typed: bool = True,
) -> str:
    """Transform the model components into a python function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        Python source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "def components(time: float, variables: Iterable[float]) -> Iterable[float]:"
    else:
        args = ", ".join(f"{valid_identifier(k)}: float" for k in free_parameters)
        model_fn = f"def components(time: float, variables: Iterable[float], {args}) -> Iterable[float]:"

    return _generate_components_code(
        model,
        imports=[
            "import math\n",
            "from collections.abc import Iterable\n",
        ],
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k}: float = {v}" if typed else "    {k} = {v}",
        sympy_inline_fn=sympy_to_inline_py,
        variables_formatter=lambda vs: f"    {', '.join(vs)} = variables",
        return_formatter=lambda vs: f"    return {', '.join(vs) or '()'}",
        end=None,
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_ts(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a typescript function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        TypeScript source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "function model(time: number, variables: number[]) {"
    else:
        args = ", ".join(f"{valid_identifier(k)}: number" for k in free_parameters)
        model_fn = f"function model(time: number, variables: number[], {args}) {{"

    return _generate_model_code(
        model,
        imports=[],
        sized=False,
        model_fn=model_fn,
        assignment_template="    let {k}: number = {v};",
        sympy_inline_fn=sympy_to_inline_js,
        variables_formatter=lambda vs: f"    let [{', '.join(vs)}] = variables;",
        return_formatter=lambda vs: (
            f"    return [{', '.join(f'd{v}dt' for v in vs) or '()'}];"
        ),
        end="};",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_ts(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a TypeScript function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        TypeScript source code of the generated model components function

    """
    if free_parameters is None:
        model_fn = "function components(time: number, variables: number[]) {"
    else:
        args = ", ".join(f"{valid_identifier(k)}: number" for k in free_parameters)
        model_fn = f"function components(time: number, variables: number[], {args}) {{"

    return _generate_components_code(
        model,
        imports=[],
        sized=False,
        model_fn=model_fn,
        assignment_template="    let {k}: number = {v};",
        sympy_inline_fn=sympy_to_inline_js,
        variables_formatter=lambda vs: f"    let [{', '.join(vs)}] = variables;",
        return_formatter=lambda vs: f"    return [{', '.join(vs) or '()'}];",
        end="};",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_rs(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a rust function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        Rust source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "fn model(time: f64, variables: &[f64; {n}]) -> [f64; {n}] {{"
    else:
        args = ", ".join(f"{valid_identifier(k)}: f64" for k in free_parameters)
        model_fn = f"fn model(time: f64, variables: &[f64; {{n}}], {args}) -> [f64; {{n}}] {{{{"

    return _generate_model_code(
        model,
        imports=None,
        sized=True,
        model_fn=model_fn,
        assignment_template="    let {k}: f64 = {v};",
        sympy_inline_fn=sympy_to_inline_rust,
        variables_formatter=lambda vs: f"    let [{', '.join(vs)}] = *variables;",
        return_formatter=lambda vs: (
            f"    return [{', '.join(f'd{v}dt' for v in vs) or '()'}]"
        ),
        end="}",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_rs(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a Rust function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        Rust source code of the generated model components function

    """
    n_vars = len(model.get_initial_conditions())
    n_out = (
        len(outputs)
        if outputs is not None
        else (
            len(model.get_raw_derived())
            + len(model.get_raw_reactions())
            + len(model.get_raw_readouts())
        )
    )
    if free_parameters is None:
        model_fn = f"fn components(time: f64, variables: &[f64; {n_vars}]) -> [f64; {n_out}] {{"
    else:
        args = ", ".join(f"{valid_identifier(k)}: f64" for k in free_parameters)
        model_fn = f"fn components(time: f64, variables: &[f64; {n_vars}], {args}) -> [f64; {n_out}] {{"

    return _generate_components_code(
        model,
        imports=None,
        sized=False,
        model_fn=model_fn,
        assignment_template="    let {k}: f64 = {v};",
        sympy_inline_fn=sympy_to_inline_rust,
        variables_formatter=lambda vs: f"    let [{', '.join(vs)}] = *variables;",
        return_formatter=lambda vs: f"    return [{', '.join(vs) or '()'}]",
        end="}",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_jl(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a julia function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        Julia source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "function model(time, variables)"
    else:
        args = ", ".join(valid_identifier(k) for k in free_parameters)
        model_fn = f"function model(time, variables, {args})"

    return _generate_model_code(
        model,
        imports=None,
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k} = {v}",
        sympy_inline_fn=sympy_to_inline_julia,
        variables_formatter=lambda vs: f"    {', '.join(vs)} = variables",
        return_formatter=lambda vs: (
            f"    return [{', '.join(f'd{v}dt' for v in vs) or '()'}]"
        ),
        end="end",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_jl(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a Julia function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        Julia source code of the generated model components function

    """
    if free_parameters is None:
        model_fn = "function components(time, variables)"
    else:
        args = ", ".join(valid_identifier(k) for k in free_parameters)
        model_fn = f"function components(time, variables, {args})"

    return _generate_components_code(
        model,
        imports=None,
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k} = {v}",
        sympy_inline_fn=sympy_to_inline_julia,
        variables_formatter=lambda vs: f"    {', '.join(vs)} = variables",
        return_formatter=lambda vs: f"    return [{', '.join(vs) or '()'}]",
        end="end",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_c(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a C99 function, inlining the function calls.

    The generated function writes derivatives into an output pointer ``dydt``
    rather than returning an array, since C does not support returning arrays.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        C99 source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "void model(double t, const double *variables, double *dydt) {"
    else:
        args_typed = ", ".join(f"double {valid_identifier(k)}" for k in free_parameters)
        model_fn = f"void model(double t, const double *variables, double *dydt, {args_typed}) {{"

    return _generate_model_code(
        model,
        imports=["#include <math.h>", ""],
        sized=False,
        model_fn=model_fn,
        assignment_template="    double {k} = {v};",
        sympy_inline_fn=sympy_to_inline_c,
        variables_formatter=lambda vs: "\n".join(
            f"    double {v} = variables[{i}];" for i, v in enumerate(vs)
        ),
        return_formatter=lambda vs: "\n".join(
            f"    dydt[{i}] = d{v}dt;" for i, v in enumerate(vs)
        ),
        end="}",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_c(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a C99 function, inlining the function calls.

    The generated function writes derived values and reaction rates into an
    output pointer ``out`` rather than returning an array.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        C99 source code of the generated model components function

    """
    if free_parameters is None:
        model_fn = "void components(double t, const double *variables, double *out) {"
    else:
        args_typed = ", ".join(f"double {valid_identifier(k)}" for k in free_parameters)
        model_fn = f"void components(double t, const double *variables, double *out, {args_typed}) {{"

    return _generate_components_code(
        model,
        imports=["#include <math.h>", ""],
        sized=False,
        model_fn=model_fn,
        assignment_template="    double {k} = {v};",
        sympy_inline_fn=sympy_to_inline_c,
        variables_formatter=lambda vs: "\n".join(
            f"    double {v} = variables[{i}];" for i, v in enumerate(vs)
        ),
        return_formatter=lambda vs: "\n".join(
            f"    out[{i}] = {v};" for i, v in enumerate(vs)
        ),
        end="}",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_cpp(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a c++ function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        C++ source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "std::array<double, {n}> model(double time, const std::array<double, {n}>& variables) {{"
    else:
        args_typed = ", ".join(f"double {valid_identifier(k)}" for k in free_parameters)
        model_fn = f"std::array<double, {{n}}> model(double time, const std::array<double, {{n}}>& variables, {args_typed}) {{{{"

    return _generate_model_code(
        model,
        sized=True,
        model_fn=model_fn,
        assignment_template="    double {k} = {v};",
        sympy_inline_fn=sympy_to_inline_cxx,
        variables_formatter=lambda vs: f"    const auto [{', '.join(vs)}] = variables;",
        return_formatter=lambda vs: (
            f"    return {{{', '.join(f'd{v}dt' for v in vs) or '()'}}};"
        ),
        imports=[
            "#include <array>",
            "#include <cmath>",
            "",
        ],
        end="}",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_cpp(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a C++ function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        C++ source code of the generated model components function

    """
    n_vars = len(model.get_initial_conditions())
    n_out = (
        len(outputs)
        if outputs is not None
        else (
            len(model.get_raw_derived())
            + len(model.get_raw_reactions())
            + len(model.get_raw_readouts())
        )
    )
    if free_parameters is None:
        model_fn = f"std::array<double, {n_out}> components(double time, const std::array<double, {n_vars}>& variables) {{"
    else:
        args_typed = ", ".join(f"double {valid_identifier(k)}" for k in free_parameters)
        model_fn = f"std::array<double, {n_out}> components(double time, const std::array<double, {n_vars}>& variables, {args_typed}) {{"

    return _generate_components_code(
        model,
        imports=[
            "#include <array>",
            "#include <cmath>",
            "",
        ],
        sized=False,
        model_fn=model_fn,
        assignment_template="    double {k} = {v};",
        sympy_inline_fn=sympy_to_inline_cxx,
        variables_formatter=lambda vs: f"    const auto [{', '.join(vs)}] = variables;",
        return_formatter=lambda vs: f"    return {{{', '.join(vs) or '()'}}};",
        end="}",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_code_matlab(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a MATLAB/Octave function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments

    Returns
    -------
    str
        MATLAB/Octave source code of the generated model function

    """
    if free_parameters is None:
        model_fn = "function dydt = model(t, variables)"
    else:
        args = ", ".join(valid_identifier(k) for k in free_parameters)
        model_fn = f"function dydt = model(t, variables, {args})"

    return _generate_model_code(
        model,
        imports=None,
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k} = {v};",
        sympy_inline_fn=sympy_to_inline_matlab,
        variables_formatter=lambda vs: (
            f"    [{', '.join(vs)}] = num2cell(variables){{:}};"
        ),
        return_formatter=lambda vs: (
            f"    dydt = [{', '.join(f'd{v}dt' for v in vs) or '()'}]';"
        ),
        end="end",
        free_parameters=free_parameters,
        custom_fns={} if custom_fns is None else custom_fns,
    )


def generate_model_components_matlab(
    model: Model,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    free_parameters: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:
    """Transform the model components into a MATLAB/Octave function, inlining the function calls.

    Parameters
    ----------
    model
        Model to generate code for
    custom_fns
        Optional custom sympy expressions to substitute for functions
    free_parameters
        Optional list of parameter names to expose as function arguments
    outputs
        Optional subset of component names to emit. All transitive dependencies
        are included automatically. When None all components are emitted.

    Returns
    -------
    str
        MATLAB/Octave source code of the generated model components function

    """
    if free_parameters is None:
        model_fn = "function out = components(t, variables)"
    else:
        args = ", ".join(valid_identifier(k) for k in free_parameters)
        model_fn = f"function out = components(t, variables, {args})"

    return _generate_components_code(
        model,
        imports=None,
        sized=False,
        model_fn=model_fn,
        assignment_template="    {k} = {v};",
        sympy_inline_fn=sympy_to_inline_matlab,
        variables_formatter=lambda vs: (
            f"    [{', '.join(vs)}] = num2cell(variables){{:}};"
        ),
        return_formatter=lambda vs: f"    out = [{', '.join(vs) or '()'}];",
        end="end",
        free_parameters=free_parameters,
        outputs=outputs,
        custom_fns={} if custom_fns is None else custom_fns,
    )
