"""Generate MxlWeb Svelte page code from a model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import sympy

from mxlpy.meta.codegen_model import _to_valid_identifier
from mxlpy.meta.source_tools import fn_to_sympy, fn_to_sympy_outputs
from mxlpy.meta.sympy_tools import list_of_symbols
from mxlpy.types import Derived, InitialAssignment

if TYPE_CHECKING:
    from mxlpy.model import Model

__all__ = ["generate_mxlweb_page"]

_LOGGER = logging.getLogger(__name__)

# Sympy function type → MxlWeb TS class name (unary)
_UNARY_FN_MAP: list[tuple[type, str]] = [
    (sympy.exp, "Exp"),
    (sympy.log, "Ln"),
    (sympy.sin, "Sin"),
    (sympy.cos, "Cos"),
    (sympy.tan, "Tan"),
    (sympy.asin, "Asin"),
    (sympy.acos, "Acos"),
    (sympy.atan, "Atan"),
    (sympy.Abs, "Abs"),
    (sympy.floor, "Floor"),
    (sympy.ceiling, "Ceiling"),
    (sympy.sinh, "Sinh"),
    (sympy.cosh, "Cosh"),
    (sympy.tanh, "Tanh"),
    (sympy.asinh, "ArcSinh"),
    (sympy.acosh, "ArcCosh"),
    (sympy.atanh, "ArcTanh"),
]

# Sympy relational/logical type → MxlWeb TS class name (n-ary)
_RELATIONAL_MAP: list[tuple[type, str]] = [
    (sympy.Eq, "Eq"),
    (sympy.Ne, "NotEqual"),
    (sympy.Gt, "GreaterThan"),
    (sympy.Ge, "GreaterEqual"),
    (sympy.Lt, "LessThan"),
    (sympy.Le, "LessEqual"),
    (sympy.And, "And"),
    (sympy.Or, "Or"),
    (sympy.Not, "Not"),
]


def _sympy_to_mxlweb(expr: sympy.Expr, used: set[str]) -> str:
    """Recursively convert a sympy expression to a MxlWeb TypeScript AST string.

    Parameters
    ----------
    expr
        Sympy expression to convert
    used
        Set collecting TS class names used, for building the import statement

    Returns
    -------
    str
        TypeScript expression constructing the equivalent MxlWeb AST node

    """
    # Numbers
    if isinstance(expr, sympy.Number):
        used.add("Num")
        return f"new Num({float(expr)})"

    # Named symbol
    if isinstance(expr, sympy.Symbol):
        used.add("Name")
        return f'new Name("{expr.name}")'

    # Addition
    if isinstance(expr, sympy.Add):
        used.add("Add")
        children = ", ".join(_sympy_to_mxlweb(a, used) for a in expr.args)
        return f"new Add([{children}])"

    # Multiplication — handle negation and division
    if isinstance(expr, sympy.Mul):
        coeff, rest_factors = expr.as_coeff_mul()

        # Separate denominator factors (Pow(x, -n)) from rest_factors first,
        # then apply sign — this avoids Mul([Divide([Num(1), f]), g]) patterns.
        numer: list[sympy.Expr] = []
        denom: list[sympy.Expr] = []
        for arg in rest_factors:
            if isinstance(arg, sympy.Pow):
                arg_exp = cast(sympy.Expr, arg.exp)
                if isinstance(arg_exp, sympy.Number) and arg_exp.is_negative:
                    neg_exp = cast(sympy.Expr, -arg_exp)
                    base = cast(sympy.Expr, arg.base)
                    denom.append(
                        base
                        if neg_exp == sympy.Integer(1)
                        else sympy.Pow(base, neg_exp)
                    )
                    continue
            numer.append(cast(sympy.Expr, arg))

        abs_coeff = cast(sympy.Expr, -coeff if coeff.is_negative else coeff)
        if abs_coeff != sympy.Integer(1):
            used.add("Num")
            numer = [cast(sympy.Expr, sympy.Number(abs_coeff)), *numer]

        if denom:
            used.add("Divide")
            n_expr: sympy.Expr = (
                sympy.Mul(*numer)
                if len(numer) > 1
                else (numer[0] if numer else sympy.Integer(1))
            )
            d_expr: sympy.Expr = sympy.Mul(*denom) if len(denom) > 1 else denom[0]
            inner: str = (
                f"new Divide([{_sympy_to_mxlweb(n_expr, used)}, "
                f"{_sympy_to_mxlweb(d_expr, used)}])"
            )
        elif len(numer) == 1:
            inner = _sympy_to_mxlweb(numer[0], used)
        else:
            used.add("Mul")
            children = ", ".join(_sympy_to_mxlweb(f, used) for f in numer)
            inner = f"new Mul([{children}])"

        if coeff.is_negative:
            used.add("Minus")
            return f"new Minus([{inner}])"
        return inner

    # Powers
    if isinstance(expr, sympy.Pow):
        base = cast(sympy.Expr, expr.args[0])
        exp = cast(sympy.Expr, expr.args[1])
        if exp == sympy.Rational(1, 2):
            used.add("Sqrt")
            used.add("Num")
            return f"new Sqrt({_sympy_to_mxlweb(base, used)}, new Num(2))"
        if exp == sympy.Integer(-1):
            used.add("Divide")
            used.add("Num")
            return f"new Divide([new Num(1), {_sympy_to_mxlweb(base, used)}])"
        used.add("Pow")
        return f"new Pow({_sympy_to_mxlweb(base, used)}, {_sympy_to_mxlweb(exp, used)})"

    # Unary functions
    for sympy_type, ts_name in _UNARY_FN_MAP:
        if isinstance(expr, sympy_type):
            used.add(ts_name)
            child = cast(sympy.Expr, expr.args[0])
            return f"new {ts_name}({_sympy_to_mxlweb(child, used)})"

    # N-ary functions
    if isinstance(expr, sympy.Max):
        used.add("Max")
        children = ", ".join(
            _sympy_to_mxlweb(cast(sympy.Expr, a), used) for a in expr.args
        )
        return f"new Max([{children}])"

    if isinstance(expr, sympy.Min):
        used.add("Min")
        children = ", ".join(
            _sympy_to_mxlweb(cast(sympy.Expr, a), used) for a in expr.args
        )
        return f"new Min([{children}])"

    # Piecewise: sympy args are ((val, cond), ...) pairs.
    # Last pair with cond=True becomes the "otherwise" branch (value only, no condition).
    if isinstance(expr, sympy.Piecewise):
        used.add("Piecewise")
        pieces: list[str] = []
        for pair in expr.args:
            val = cast(sympy.Expr, pair.args[0])
            cond = cast(sympy.Expr, pair.args[1])
            pieces.append(_sympy_to_mxlweb(val, used))
            if cond is not sympy.true:
                pieces.append(_sympy_to_mxlweb(cond, used))
        return f"new Piecewise([{', '.join(pieces)}])"

    # Relational and logical operators
    for sympy_type, ts_name in _RELATIONAL_MAP:
        if isinstance(expr, sympy_type):
            used.add(ts_name)
            children = ", ".join(
                _sympy_to_mxlweb(cast(sympy.Expr, a), used) for a in expr.args
            )
            return f"new {ts_name}([{children}])"

    msg = f"Cannot convert sympy type {type(expr).__name__} ({expr!r}) to MxlWeb AST"
    raise ValueError(msg)


def _fn_to_mxlweb(
    fn: object,
    name: str,
    args: list[str],
    used: set[str],
    sym_subs: list[tuple[sympy.Expr, sympy.Expr]] | None = None,
) -> str:
    """Convert a rate/derived function to a MxlWeb AST TS string.

    Parameters
    ----------
    fn
        Callable to convert via sympy
    name
        Component name (used in error messages)
    args
        Argument names for the function
    used
        Set collecting used TS class names
    sym_subs
        Optional substitution list to sanitize symbol names

    Returns
    -------
    str
        TypeScript expression for the MxlWeb AST node

    """
    expr = fn_to_sympy(fn, origin=name, model_args=list_of_symbols(args))  # type: ignore[arg-type]
    if expr is None:
        msg = f"Cannot convert '{name}' to a sympy expression — the function may use unsupported constructs (closures, side effects, non-arithmetic operations)"
        raise ValueError(msg)
    if sym_subs:
        expr = cast(sympy.Expr, expr.subs(sym_subs))
    return _sympy_to_mxlweb(expr, used)


def generate_mxlweb_page(
    model: Model,
    name: str = "Model",
    description: str = "",
    t_end: float = 1e-3,
) -> str:
    """Generate a MxlWeb Svelte page from a model.

    The output is a complete ``+page.svelte`` file that can be dropped into
    ``src/mxlweb/routes/<route-name>/+page.svelte``.  Rate functions and
    derived values are inlined as MxlWeb AST nodes.

    Parameters
    ----------
    model
        Model to generate the page for
    name
        Display name shown in the page heading
    description
        Optional description paragraph rendered below the heading
    t_end
        Default simulation end time

    Returns
    -------
    str
        Complete Svelte page source code

    """
    used: set[str] = set()
    lines: list[str] = []

    variables_raw = model.get_raw_variables()
    variables = model.get_initial_conditions()
    parameters = model.get_parameter_values()
    derived_raw = model.get_raw_derived()
    reactions_raw = model.get_raw_reactions()
    surrogates_raw = model.get_raw_surrogates()

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
    name_map = {n: _to_valid_identifier(n) for n in all_names}
    sym_subs: list[tuple[sympy.Expr, sympy.Expr]] = [
        (sympy.Symbol(orig), sympy.Symbol(san))
        for orig, san in name_map.items()
        if orig != san
    ]

    def _tex(k: str) -> str:
        return k.replace("_", r"\_")

    # Collect component strings (populates `used` as a side-effect)
    param_lines: list[str] = []
    for k, v in parameters.items():
        param_lines.append(
            f'      .addParameter("{name_map[k]}", {{ value: {v!r}, texName: {_tex(k)!r}}})'
        )

    var_lines: list[str] = []
    for k in variables:
        raw_var = variables_raw[k]
        if isinstance(raw_var.initial_value, InitialAssignment):
            try:
                fn_ts = _fn_to_mxlweb(
                    raw_var.initial_value.fn,
                    k,
                    raw_var.initial_value.args,
                    used,
                    sym_subs,
                )
                var_lines.append(
                    f'      .addVariable("{name_map[k]}", {{ value: {fn_ts}, texName: {_tex(k)!r} }})'
                )
            except (ValueError, KeyError) as exc:
                _LOGGER.warning(
                    "Variable '%s': cannot convert initial assignment, using numeric fallback: %s",
                    k,
                    exc,
                )
                var_lines.append(
                    f'      .addVariable("{name_map[k]}", {{ value: {variables[k]!r}, texName: {_tex(k)!r} }})'
                )
        else:
            var_lines.append(
                f'      .addVariable("{name_map[k]}", {{ value: {variables[k]!r}, texName: {_tex(k)!r} }})'
            )

    assign_lines: list[str] = []
    for k, der in derived_raw.items():
        try:
            fn_ts = _fn_to_mxlweb(der.fn, k, der.args, used, sym_subs)
        except (ValueError, KeyError) as exc:
            _LOGGER.warning("Skipping derived '%s': %s", k, exc)
            continue
        assign_lines.append(
            f'      .addAssignment("{name_map[k]}", {{ fn: {fn_ts}, texName: {_tex(k)!r}}})'
        )

    rxn_lines: list[str] = []
    for k, rxn in reactions_raw.items():
        try:
            fn_ts = _fn_to_mxlweb(rxn.fn, k, rxn.args, used, sym_subs)
        except (ValueError, KeyError) as exc:
            _LOGGER.warning("Skipping reaction '%s': %s", k, exc)
            continue

        stoich_parts: list[str] = []
        for var_name, factor in rxn.stoichiometry.items():
            san_var = name_map.get(var_name, var_name)
            if isinstance(factor, Derived):
                try:
                    stoich_ts = _fn_to_mxlweb(
                        factor.fn, f"{k}_stoich_{var_name}", factor.args, used, sym_subs
                    )
                except (ValueError, KeyError) as exc:
                    _LOGGER.warning(
                        "Reaction '%s': cannot convert derived stoichiometry for '%s': %s",
                        k,
                        var_name,
                        exc,
                    )
                    continue
                stoich_parts.append(f'{{ name: "{san_var}", value: {stoich_ts} }}')
            else:
                used.add("Num")
                stoich_parts.append(
                    f'{{ name: "{san_var}", value: new Num({float(factor)}) }}'
                )

        stoich_ts = f"[{', '.join(stoich_parts)}]"
        rxn_lines.append(
            f'      .addReaction("{name_map[k]}", {{\n'
            f"        fn: {fn_ts},\n"
            f"        stoichiometry: {stoich_ts},\n"
            f"        texName: {_tex(k)!r},\n"
            f"      }})"
        )

    # Surrogates — each output becomes either a reaction or an assignment
    for k, surrogate in surrogates_raw.items():
        model_fn = getattr(surrogate, "model", None) or getattr(surrogate, "fn", None)
        if not callable(model_fn):
            _LOGGER.warning("Skipping surrogate '%s': no callable model/fn", k)
            continue

        exprs = fn_to_sympy_outputs(model_fn, k, list_of_symbols(surrogate.args))
        if exprs is None or len(exprs) != len(surrogate.outputs):
            _LOGGER.warning(
                "Skipping surrogate '%s': cannot convert to sympy (%d outputs expected, got %s)",
                k,
                len(surrogate.outputs),
                len(exprs) if exprs is not None else "None",
            )
            continue

        for output_name, out_expr in zip(surrogate.outputs, exprs, strict=True):
            san_output = name_map.get(output_name, output_name)
            final_expr = (
                cast(sympy.Expr, out_expr.subs(sym_subs)) if sym_subs else out_expr
            )
            try:
                fn_ts = _sympy_to_mxlweb(final_expr, used)
            except (ValueError, KeyError) as exc:
                _LOGGER.warning(
                    "Surrogate '%s' output '%s': cannot convert: %s",
                    k,
                    output_name,
                    exc,
                )
                continue

            stoich_dict = surrogate.stoichiometries.get(output_name)
            if stoich_dict is not None:
                stoich_parts: list[str] = []
                for var_name, factor in stoich_dict.items():
                    san_var = name_map.get(var_name, var_name)
                    if isinstance(factor, Derived):
                        try:
                            stoich_ts = _fn_to_mxlweb(
                                factor.fn,
                                f"{k}_{output_name}_stoich_{var_name}",
                                factor.args,
                                used,
                                sym_subs,
                            )
                        except (ValueError, KeyError) as exc:
                            _LOGGER.warning(
                                "Surrogate '%s' output '%s': cannot convert derived stoich for '%s': %s",
                                k,
                                output_name,
                                var_name,
                                exc,
                            )
                            continue
                        stoich_parts.append(
                            f'{{ name: "{san_var}", value: {stoich_ts} }}'
                        )
                    else:
                        used.add("Num")
                        stoich_parts.append(
                            f'{{ name: "{san_var}", value: new Num({float(factor)}) }}'
                        )
                rxn_stoich_ts = f"[{', '.join(stoich_parts)}]"
                rxn_lines.append(
                    f'      .addReaction("{san_output}", {{\n'
                    f"        fn: {fn_ts},\n"
                    f"        stoichiometry: {rxn_stoich_ts},\n"
                    f"        texName: {_tex(k)!r},\n"
                    f"      }})"
                )
            else:
                assign_lines.append(
                    f'      .addAssignment("{san_output}", {{ fn: {fn_ts}, texName: {_tex(k)!r} }})'
                )

    # Build import list from collected class names
    mathml_import_str = ", ".join(sorted(used))

    builder_chain = "\n".join(param_lines + var_lines + assign_lines + rxn_lines)

    lines.append('<script lang="ts">')
    lines.append('  import type { Analyses } from "$lib";')
    lines.append(f'  import {{ {mathml_import_str} }} from "$lib/mathml";')
    lines.append(
        '  import AnalysesDashboard from "$lib/model-editor/AnalysesDashboard.svelte";'
    )
    lines.append('  import { ModelBuilder } from "$lib/model-editor/modelBuilder";')
    lines.append("")
    lines.append("  function initModel(): ModelBuilder {")
    lines.append("    return new ModelBuilder()")
    lines.append(builder_chain)
    lines.append("  }")
    lines.append("")
    lines.append("  let analyses: Analyses = $state([")
    lines.append("    {")
    lines.append('      type: "simulation" as const,')
    lines.append("      id: 0,")
    lines.append("      idx: 0,")
    lines.append('      title: "Simulation",')
    lines.append("      col: 1,")
    lines.append("      span: 6,")
    lines.append(f"      tEnd: {t_end},")
    lines.append("      xMin: undefined,")
    lines.append("      xMax: undefined,")
    lines.append("      yMin: undefined,")
    lines.append("      yMax: undefined,")
    lines.append("      timeoutInSeconds: 20,")
    lines.append("    },")
    lines.append("  ]);")
    lines.append("</script>")
    lines.append("")
    lines.append("<AnalysesDashboard")
    lines.append(f'  name={{"{name}"}}')
    lines.append("  initModel={initModel}")
    lines.append("  bind:analyses={analyses}")
    lines.append(">")
    if description:
        lines.append(f"  <p>{description}</p>")
    lines.append("</AnalysesDashboard>")

    return "\n".join(lines)
