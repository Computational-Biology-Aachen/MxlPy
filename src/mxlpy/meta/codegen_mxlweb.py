"""Generate MxlWeb Svelte page code from a model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import sympy

from mxlpy.meta.source_tools import fn_to_sympy, fn_to_sympy_outputs
from mxlpy.meta.sympy_tools import list_of_symbols, sympy_to_inline_mxlweb
from mxlpy.meta.utils import _to_valid_identifier
from mxlpy.types import Derived, InitialAssignment

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model

__all__ = ["generate_mode_code_mxlweb", "generate_mxlweb_page"]

_LOGGER = logging.getLogger(__name__)


def _fn_to_mxlweb(
    fn: Callable,
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
        msg = (
            f"Cannot convert '{name}' to a sympy expression - "
            "the function may use unsupported constructs "
            "(closures, side effects, non-arithmetic operations)"
        )
        raise ValueError(msg)
    if sym_subs:
        expr = cast(sympy.Expr, expr.subs(sym_subs))
    return sympy_to_inline_mxlweb(expr, used)


def generate_mode_code_mxlweb(model: Model) -> tuple[str, str]:
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
                fn_ts = sympy_to_inline_mxlweb(final_expr, used)
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

    model_builder_str = "\n".join(
        (
            "  function initModel(): ModelBuilder {",
            "\n".join(param_lines + var_lines + assign_lines + rxn_lines),
            "    return new ModelBuilder()",
            "  }",
        )
    )

    return (mathml_import_str, model_builder_str)


def generate_mxlweb_page(
    model: Model,
    name: str = "Model",
    description: str = "",
    t_end: float = 100,
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
    mathml_import_str, builder_chain = generate_mode_code_mxlweb(model)

    return "\n".join(
        [
            '<script lang="ts">',
            '  import type { Analyses } from "$lib";',
            f'  import {{ {mathml_import_str} }} from "$lib/mathml";',
            '  import AnalysesDashboard from "$lib/model-editor/AnalysesDashboard.svelte";',
            '  import { ModelBuilder } from "$lib/model-editor/modelBuilder";',
            "",
            builder_chain,
            "",
            "  let analyses: Analyses = $state([",
            "    {",
            '      type: "simulation" as const,',
            "      id: 0,",
            "      idx: 0,",
            '      title: "Simulation",',
            "      col: 1,",
            "      span: 6,",
            f"      tEnd: {t_end},",
            "      xMin: undefined,",
            "      xMax: undefined,",
            "      yMin: undefined,",
            "      yMax: undefined,",
            "      timeoutInSeconds: 20,",
            '      method: "LSODA",',
            "      nTimePoints: 100,",
            "    },",
            "  ]);",
            "</script>",
            "",
            "<AnalysesDashboard",
            f'  name={{"{name}"}}',
            "  initModel={initModel}",
            "  bind:analyses={analyses}",
            ">",
            f"  <p>{description}</p>" if description else "",
            "</AnalysesDashboard>",
        ]
    )
