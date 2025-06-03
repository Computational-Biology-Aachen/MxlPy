"""Module to export models as code."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from mxlpy.meta.sympy_tools import (
    fn_to_sympy,
    list_of_symbols,
    stoichiometries_to_sympy,
    sympy_to_inline_rust,
)

if TYPE_CHECKING:
    from mxlpy.model import Model

__all__ = [
    "generate_model_code_rs",
]


# FIXME: generate from SymbolicModel, should be easier?
def generate_model_code_rs(model: Model) -> str:
    """Transform the model into a single function, inlining the function calls."""
    n_variables = len(model.variables)

    source = [
        f"fn model(time: f64, variables: &[f64; {n_variables}]) -> [f64; {n_variables}] {{",
    ]

    # Variables
    variables = model.variables
    if len(variables) > 0:
        source.append("    let [{}] = *variables;".format(", ".join(variables)))

    # Parameters
    parameters = model.parameters
    if len(parameters) > 0:
        source.append(
            "\n".join(f"    let {k}: f64 = {v};" for k, v in model.parameters.items())
        )

    # Derived
    for name, derived in model.derived.items():
        expr = fn_to_sympy(derived.fn, model_args=list_of_symbols(derived.args))
        source.append(f"    let {name}: f64 = {sympy_to_inline_rust(expr)};")

    # Reactions
    for name, rxn in model.reactions.items():
        expr = fn_to_sympy(rxn.fn, model_args=list_of_symbols(rxn.args))
        source.append(f"    let {name}: f64 = {sympy_to_inline_rust(expr)};")

    # Stoichiometries; FIXME: do this with sympy instead as well?
    diff_eqs = {}
    for rxn_name, rxn in model.reactions.items():
        for var_name, factor in rxn.stoichiometry.items():
            diff_eqs.setdefault(var_name, {})[rxn_name] = factor

    for variable, stoich in diff_eqs.items():
        source.append(
            f"    let d{variable}dt: f64 = {sympy_to_inline_rust(stoichiometries_to_sympy(stoich))};"
        )

    # Surrogates
    if len(model._surrogates) > 0:  # noqa: SLF001
        warnings.warn(
            "Generating code for Surrogates not yet supported.",
            stacklevel=1,
        )

    # Return
    if len(variables) > 0:
        source.append(
            "    [{}]".format(
                ", ".join(f"d{i}dt" for i in variables),
            ),
        )
    else:
        source.append("[]")
    source.append("}")

    return "\n".join(source)
