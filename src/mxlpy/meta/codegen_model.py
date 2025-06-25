"""Module to export models as code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mxlpy.meta.sympy_tools import (
    fn_to_sympy,
    list_of_symbols,
    stoichiometries_to_sympy,
    sympy_to_inline_py,
    sympy_to_inline_rust,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import sympy

    from mxlpy.model import Model

__all__ = [
    "generate_model_code_py",
    "generate_model_code_rs",
]

_LOGGER = logging.getLogger(__name__)


def _generate_model_code(
    model: Model,
    *,
    sized: bool,
    model_fn: str,
    variables_template: str,
    assignment_template: str,
    sympy_inline_fn: Callable[[sympy.Expr], str],
    return_template: str,
    imports: list[str] | None = None,
    end: str | None = None,
    free_parameters: list[str] | None = None,
) -> str:
    source: list[str] = []
    # Model components
    variables = model.get_initial_conditions()
    parameters = model.get_parameter_values()

    if imports is not None:
        source.extend(imports)

    if not sized:
        source.append(model_fn)
    else:
        source.append(model_fn.format(n=len(variables)))

    if len(variables) > 0:
        source.append(variables_template.format(", ".join(variables)))

    # Parameters
    if free_parameters is not None:
        for key in free_parameters:
            parameters.pop(key)
    if len(parameters) > 0:
        source.append(
            "\n".join(
                assignment_template.format(k=k, v=v) for k, v in parameters.items()
            )
        )

    # Derived
    for name, derived in model.get_raw_derived().items():
        expr = fn_to_sympy(
            derived.fn,
            origin=name,
            model_args=list_of_symbols(derived.args),
        )
        if expr is None:
            msg = f"Unable to parse fn for derived value '{name}'"
            raise ValueError(msg)
        source.append(assignment_template.format(k=name, v=sympy_inline_fn(expr)))

    # Reactions
    for name, rxn in model.get_raw_reactions().items():
        expr = fn_to_sympy(
            rxn.fn,
            origin=name,
            model_args=list_of_symbols(rxn.args),
        )
        if expr is None:
            msg = f"Unable to parse fn for reaction value '{name}'"
            raise ValueError(msg)
        source.append(assignment_template.format(k=name, v=sympy_inline_fn(expr)))

    # Diff eqs
    diff_eqs = {}
    for rxn_name, rxn in model.get_raw_reactions().items():
        for var_name, factor in rxn.stoichiometry.items():
            diff_eqs.setdefault(var_name, {})[rxn_name] = factor

    for variable, stoich in diff_eqs.items():
        expr = stoichiometries_to_sympy(origin=variable, stoichs=stoich)
        source.append(
            assignment_template.format(k=f"d{variable}dt", v=sympy_inline_fn(expr))
        )

    # Surrogates
    if len(model._surrogates) > 0:  # noqa: SLF001
        msg = "Generating code for Surrogates not yet supported."
        _LOGGER.warning(msg)

    # Return
    ret_order = [i for i in variables if i in diff_eqs]
    ret = ", ".join(f"d{i}dt" for i in ret_order) if len(diff_eqs) > 0 else "()"
    source.append(return_template.format(ret))

    if end is not None:
        source.append(end)

    # print(source)
    return "\n".join(source)


def generate_model_code_py(
    model: Model,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a python function, inlining the function calls."""
    if free_parameters is None:
        model_fn = (
            "def model(time: float, variables: Iterable[float]) -> Iterable[float]:"
        )
    else:
        args = ", ".join(f"{k}: float" for k in free_parameters)
        model_fn = f"def model(time: float, variables: Iterable[float], {args}) -> Iterable[float]:"

    return _generate_model_code(
        model,
        imports=[
            "from collections.abc import Iterable\n",
        ],
        sized=False,
        model_fn=model_fn,
        variables_template="    {} = variables",
        assignment_template="    {k} = {v}",
        sympy_inline_fn=sympy_to_inline_py,
        return_template="    return {}",
        end=None,
        free_parameters=free_parameters,
    )


def generate_model_code_rs(
    model: Model,
    free_parameters: list[str] | None = None,
) -> str:
    """Transform the model into a rust function, inlining the function calls."""
    if free_parameters is None:
        model_fn = "fn model(time: f64, variables: &[f64; {n}]) -> [f64; {n}] {{"
    else:
        args = ", ".join(f"{k}: f64" for k in free_parameters)
        model_fn = f"fn model(time: f64, variables: &[f64; {{n}}], {args}) -> [f64; {{n}}] {{{{"

    return _generate_model_code(
        model,
        imports=None,
        sized=True,
        model_fn=model_fn,
        variables_template="    let [{}] = *variables;",
        assignment_template="    let {k}: f64 = {v};",
        sympy_inline_fn=sympy_to_inline_rust,
        return_template="    return [{}]",
        end="}",
        free_parameters=free_parameters,
    )
