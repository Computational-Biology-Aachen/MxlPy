"""Module to export models as code."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import sympy

from mxlpy.meta.source_tools import fn_to_sympy, sympy_to_inline
from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from mxlpy.model import Model

__all__ = [
    "generate_model_code_py",
]


def _conditional_join[T](
    iterable: Iterable[T],
    question: Callable[[T], bool],
    true_pat: str,
    false_pat: str,
) -> str:
    """Join an iterable, applying a pattern to each element based on a condition."""

    def inner(it: Iterator[T]) -> Generator[str, None, None]:
        yield str(next(it))
        while True:
            try:
                el = next(it)
                if question(el):
                    yield f"{true_pat}{el}"
                else:
                    yield f"{false_pat}{el}"
            except StopIteration:
                break

    return "".join(inner(iter(iterable)))


def _list_of_symbols(args: list[str]) -> list[sympy.Symbol | sympy.Expr]:
    return [sympy.Symbol(arg) for arg in args]


# FIXME: generate from SymbolicModel, should be easier?
def generate_model_code_py(model: Model) -> str:
    """Transform the model into a single function, inlining the function calls."""
    source = [
        "from collections.abc import Iterable\n",
        "from mxlpy.types import Float\n",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
    ]

    # Variables
    variables = model.variables
    if len(variables) > 0:
        source.append("    {} = variables".format(", ".join(variables)))

    # Parameters
    parameters = model.parameters
    if len(parameters) > 0:
        source.append("\n".join(f"    {k} = {v}" for k, v in model.parameters.items()))

    # Derived
    for name, derived in model.derived.items():
        expr = fn_to_sympy(derived.fn, model_args=_list_of_symbols(derived.args))
        source.append(f"    {name} = {sympy_to_inline(expr)}")

    # Reactions
    for name, rxn in model.reactions.items():
        expr = fn_to_sympy(rxn.fn, model_args=_list_of_symbols(rxn.args))
        source.append(f"    {name} = {sympy_to_inline(expr)}")

    # Stoichiometries; FIXME: do this with sympy instead as well?
    stoich_srcs = {}
    for rxn_name, rxn in model.reactions.items():
        for i, (cpd_name, factor) in enumerate(rxn.stoichiometry.items()):
            if isinstance(factor, Derived):
                expr = fn_to_sympy(factor.fn, model_args=_list_of_symbols(factor.args))
                src = f"{sympy_to_inline(expr)} * {rxn_name}"
            elif factor == 1:
                src = rxn_name
            elif factor == -1:
                src = f"-{rxn_name}" if i == 0 else f"- {rxn_name}"
            else:
                src = f"{factor} * {rxn_name}"
            stoich_srcs.setdefault(cpd_name, []).append(src)
    for variable, stoich in stoich_srcs.items():
        source.append(
            f"    d{variable}dt = {_conditional_join(stoich, lambda x: x.startswith('-'), ' ', ' + ')}"
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
            "    return {}".format(
                ", ".join(f"d{i}dt" for i in variables),
            ),
        )
    else:
        source.append("    return ()")

    return "\n".join(source)
