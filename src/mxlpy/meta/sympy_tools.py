"""Tools for working with sympy expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy
from sympy.printing import jscode, rust_code
from sympy.printing.pycode import pycode

from mxlpy.meta.source_tools import fn_to_sympy
from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "list_of_symbols",
    "stoichiometries_to_sympy",
    "sympy_to_inline_js",
    "sympy_to_inline_py",
    "sympy_to_inline_rust",
    "sympy_to_python_fn",
]


def list_of_symbols(args: Iterable[str]) -> list[sympy.Symbol | sympy.Expr]:
    """Convert list of strings to list of symbols."""
    return [sympy.Symbol(arg) for arg in args]


def sympy_to_inline_py(expr: sympy.Expr) -> str:
    """Convert a sympy expression to inline Python code.

    Parameters
    ----------
    expr
        The sympy expression to convert

    Returns
    -------
    str
        Python code string for the expression

    Examples
    --------
    >>> import sympy
    >>> x = sympy.Symbol('x')
    >>> expr = x**2 + 2*x + 1
    >>> sympy_to_inline(expr)
    'x**2 + 2*x + 1'

    """
    return cast(str, pycode(expr, fully_qualified_modules=True, full_prec=False))


def sympy_to_inline_js(expr: sympy.Expr) -> str:
    """Create rust code from sympy expression."""
    return cast(str, jscode(expr, full_prec=False))


def sympy_to_inline_rust(expr: sympy.Expr) -> str:
    """Create rust code from sympy expression."""
    return cast(str, rust_code(expr, full_prec=False))


def sympy_to_python_fn(
    *,
    fn_name: str,
    args: list[str],
    expr: sympy.Expr,
) -> str:
    """Convert a sympy expression to a python function.

    Parameters
    ----------
    fn_name
        Name of the function to generate
    args
        List of argument names for the function
    expr
        Sympy expression to convert to a function body

    Returns
    -------
    str
        String representation of the generated function

    Examples
    --------
    >>> import sympy
    >>> x, y = sympy.symbols('x y')
    >>> expr = x**2 + y
    >>> print(sympy_to_fn(fn_name="square_plus_y", args=["x", "y"], expr=expr))
    def square_plus_y(x: float, y: float) -> float:
        return x**2 + y

    """
    fn_args = ", ".join(f"{i}: float" for i in args)

    return f"""def {fn_name}({fn_args}) -> float:
    return {pycode(expr, fully_qualified_modules=True, full_prec=False)}
    """.replace("math.factorial", "scipy.special.factorial")


def stoichiometries_to_sympy(
    origin: str,
    stoichs: Mapping[str, float | Derived],
) -> sympy.Expr:
    """Convert mxlpy stoichiometries to single expression."""
    expr = sympy.Integer(0)

    for rxn_name, rxn_stoich in stoichs.items():
        if isinstance(rxn_stoich, Derived):
            sympy_fn = fn_to_sympy(
                rxn_stoich.fn,
                origin=origin,
                model_args=list_of_symbols(rxn_stoich.args),
            )
            expr = expr + sympy_fn * sympy.Symbol(rxn_name)  # type: ignore
        else:
            expr = expr + rxn_stoich * sympy.Symbol(rxn_name)  # type: ignore
    return expr.subs(1.0, 1)  # type: ignore
