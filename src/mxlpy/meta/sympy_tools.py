"""Tools for working with sympy expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy

from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "list_of_symbols",
    "stoichiometries_to_sympy",
    "sympy_to_inline_c",
    "sympy_to_inline_cxx",
    "sympy_to_inline_js",
    "sympy_to_inline_julia",
    "sympy_to_inline_matlab",
    "sympy_to_inline_mxlweb",
    "sympy_to_inline_py",
    "sympy_to_inline_rust",
    "sympy_to_python_fn",
]

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


def list_of_symbols(args: Iterable[str]) -> list[sympy.Symbol | sympy.Expr]:
    """Convert list of strings to list of symbols.

    Parameters
    ----------
    args
        Iterable of string names to convert to sympy symbols

    Returns
    -------
    list[sympy.Symbol | sympy.Expr]
        List of sympy symbols

    """
    return [sympy.Symbol(arg) for arg in args]


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
    return {sympy.pycode(expr, fully_qualified_modules=True, full_prec=False)}
    """.replace("math.factorial", "scipy.special.factorial")


def stoichiometries_to_sympy(
    origin: str,
    stoichs: Mapping[str, float | Derived],
) -> sympy.Expr:
    """Convert mxlpy stoichiometries to single expression.

    Parameters
    ----------
    origin
        Name of the variable the stoichiometries belong to
    stoichs
        Mapping of reaction names to stoichiometric coefficients or derived expressions

    Returns
    -------
    sympy.Expr
        Combined sympy expression representing the stoichiometries

    """
    expr = sympy.Integer(0)

    for rxn_name, rxn_stoich in stoichs.items():
        if isinstance(rxn_stoich, Derived):
            sympy_fn = fn_to_sympy_expr(
                rxn_stoich.fn,
                origin=origin,
                model_args=list_of_symbols(rxn_stoich.args),
            )
            expr = expr + sympy_fn * sympy.Symbol(rxn_name)  # type: ignore
        else:
            expr = expr + rxn_stoich * sympy.Symbol(rxn_name)  # type: ignore
    return expr.subs(1.0, 1)  # type: ignore


###############################################################################
# Inline expressions
###############################################################################


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
    return cast(str, sympy.pycode(expr, fully_qualified_modules=True, full_prec=False))


def sympy_to_inline_c(expr: sympy.Expr) -> str:
    """Create C99 code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        C99 code string

    """
    return cast(str, sympy.ccode(expr, full_prec=False))


def sympy_to_inline_js(expr: sympy.Expr) -> str:
    """Create JavaScript code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        JavaScript code string

    """
    return cast(str, sympy.jscode(expr, full_prec=False))


def sympy_to_inline_rust(expr: sympy.Expr) -> str:
    """Create Rust code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        Rust code string

    """
    return cast(str, sympy.rust_code(expr, full_prec=False))


def sympy_to_inline_julia(expr: sympy.Expr) -> str:
    """Create Julia code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        Julia code string

    """
    return cast(str, sympy.julia_code(expr, full_prec=False))


def sympy_to_inline_matlab(expr: sympy.Expr) -> str:
    """Create Julia code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        Julia code string

    """
    return cast(str, sympy.octave_code(expr, full_prec=False))


def sympy_to_inline_cxx(expr: sympy.Expr) -> str:
    """Create Julia code from sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    str
        Julia code string

    """
    return cast(str, sympy.cxxcode(expr, full_prec=False))


def sympy_to_inline_mxlweb(
    expr: sympy.Expr,
    used: set[str],
    subs: dict[sympy.Symbol, sympy.Symbol] | None,
) -> str:
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
    if subs is not None:
        expr = cast(sympy.Expr, expr.subs(subs))

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
        children = ", ".join(sympy_to_inline_mxlweb(a, used, subs) for a in expr.args)
        return f"new Add([{children}])"

    # Multiplication - handle negation and division
    if isinstance(expr, sympy.Mul):
        coeff, rest_factors = expr.as_coeff_mul()

        # Separate denominator factors (Pow(x, -n)) from rest_factors first,
        # then apply sign - this avoids Mul([Divide([Num(1), f]), g]) patterns.
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
                f"new Divide([{sympy_to_inline_mxlweb(n_expr, used, subs)}, "
                f"{sympy_to_inline_mxlweb(d_expr, used, subs)}])"
            )
        elif len(numer) == 1:
            inner = sympy_to_inline_mxlweb(numer[0], used, subs)
        else:
            used.add("Mul")
            children = ", ".join(sympy_to_inline_mxlweb(f, used, subs) for f in numer)
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
            return f"new Sqrt({sympy_to_inline_mxlweb(base, used, subs)}, new Num(2))"
        if exp == sympy.Integer(-1):
            used.add("Divide")
            used.add("Num")
            return (
                f"new Divide([new Num(1), {sympy_to_inline_mxlweb(base, used, subs)}])"
            )
        used.add("Pow")
        return f"new Pow({sympy_to_inline_mxlweb(base, used, subs)}, {sympy_to_inline_mxlweb(exp, used, subs)})"

    # Unary functions
    for sympy_type, ts_name in _UNARY_FN_MAP:
        if isinstance(expr, sympy_type):
            used.add(ts_name)
            child = cast(sympy.Expr, expr.args[0])
            return f"new {ts_name}({sympy_to_inline_mxlweb(child, used, subs)})"

    # N-ary functions
    if isinstance(expr, sympy.Max):
        used.add("Max")
        children = ", ".join(
            sympy_to_inline_mxlweb(cast(sympy.Expr, a), used, subs) for a in expr.args
        )
        return f"new Max([{children}])"

    if isinstance(expr, sympy.Min):
        used.add("Min")
        children = ", ".join(
            sympy_to_inline_mxlweb(cast(sympy.Expr, a), used, subs) for a in expr.args
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
            pieces.append(sympy_to_inline_mxlweb(val, used, subs))
            if cond is not sympy.true:
                pieces.append(sympy_to_inline_mxlweb(cond, used, subs))
        return f"new Piecewise([{', '.join(pieces)}])"

    # Relational and logical operators
    for sympy_type, ts_name in _RELATIONAL_MAP:
        if isinstance(expr, sympy_type):
            used.add(ts_name)
            children = ", ".join(
                sympy_to_inline_mxlweb(cast(sympy.Expr, a), used, subs)
                for a in expr.args
            )
            return f"new {ts_name}([{children}])"

    msg = f"Cannot convert sympy type {type(expr).__name__} ({expr!r}) to MxlWeb AST"
    raise ValueError(msg)
