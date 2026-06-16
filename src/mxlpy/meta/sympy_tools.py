"""Tools for working with sympy expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import sympy

from mxlpy.meta import _mathml as mml
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

__all__ = [
    "list_of_symbols",
    "mathml_to_sympy",
    "stoichiometries_to_sympy",
    "sympy_to_inline_c",
    "sympy_to_inline_cxx",
    "sympy_to_inline_js",
    "sympy_to_inline_julia",
    "sympy_to_inline_matlab",
    "sympy_to_inline_mxlweb",
    "sympy_to_inline_py",
    "sympy_to_inline_rust",
    "sympy_to_mathml",
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

    if expr == sympy.pi:
        used.add("Num")
        return f"new Num({float(expr)})"

    msg = f"Cannot convert sympy type {type(expr).__name__} ({expr!r}) to MxlWeb AST"
    raise ValueError(msg)


###############################################################################
# sympy <-> MathML node tree
###############################################################################

# sympy function type -> MathML unary node class
_UNARY_NODE_MAP: dict[type, Callable[[mml.Base], mml.Base]] = {
    sympy_type: getattr(mml, name) for sympy_type, name in _UNARY_FN_MAP
}
# MathML unary node class name -> sympy callable
_NODE_TO_UNARY_SYMPY: dict[str, Callable[..., sympy.Expr]] = {
    name: sympy_type for sympy_type, name in _UNARY_FN_MAP
}
# sympy relational/logical type -> MathML n-ary node class
_RELATIONAL_NODE_MAP: dict[type, Callable[[list[mml.Base]], mml.Base]] = {
    sympy_type: getattr(mml, name) for sympy_type, name in _RELATIONAL_MAP
}
_NODE_TO_RELATIONAL_SYMPY: dict[str, Callable[..., sympy.Expr]] = {
    name: sympy_type for sympy_type, name in _RELATIONAL_MAP
}


def sympy_to_mathml(expr: sympy.Expr) -> mml.Base:
    """Convert a sympy expression into a MathML node tree.

    The resulting tree uses the shared mxlpy/mxlweb node set
    (:mod:`mxlpy.meta._mathml`) and is the in-memory form behind the native
    JSON model format.

    Parameters
    ----------
    expr
        Sympy expression to convert

    Returns
    -------
    mml.Base
        Equivalent MathML expression node

    """
    # Constants (checked before Number; pi/E are not sympy.Number instances)
    if expr is sympy.true:
        return mml.Bool(value=True)
    if expr is sympy.false:
        return mml.Bool(value=False)
    if expr is sympy.pi:
        return mml.Pi()
    if expr is sympy.E:
        return mml.E()

    # Numbers
    if isinstance(expr, sympy.Number):
        return mml.Num(value=float(expr))

    # Named symbol
    if isinstance(expr, sympy.Symbol):
        return mml.Name(name=expr.name)

    # Addition
    if isinstance(expr, sympy.Add):
        return mml.Add([sympy_to_mathml(cast(sympy.Expr, a)) for a in expr.args])

    # Multiplication - handle negation and division
    if isinstance(expr, sympy.Mul):
        return _mul_to_mathml(expr)

    # Powers
    if isinstance(expr, sympy.Pow):
        base = cast(sympy.Expr, expr.args[0])
        exp = cast(sympy.Expr, expr.args[1])
        if exp == sympy.Rational(1, 2):
            return mml.Sqrt(child=sympy_to_mathml(base), base=mml.Num(value=2.0))
        if exp == sympy.Integer(-1):
            return mml.Divide([mml.Num(value=1.0), sympy_to_mathml(base)])
        return mml.Pow(left=sympy_to_mathml(base), right=sympy_to_mathml(exp))

    # Unary functions
    for sympy_type, node_cls in _UNARY_NODE_MAP.items():
        if isinstance(expr, sympy_type):
            child = sympy_to_mathml(cast(sympy.Expr, expr.args[0]))
            return node_cls(child)

    # N-ary min/max
    if isinstance(expr, sympy.Max):
        return mml.Max([sympy_to_mathml(cast(sympy.Expr, a)) for a in expr.args])
    if isinstance(expr, sympy.Min):
        return mml.Min([sympy_to_mathml(cast(sympy.Expr, a)) for a in expr.args])

    # Piecewise: sympy args are ((val, cond), ...) pairs. A trailing cond=True
    # becomes the value-only "otherwise" branch.
    if isinstance(expr, sympy.Piecewise):
        children: list[mml.Base] = []
        for pair in expr.args:
            val = cast(sympy.Expr, pair.args[0])
            cond = cast(sympy.Expr, pair.args[1])
            children.append(sympy_to_mathml(val))
            if cond is not sympy.true:
                children.append(sympy_to_mathml(cond))
        return mml.Piecewise(children)

    # Relational and logical operators
    for sympy_type, node_cls in _RELATIONAL_NODE_MAP.items():
        if isinstance(expr, sympy_type):
            return node_cls([sympy_to_mathml(cast(sympy.Expr, a)) for a in expr.args])

    msg = f"Cannot convert sympy type {type(expr).__name__} ({expr!r}) to MathML node"
    raise ValueError(msg)


def _mul_to_mathml(expr: sympy.Mul) -> mml.Base:
    coeff, rest_factors = expr.as_coeff_mul()

    numer: list[sympy.Expr] = []
    denom: list[sympy.Expr] = []
    for arg in rest_factors:
        if isinstance(arg, sympy.Pow):
            arg_exp = cast(sympy.Expr, arg.exp)
            if isinstance(arg_exp, sympy.Number) and arg_exp.is_negative:
                neg_exp = cast(sympy.Expr, -arg_exp)
                base = cast(sympy.Expr, arg.base)
                denom.append(
                    base if neg_exp == sympy.Integer(1) else sympy.Pow(base, neg_exp)
                )
                continue
        numer.append(cast(sympy.Expr, arg))

    abs_coeff = cast(sympy.Expr, -coeff if coeff.is_negative else coeff)
    if abs_coeff != sympy.Integer(1):
        numer = [cast(sympy.Expr, sympy.Number(abs_coeff)), *numer]

    if denom:
        n_expr: sympy.Expr = (
            sympy.Mul(*numer)
            if len(numer) > 1
            else (numer[0] if numer else sympy.Integer(1))
        )
        d_expr: sympy.Expr = sympy.Mul(*denom) if len(denom) > 1 else denom[0]
        inner: mml.Base = mml.Divide([sympy_to_mathml(n_expr), sympy_to_mathml(d_expr)])
    elif len(numer) == 1:
        inner = sympy_to_mathml(numer[0])
    else:
        inner = mml.Mul([sympy_to_mathml(f) for f in numer])

    if coeff.is_negative:
        return mml.Minus([inner])
    return inner


def mathml_to_sympy(node: mml.Base) -> sympy.Expr:
    """Convert a MathML node tree back into a sympy expression.

    Parameters
    ----------
    node
        MathML expression node, e.g. produced by :func:`sympy_to_mathml` or
        :func:`mxlpy.meta._mathml.node_from_dict`.

    Returns
    -------
    sympy.Expr
        Equivalent sympy expression

    """
    if isinstance(node, mml.Num):
        value = node.value
        if float(value).is_integer():
            return sympy.Integer(int(value))
        return sympy.Float(value)
    if isinstance(node, mml.Name):
        return sympy.Symbol(node.name)
    if isinstance(node, mml.Pi):
        return cast(sympy.Expr, sympy.pi)
    if isinstance(node, mml.E):
        return cast(sympy.Expr, sympy.E)
    if isinstance(node, mml.Bool):
        return cast(sympy.Expr, sympy.true if node.value else sympy.false)

    if isinstance(node, mml.Add):
        return sympy.Add(*(mathml_to_sympy(c) for c in node.children))
    if isinstance(node, mml.Mul):
        return sympy.Mul(*(mathml_to_sympy(c) for c in node.children))
    if isinstance(node, mml.Minus):
        children = [mathml_to_sympy(c) for c in node.children]
        if len(children) == 1:
            return cast(sympy.Expr, sympy.Mul(sympy.Integer(-1), children[0]))
        rest = sympy.Mul(sympy.Integer(-1), sympy.Add(*children[1:]))
        return cast(sympy.Expr, sympy.Add(children[0], rest))
    if isinstance(node, mml.Divide):
        children = [mathml_to_sympy(c) for c in node.children]
        denom = sympy.Pow(sympy.Mul(*children[1:]), sympy.Integer(-1))
        return cast(sympy.Expr, sympy.Mul(children[0], denom))
    if isinstance(node, mml.Pow):
        return sympy.Pow(mathml_to_sympy(node.left), mathml_to_sympy(node.right))
    if isinstance(node, mml.Sqrt):
        inv_base = sympy.Pow(mathml_to_sympy(node.base), sympy.Integer(-1))
        return sympy.Pow(mathml_to_sympy(node.child), inv_base)
    if isinstance(node, mml.Log):
        inv_log_base = sympy.Pow(
            sympy.log(mathml_to_sympy(node.base)), sympy.Integer(-1)
        )
        return cast(
            sympy.Expr,
            sympy.Mul(sympy.log(mathml_to_sympy(node.child)), inv_log_base),
        )
    if isinstance(node, mml.Max):
        return cast(sympy.Expr, sympy.Max(*(mathml_to_sympy(c) for c in node.children)))
    if isinstance(node, mml.Min):
        return cast(sympy.Expr, sympy.Min(*(mathml_to_sympy(c) for c in node.children)))
    if isinstance(node, mml.Piecewise):
        return _piecewise_to_sympy(node)

    name = type(node).__name__
    node_any = cast(Any, node)
    if (unary_fn := _NODE_TO_UNARY_SYMPY.get(name)) is not None:
        return cast(sympy.Expr, unary_fn(mathml_to_sympy(node_any.child)))
    if (rel_fn := _NODE_TO_RELATIONAL_SYMPY.get(name)) is not None:
        return cast(
            sympy.Expr,
            rel_fn(*(mathml_to_sympy(c) for c in node_any.children)),
        )

    msg = f"Cannot convert MathML node {name!r} to sympy"
    raise ValueError(msg)


def _piecewise_to_sympy(node: mml.Piecewise) -> sympy.Expr:
    children = node.children
    pairs: list[tuple[sympy.Expr, sympy.Expr]] = []
    i = 0
    while i + 1 < len(children):
        pairs.append((mathml_to_sympy(children[i]), mathml_to_sympy(children[i + 1])))
        i += 2
    if i < len(children):  # trailing "otherwise" value
        pairs.append((mathml_to_sympy(children[i]), cast(sympy.Expr, sympy.true)))
    return cast(sympy.Expr, sympy.Piecewise(*pairs))
