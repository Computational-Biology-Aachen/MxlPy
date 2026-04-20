"""Tools for working with python source files."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import logging
import math
import textwrap
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import dill
import numpy as np
import sympy
from wadler_lindig import pformat

if TYPE_CHECKING:
    from collections.abc import Callable

    from sympy.core.function import Function

__all__ = [
    "Context",
    "KNOWN_CONSTANTS",
    "KNOWN_FNS",
    "PARSE_ERROR",
    "fn_to_sympy",
    "fn_to_sympy_outputs",
    "get_fn_ast",
    "get_fn_source",
]

_LOGGER = logging.getLogger(__name__)
PARSE_ERROR = sympy.Symbol("ERROR")


def _sympy_log10(x: sympy.Expr) -> Function:
    return sympy.log(x, 10)  # type: ignore


def _sympy_log2(x: sympy.Expr) -> Function:
    return sympy.log(x, 2)  # type: ignore


def _sympy_log1p(x: sympy.Expr) -> sympy.Expr:
    return sympy.log(1 + x)  # type: ignore


def _sympy_exp2(x: sympy.Expr) -> sympy.Expr:
    return sympy.Pow(2, x)  # type: ignore


def _sympy_expm1(x: sympy.Expr) -> sympy.Expr:
    return sympy.exp(x) - 1  # type: ignore


def _sympy_degrees(x: sympy.Expr) -> sympy.Expr:
    return x * 180 / sympy.pi  # type: ignore


def _sympy_hypot(x: sympy.Expr, y: sympy.Expr) -> sympy.Expr:
    return sympy.sqrt(x**2 + y**2)  # type: ignore


def _sympy_isqrt(x: sympy.Expr) -> sympy.Expr:
    return sympy.floor(sympy.sqrt(x))


def _sympy_ldexp(x: sympy.Expr, i: sympy.Expr) -> sympy.Expr:
    return x * sympy.Pow(2, i)  # type: ignore


def _sympy_copysign(x: sympy.Expr, y: sympy.Expr) -> sympy.Expr:
    return sympy.Abs(x) * sympy.sign(y)  # type: ignore


def _sympy_perm(n: sympy.Expr, k: sympy.Expr) -> sympy.Expr:
    return sympy.FallingFactorial(n, k)  # type: ignore


def _sympy_square(x: sympy.Expr) -> sympy.Expr:
    return sympy.Pow(x, 2)


def _sympy_subtract(x: sympy.Expr, y: sympy.Expr) -> sympy.Expr:
    return x - y  # type: ignore


def _sympy_true_divide(x: sympy.Expr, y: sympy.Expr) -> sympy.Expr:
    return x / y  # type: ignore


KNOWN_CONSTANTS: dict[float, sympy.Float] = {
    math.e: sympy.E,
    math.pi: sympy.pi,
    math.nan: sympy.nan,
    math.tau: sympy.pi * 2,
    math.inf: sympy.oo,
    # numpy
    np.e: sympy.E,
    np.pi: sympy.pi,
    np.nan: sympy.nan,
    np.inf: sympy.oo,
}

KNOWN_FNS: dict[Callable, sympy.Expr] = {
    # built-ins
    abs: sympy.Abs,  # type: ignore
    min: sympy.Min,
    max: sympy.Max,
    pow: sympy.Pow,
    # round: sympy
    # divmod
    # math module
    math.acos: sympy.acos,
    math.acosh: sympy.acosh,
    math.asin: sympy.asin,
    math.asinh: sympy.asinh,
    math.atan: sympy.atan,
    math.atan2: sympy.atan2,
    math.atanh: sympy.atanh,
    math.cbrt: sympy.cbrt,
    math.ceil: sympy.ceiling,
    math.comb: sympy.binomial,
    math.copysign: _sympy_copysign,
    math.cos: sympy.cos,
    math.cosh: sympy.cosh,
    math.degrees: _sympy_degrees,
    # math.dist: no scalar sympy equivalent
    math.erf: sympy.erf,
    math.erfc: sympy.erfc,
    math.exp: sympy.exp,
    math.exp2: _sympy_exp2,
    math.expm1: _sympy_expm1,
    math.fabs: sympy.Abs,
    math.factorial: sympy.factorial,
    math.floor: sympy.floor,
    math.fmod: sympy.Mod,
    # math.frexp: returns tuple, no sympy equivalent
    # math.fsum: takes iterable, no sympy equivalent
    math.gamma: sympy.gamma,
    math.gcd: sympy.gcd,
    math.hypot: _sympy_hypot,
    # math.isclose: boolean, no sympy equivalent
    # math.isfinite: boolean, no sympy equivalent
    # math.isinf: boolean, no sympy equivalent
    # math.isnan: boolean, no sympy equivalent
    math.isqrt: _sympy_isqrt,
    math.lcm: sympy.lcm,
    math.ldexp: _sympy_ldexp,
    math.lgamma: sympy.loggamma,
    math.log: sympy.log,
    math.log10: _sympy_log10,
    math.log1p: _sympy_log1p,
    math.log2: _sympy_log2,
    # math.modf: returns tuple, no sympy equivalent
    # math.nextafter: float-specific, no sympy equivalent
    math.perm: _sympy_perm,
    math.pow: sympy.Pow,
    math.prod: sympy.prod,
    math.radians: sympy.rad,
    math.remainder: sympy.rem,
    math.sin: sympy.sin,
    math.sinh: sympy.sinh,
    math.sqrt: sympy.sqrt,
    # math.sumprod: takes iterables, no sympy equivalent
    math.tan: sympy.tan,
    math.tanh: sympy.tanh,
    math.trunc: sympy.trunc,
    # math.ulp: float-specific, no sympy equivalent
    # numpy
    np.abs: sympy.Abs,
    np.acos: sympy.acos,
    np.acosh: sympy.acosh,
    np.asin: sympy.asin,
    np.asinh: sympy.asinh,
    np.atan: sympy.atan,
    np.atanh: sympy.atanh,
    np.atan2: sympy.atan2,
    np.pow: sympy.Pow,
    np.absolute: sympy.Abs,
    np.add: sympy.Add,
    np.arccos: sympy.acos,
    np.arccosh: sympy.acosh,
    np.arcsin: sympy.asin,
    np.arcsinh: sympy.asinh,
    np.arctan2: sympy.atan2,
    np.arctan: sympy.atan,
    np.arctanh: sympy.atanh,
    np.cbrt: sympy.cbrt,
    np.ceil: sympy.ceiling,
    np.conjugate: sympy.conjugate,
    np.cos: sympy.cos,
    np.cosh: sympy.cosh,
    np.exp: sympy.exp,
    np.floor: sympy.floor,
    np.gcd: sympy.gcd,
    np.greater: sympy.GreaterThan,
    np.greater_equal: sympy.Ge,
    np.invert: sympy.invert,
    np.lcm: sympy.lcm,
    np.less: sympy.LessThan,
    np.less_equal: sympy.Le,
    np.log: sympy.log,
    np.log10: _sympy_log10,
    np.maximum: sympy.maximum,
    np.minimum: sympy.minimum,
    np.mod: sympy.Mod,
    np.positive: sympy.Abs,
    np.power: sympy.Pow,
    np.sign: sympy.sign,
    np.sin: sympy.sin,
    np.sinh: sympy.sinh,
    np.sqrt: sympy.sqrt,
    np.square: _sympy_square,
    np.subtract: _sympy_subtract,
    np.tan: sympy.tan,
    np.tanh: sympy.tanh,
    np.true_divide: _sympy_true_divide,
    np.trunc: sympy.trunc,
    # np.vecdot: vector operation, no scalar sympy equivalent
}


@dataclass
class Context:
    """Context for converting a function to sympy expression."""

    symbols: dict[str, sympy.Symbol | sympy.Expr]
    caller: Callable
    parent_module: ModuleType | None
    origin: str
    modules: dict[str, ModuleType]
    fns: dict[str, Callable]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def updated(
        self,
        symbols: dict[str, sympy.Symbol | sympy.Expr] | None = None,
        caller: Callable | None = None,
        parent_module: ModuleType | None = None,
    ) -> Context:
        """Update the context with new values.

        Parameters
        ----------
        symbols
            New symbol mapping, or None to keep existing
        caller
            New caller function, or None to keep existing
        parent_module
            New parent module, or None to keep existing

        Returns
        -------
        Context
            New context with updated values

        """
        return Context(
            symbols=self.symbols if symbols is None else symbols,
            caller=self.caller if caller is None else caller,
            parent_module=self.parent_module
            if parent_module is None
            else parent_module,
            origin=self.origin,
            modules=self.modules,
            fns=self.fns,
        )


def _find_root(value: ast.Attribute | ast.Name, levels: list) -> list[str]:
    if isinstance(value, ast.Attribute):
        return _find_root(
            cast(ast.Attribute, value.value),
            [value.attr, *levels],
        )

    root = str(value.id)
    return [root, *levels]


def get_fn_source(fn: Callable) -> str:
    """Get the string representation of a function.

    Parameters
    ----------
    fn
        The function to extract source from

    Returns
    -------
    str
        String representation of the function's source code

    Examples
    --------
    >>> def example_fn(x): return x * 2
    >>> source = get_fn_source(example_fn)
    >>> print(source)
    def example_fn(x): return x * 2

    """
    try:
        return inspect.getsource(fn)
    except OSError:  # could not get source code
        return dill.source.getsource(fn)


def get_fn_ast(fn: Callable) -> ast.FunctionDef:
    """Get the source code of a function as an AST.

    Parameters
    ----------
    fn
        The function to convert to AST

    Returns
    -------
    ast.FunctionDef
        Abstract syntax tree representation of the function

    Raises
    ------
    TypeError
        If the input is not a function

    Examples
    --------
    >>> def example_fn(x): return x * 2
    >>> ast_tree = get_fn_ast(example_fn)
    >>> isinstance(ast_tree, ast.FunctionDef)
    True

    """
    tree = ast.parse(textwrap.dedent(get_fn_source(fn)))
    if not isinstance(fn_def := tree.body[0], ast.FunctionDef):
        msg = f"Expected a function definition but got {type(fn_def).__name__} — pass a named function, not a class or bare expression"
        raise TypeError(msg)
    return fn_def


def fn_to_sympy(
    fn: Callable,
    origin: str,
    model_args: list[sympy.Symbol | sympy.Expr] | None = None,
) -> sympy.Expr | None:
    """Convert a python function to a sympy expression.

    Parameters
    ----------
    fn
        The function to convert
    origin
        Name of the original caller. Used for error messages.
    model_args
        Optional list of sympy symbols to substitute for function arguments

    Returns
    -------
        Sympy expression equivalent to the function

    Examples
    --------
        >>> def square_fn(x):
        ...     return x**2
        >>> import sympy
        >>> fn_to_sympy(square_fn)
        x**2
        >>> # With model_args
        >>> y = sympy.Symbol('y')
        >>> fn_to_sympy(square_fn, [y])
        y**2

    """
    try:
        fn_def = get_fn_ast(fn)
        fn_args = [str(arg.arg) for arg in fn_def.args.args]

        sympy_expr = _handle_fn_body(
            fn_def.body,
            ctx=Context(
                symbols={name: sympy.Symbol(name) for name in fn_args},
                caller=fn,
                parent_module=inspect.getmodule(fn),
                origin=origin,
                modules={},
                fns={},
            ),
        )
        if sympy_expr is None:
            return None
        # Evaluated fns and floats from attributes
        if isinstance(sympy_expr, float):
            return sympy.Float(sympy_expr)
        if model_args is not None and len(model_args):
            sympy_expr = sympy_expr.subs(dict(zip(fn_args, model_args, strict=True)))
        return cast(sympy.Expr, sympy_expr)

    except (TypeError, ValueError, NotImplementedError) as e:
        msg = f"Failed parsing function of {origin}"
        _LOGGER.warning(msg)
        _LOGGER.debug("", exc_info=e)
        return None


def fn_to_sympy_outputs(
    fn: Callable,
    origin: str,
    model_args: list[sympy.Symbol | sympy.Expr] | None = None,
) -> list[sympy.Expr] | None:
    """Convert a (possibly multi-output) function to a list of sympy expressions.

    Like :func:`fn_to_sympy` but returns one expression per return value.
    Single-output functions return a one-element list.  Tuple-returning
    functions return one element per tuple component.

    Parameters
    ----------
    fn
        The function to convert
    origin
        Name of the original caller. Used for error messages.
    model_args
        Optional list of sympy symbols to substitute for function arguments

    Returns
    -------
    list[sympy.Expr] | None
        List of sympy expressions, one per output, or None on failure

    """
    try:
        fn_def = get_fn_ast(fn)
        fn_args = [str(arg.arg) for arg in fn_def.args.args]

        ctx = Context(
            symbols={name: sympy.Symbol(name) for name in fn_args},
            caller=fn,
            parent_module=inspect.getmodule(fn),
            origin=origin,
            modules={},
            fns={},
        )

        exprs = _handle_fn_body_outputs(fn_def.body, ctx)
        if exprs is None:
            return None

        if model_args is not None and len(model_args):
            subs = dict(zip(fn_args, model_args, strict=True))
            exprs = [cast(sympy.Expr, e.subs(subs)) for e in exprs]

    except (TypeError, ValueError, NotImplementedError) as e:
        msg = f"Failed parsing function of {origin}"
        _LOGGER.warning(msg)
        _LOGGER.debug("", exc_info=e)
        return None
    else:
        return exprs


def _handle_fn_body_outputs(
    body: list[ast.stmt], ctx: Context
) -> list[sympy.Expr] | None:
    """Process a function body and return one sympy expr per output.

    Handles assignments (to build up context) and a final ``return``.
    Tuple returns produce one expression per element; scalar returns produce
    a one-element list.  Piecewise / if-else functions are not supported.
    """
    for node in body:
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Tuple):
                target_elements = node.targets[0].elts
                if isinstance(node.value, ast.Tuple):
                    for target, value_expr in zip(
                        target_elements, node.value.elts, strict=True
                    ):
                        if isinstance(target, ast.Name):
                            expr = _handle_expr(value_expr, ctx)
                            if expr is None:
                                return None
                            ctx.symbols[target.id] = expr
            else:
                if not isinstance(target := node.targets[0], ast.Name):
                    msg = f"Only simple 'x = expr' assignments are supported in rate functions — got {type(target).__name__} assignment target at line {node.lineno}"
                    raise TypeError(msg)
                value = _handle_expr(node.value, ctx)
                if value is None:
                    return None
                ctx.symbols[target.id] = value

        elif isinstance(node, ast.Return):
            if node.value is None:
                return None
            if isinstance(node.value, ast.Tuple):
                result: list[sympy.Expr] = []
                for elt in node.value.elts:
                    expr = _handle_expr(elt, ctx)
                    if expr is None:
                        return None
                    result.append(cast(sympy.Expr, expr))
                return result
            expr = _handle_expr(node.value, ctx)
            if expr is None:
                return None
            return [cast(sympy.Expr, expr)]

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                ctx.modules[name] = importlib.import_module(name)

        elif isinstance(node, ast.ImportFrom):
            package = cast(str, node.module)
            module = importlib.import_module(package)
            contents = dict(inspect.getmembers(module))
            for alias in node.names:
                name = alias.name
                el = contents[name]
                if isinstance(el, float):
                    ctx.symbols[name] = sympy.Float(el)
                elif callable(el):
                    ctx.fns[name] = el
                elif isinstance(el, ModuleType):
                    ctx.modules[name] = el
                else:
                    _LOGGER.debug("Skipping import %s", node)

    return None


def _handle_fn_body(body: list[ast.stmt], ctx: Context) -> sympy.Expr | None:
    pieces = []
    remaining_body = list(body)

    while remaining_body:
        node = remaining_body.pop(0)

        if isinstance(node, ast.If):
            condition = _handle_expr(node.test, ctx)
            if_expr = _handle_fn_body(node.body, ctx)
            pieces.append((if_expr, condition))

            # If there's an else clause
            if node.orelse:
                # Check if it's an elif (an If node in orelse)
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # Push the elif back to the beginning of remaining_body to process next
                    remaining_body.insert(0, node.orelse[0])
                else:
                    # It's a regular else
                    else_expr = _handle_fn_body(node.orelse, ctx)  # FIXME: copy here
                    pieces.append((else_expr, True))
                    break  # We're done with this chain

            elif not remaining_body and any(
                isinstance(n, ast.Return) for n in body[body.index(node) + 1 :]
            ):
                else_expr = _handle_fn_body(
                    body[body.index(node) + 1 :], ctx
                )  # FIXME: copy here
                pieces.append((else_expr, True))

        elif isinstance(node, ast.Return):
            if (value := node.value) is None:
                msg = "Bare 'return' with no value is not supported in rate functions — return a float expression"
                raise ValueError(msg)

            expr = _handle_expr(value, ctx)
            if not pieces:
                return expr
            pieces.append((expr, True))
            break

        elif isinstance(node, ast.Assign):
            # Handle tuple assignments like c, d = a, b
            if isinstance(node.targets[0], ast.Tuple):
                # Handle tuple unpacking
                target_elements = node.targets[0].elts

                if isinstance(node.value, ast.Tuple):
                    # Direct unpacking like c, d = a, b
                    value_elements = node.value.elts
                    for target, value_expr in zip(
                        target_elements, value_elements, strict=True
                    ):
                        if isinstance(target, ast.Name):
                            expr = _handle_expr(value_expr, ctx)
                            if expr is None:
                                return None
                            ctx.symbols[target.id] = expr
                else:
                    # Handle potential iterable unpacking
                    value = _handle_expr(node.value, ctx)
            else:
                # Regular single assignment
                if not isinstance(target := node.targets[0], ast.Name):
                    msg = f"Only simple 'x = expr' assignments are supported in rate functions — got {type(target).__name__} assignment target at line {node.lineno}"
                    raise TypeError(msg)
                target_name = target.id
                value = _handle_expr(node.value, ctx)
                if value is None:
                    return None
                ctx.symbols[target_name] = value

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                ctx.modules[name] = importlib.import_module(name)

        elif isinstance(node, ast.ImportFrom):
            package = cast(str, node.module)
            module = importlib.import_module(package)
            contents = dict(inspect.getmembers(module))
            for alias in node.names:
                name = alias.name
                el = contents[name]
                if isinstance(el, float):
                    ctx.symbols[name] = sympy.Float(el)
                elif callable(el):
                    ctx.fns[name] = el
                elif isinstance(el, ModuleType):
                    ctx.modules[name] = el
                else:
                    _LOGGER.debug("Skipping import %s", node)
        else:
            _LOGGER.debug("Skipping node of type %s", type(node))

    # If we have pieces to combine into a Piecewise
    if pieces:
        return sympy.Piecewise(*pieces)

    # If no return was found but we have assignments, return the last assigned variable
    for node in reversed(body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            return ctx.symbols[target_name]

    msg = "Rate function has no return statement — add 'return <float_expr>' at the end of the function"
    raise ValueError(msg)


def _handle_expr(node: ast.expr, ctx: Context) -> sympy.Expr | None:
    """Key dispatch function."""
    if isinstance(node, float):
        return sympy.Float(node)
    if isinstance(node, ast.UnaryOp):
        return _handle_unaryop(node, ctx)
    if isinstance(node, ast.BinOp):
        return _handle_binop(node, ctx)
    if isinstance(node, ast.Name):
        return _handle_name(node, ctx)
    if isinstance(node, ast.Constant):
        if isinstance(val := node.value, (float, int)):
            return sympy.Float(val)
        msg = (
            f"Non-numeric constant {node.value!r} (type {type(node.value).__name__!r}) "
            "not supported in rate functions — only int/float literals are allowed"
        )
        raise NotImplementedError(msg)
    if isinstance(node, ast.Call):
        return _handle_call(node, ctx=ctx)
    if isinstance(node, ast.Attribute):
        return _handle_attribute(node, ctx=ctx)

    if isinstance(node, ast.Compare):
        # Handle chained comparisons like 1 < a < 2
        left = cast(Any, _handle_expr(node.left, ctx))
        comparisons = []

        # Build all individual comparisons from the chain
        prev_value = left
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = cast(Any, _handle_expr(comparator, ctx))

            if isinstance(op, ast.Gt):
                comparisons.append(prev_value > right)
            elif isinstance(op, ast.GtE):
                comparisons.append(prev_value >= right)
            elif isinstance(op, ast.Lt):
                comparisons.append(prev_value < right)
            elif isinstance(op, ast.LtE):
                comparisons.append(prev_value <= right)
            elif isinstance(op, ast.Eq):
                comparisons.append(prev_value == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(prev_value != right)

            prev_value = right

        # Combine all comparisons with logical AND
        result = comparisons[0]
        for comp in comparisons[1:]:
            result = sympy.And(result, comp)
        return cast(sympy.Expr, result)

    # Handle conditional expressions (ternary operators)
    if isinstance(node, ast.IfExp):
        condition = _handle_expr(node.test, ctx)
        if_true = _handle_expr(node.body, ctx)
        if_false = _handle_expr(node.orelse, ctx)
        return sympy.Piecewise((if_true, condition), (if_false, True))

    msg = f"Expression type {type(node).__name__!r} is not supported in symbolic conversion — simplify the rate function to use only arithmetic operators and supported math functions"
    raise NotImplementedError(msg)


def _handle_name(node: ast.Name, ctx: Context) -> sympy.Symbol | sympy.Expr:
    value = ctx.symbols.get(node.id)
    if value is None:
        global_variables = dict(
            inspect.getmembers(
                ctx.parent_module,
                predicate=lambda x: isinstance(x, float),
            )
        )
        value = sympy.Float(global_variables[node.id])
    return value


def _handle_unaryop(node: ast.UnaryOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.operand, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    match node.op:
        case ast.UAdd():
            return +left
        case ast.USub():
            return -left
        case _:
            msg = f"Unary operator {type(node.op).__name__!r} is not supported — only +x and -x are allowed in rate functions"
            raise NotImplementedError(msg)


def _handle_binop(node: ast.BinOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.left, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    right = _handle_expr(node.right, ctx)
    right = cast(Any, right)  # stupid sympy types don't allow ops on symbols

    match node.op:
        case ast.Add():
            return left + right
        case ast.Sub():
            return left - right
        case ast.Mult():
            return left * right
        case ast.Div():
            return left / right
        case ast.Pow():
            return left**right
        case ast.Mod():
            return left % right
        case ast.FloorDiv():
            return left // right
        case _:
            msg = f"Binary operator {type(node.op).__name__!r} is not supported in rate functions — allowed: +, -, *, /, **, %, //"
            raise NotImplementedError(msg)


def _get_inner_object(obj: object, levels: list[str]) -> sympy.Float | None:
    # Check if object is instantiated, otherwise instantiate first
    if isinstance(obj, type):
        obj = obj()

    for level in levels:
        _LOGGER.debug("obj %s, level %s", obj, level)
        obj = getattr(obj, level, None)

    if obj is None:
        return None

    if isinstance(obj, float):
        if (value := KNOWN_CONSTANTS.get(obj)) is not None:
            return value
        return sympy.Float(obj)

    _LOGGER.debug("Inner object not float: %s", obj)
    return None


# FIXME: check if target isn't an object or class
def _handle_attribute(node: ast.Attribute, ctx: Context) -> sympy.Expr | None:
    """Handle an attribute.

    Structures to expect:
        Attribute(Name(id), attr)             | direct
        Attribute(Attribute(Name(id)), attr)  | single layer of nesting
        Attribute(Attribute(...), attr)       | arbitrary nesting

    Targets to expect:
        - modules (both absolute and relative import)
            - import a; a.attr
            - import a; a.b.attr
            - from a import b; b.attr
        - objects, e.g. Parameters().a
        - classes, e.g. Parameters.a

    Watch out for relative imports and the different ways they can be called
    import a
    from a import b
    from a.b import c

    a.attr
    b.attr
    c.attr
    a.b.attr
    b.c.attr
    a.b.c.attr
    """
    name = str(node.attr)
    module: ModuleType | None = None
    modules = (
        dict(inspect.getmembers(ctx.parent_module, predicate=inspect.ismodule))
        | ctx.modules
    )
    variables = vars(ctx.parent_module)

    match node.value:
        case ast.Name(l1):
            module_name = l1
            module = modules.get(module_name)
            if module is None and (var := variables.get(l1)) is not None:
                return _get_inner_object(var, [node.attr])
        case ast.Attribute():
            levels = _find_root(node.value, levels=[])
            _LOGGER.debug("Attribute levels %s", levels)
            module_name = ".".join(levels)

            for idx, level in enumerate(levels[:-1]):
                if (module := modules.get(level)) is not None:
                    modules.update(
                        dict(
                            inspect.getmembers(
                                module,
                                predicate=inspect.ismodule,
                            )
                        )
                    )
                elif (var := variables.get(level)) is not None:
                    _LOGGER.debug("var %s", var)
                    return _get_inner_object(var, [*levels[idx + 1 :], node.attr])

                else:
                    _LOGGER.debug("No target found")

            module = modules.get(levels[-1])
        case _:
            msg = f"Unsupported attribute access pattern in rate function: {ast.dump(node.value)[:100]!r}"
            raise NotImplementedError(msg)

    # Fall-back to absolute import
    if module is None:
        module = importlib.import_module(module_name)

    element = dict(
        inspect.getmembers(
            module,
            predicate=lambda x: isinstance(x, float),
        )
    ).get(name)

    if element is None:
        return None

    if (value := KNOWN_CONSTANTS.get(element)) is not None:
        return value
    return sympy.Float(element)


# FIXME: check if target isn't an object or class
def _handle_call(node: ast.Call, ctx: Context) -> sympy.Expr | None:
    """Handle call expression.

    Variants
        - mass_action(x, k1)
        - fns.mass_action(x, k1)
        - mxlpy.fns.mass_action(x, k1)

    In future think about?
        - object.call
        - Class.call
    """
    model_args: list[sympy.Expr] = []
    for i in node.args:
        if (expr := _handle_expr(i, ctx)) is None:
            return None
        model_args.append(expr)
    _LOGGER.debug("Fn args: %s", model_args)

    match node.func:
        case ast.Name(id):
            fn_name = str(id)
            fns = (
                dict(inspect.getmembers(ctx.parent_module, predicate=callable))
                | dict(inspect.getmembers(builtins, predicate=callable))
                | ctx.fns
            )
            py_fn = fns.get(fn_name)

        # FIXME: use _handle_attribute for this
        case ast.Attribute(attr=fn_name):
            module: ModuleType | None = None
            modules = (
                dict(inspect.getmembers(ctx.parent_module, predicate=inspect.ismodule))
                | ctx.modules
            )

            levels = _find_root(node.func, [])
            module_name = ".".join(levels[:-1])

            _LOGGER.debug("Searching for module %s", module_name)
            for level in levels[:-1]:
                modules.update(
                    dict(inspect.getmembers(modules[level], predicate=inspect.ismodule))
                )
            module = modules.get(levels[-2])

            # Fall-back to absolute import
            if module is None:
                module = importlib.import_module(module_name)

            fns = dict(inspect.getmembers(module, predicate=callable))
            py_fn = fns.get(fn_name)
        case _:
            msg = f"Unsupported function call pattern in rate function: {ast.dump(node.func)[:100]!r}"
            raise NotImplementedError(msg)

    if py_fn is None:
        return None

    if (fn := KNOWN_FNS.get(py_fn)) is not None:
        result = fn(*model_args)  # type: ignore
        if isinstance(result, (int, float)):
            return sympy.Float(result)
        return cast(sympy.Expr, result)

    return fn_to_sympy(
        py_fn,
        origin=ctx.origin,
        model_args=model_args,
    )
