"""Tools for working with python source files."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import dill
import sympy
from sympy.printing import rust_code
from sympy.printing.pycode import pycode

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

__all__ = [
    "Context",
    "fn_to_sympy",
    "get_fn_ast",
    "get_fn_source",
    "sympy_to_fn",
    "sympy_to_inline",
    "sympy_to_inline_rust",
]


@dataclass
class Context:
    """Context for converting a function to sympy expression."""

    symbols: dict[str, sympy.Symbol | sympy.Expr]
    caller: Callable
    parent_module: ModuleType | None

    def updated(
        self,
        symbols: dict[str, sympy.Symbol | sympy.Expr] | None = None,
        caller: Callable | None = None,
        parent_module: ModuleType | None = None,
    ) -> Context:
        """Update the context with new values."""
        return Context(
            symbols=self.symbols if symbols is None else symbols,
            caller=self.caller if caller is None else caller,
            parent_module=self.parent_module
            if parent_module is None
            else parent_module,
        )


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
        msg = "Not a function"
        raise TypeError(msg)
    return fn_def


def sympy_to_inline(expr: sympy.Expr) -> str:
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
    return cast(str, pycode(expr, fully_qualified_modules=True))


def sympy_to_inline_rust(expr: sympy.Expr) -> str:
    """Create rust code from sympy expression."""
    return cast(str, rust_code(expr))


def sympy_to_fn(
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
    return {pycode(expr)}
    """


def fn_to_sympy(
    fn: Callable,
    model_args: list[sympy.Symbol | sympy.Expr] | None = None,
) -> sympy.Expr:
    """Convert a python function to a sympy expression.

    Parameters
    ----------
    fn
        The function to convert
    model_args
        Optional list of sympy symbols to substitute for function arguments

    Returns
    -------
    sympy.Expr
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
    fn_def = get_fn_ast(fn)
    fn_args = [str(arg.arg) for arg in fn_def.args.args]
    sympy_expr = _handle_fn_body(
        fn_def.body,
        ctx=Context(
            symbols={name: sympy.Symbol(name) for name in fn_args},
            caller=fn,
            parent_module=inspect.getmodule(fn),
        ),
    )
    if model_args is not None:
        sympy_expr = sympy_expr.subs(dict(zip(fn_args, model_args, strict=True)))
    return cast(sympy.Expr, sympy_expr)


def _handle_name(node: ast.Name, ctx: Context) -> sympy.Symbol | sympy.Expr:
    return ctx.symbols[node.id]


def _handle_expr(node: ast.expr, ctx: Context) -> sympy.Expr:
    if isinstance(node, ast.UnaryOp):
        return _handle_unaryop(node, ctx)
    if isinstance(node, ast.BinOp):
        return _handle_binop(node, ctx)
    if isinstance(node, ast.Name):
        return _handle_name(node, ctx)
    if isinstance(node, ast.Constant):
        return node.value
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
    if isinstance(node, ast.Call):
        return _handle_call(node, ctx)

    # Handle conditional expressions (ternary operators)
    if isinstance(node, ast.IfExp):
        condition = _handle_expr(node.test, ctx)
        if_true = _handle_expr(node.body, ctx)
        if_false = _handle_expr(node.orelse, ctx)
        return sympy.Piecewise((if_true, condition), (if_false, True))

    msg = f"Expression type {type(node).__name__} not implemented"
    raise NotImplementedError(msg)


def _handle_fn_body(body: list[ast.stmt], ctx: Context) -> sympy.Expr:
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
                msg = "Return value cannot be None"
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
                            ctx.symbols[target.id] = _handle_expr(value_expr, ctx)
                else:
                    # Handle potential iterable unpacking
                    value = _handle_expr(node.value, ctx)
            else:
                # Regular single assignment
                if not isinstance(target := node.targets[0], ast.Name):
                    msg = "Only single variable assignments are supported"
                    raise TypeError(msg)
                target_name = target.id
                value = _handle_expr(node.value, ctx)
                ctx.symbols[target_name] = value

    # If we have pieces to combine into a Piecewise
    if pieces:
        return sympy.Piecewise(*pieces)

    # If no return was found but we have assignments, return the last assigned variable
    for node in reversed(body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            return ctx.symbols[target_name]

    msg = "No return value found in function body"
    raise ValueError(msg)


def _handle_unaryop(node: ast.UnaryOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.operand, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    if isinstance(node.op, ast.UAdd):
        return +left
    if isinstance(node.op, ast.USub):
        return -left

    msg = f"Operation {type(node.op).__name__} not implemented"
    raise NotImplementedError(msg)


def _handle_binop(node: ast.BinOp, ctx: Context) -> sympy.Expr:
    left = _handle_expr(node.left, ctx)
    left = cast(Any, left)  # stupid sympy types don't allow ops on symbols

    right = _handle_expr(node.right, ctx)
    right = cast(Any, right)  # stupid sympy types don't allow ops on symbols

    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        return left**right
    if isinstance(node.op, ast.Mod):
        return left % right
    if isinstance(node.op, ast.FloorDiv):
        return left // right

    msg = f"Operation {type(node.op).__name__} not implemented"
    raise NotImplementedError(msg)


def _handle_call(node: ast.Call, ctx: Context) -> sympy.Expr:
    # direct call, e.g. mass_action(x, k1)
    if isinstance(callee := node.func, ast.Name):
        fn_name = str(callee.id)
        fns = dict(inspect.getmembers(ctx.parent_module, predicate=callable))

        return fn_to_sympy(
            fns[fn_name],
            model_args=[_handle_expr(i, ctx) for i in node.args],
        )

    # search for fn in other namespace
    if isinstance(attr := node.func, ast.Attribute):
        imports = dict(inspect.getmembers(ctx.parent_module, inspect.ismodule))

        # Single level, e.g. fns.mass_action(x, k1)
        if isinstance(module_name := attr.value, ast.Name):
            return _handle_call(
                ast.Call(func=ast.Name(attr.attr), args=node.args, keywords=[]),
                ctx=ctx.updated(parent_module=imports[module_name.id]),
            )

        # Multiple levels, e.g. mxlpy.fns.mass_action(x, k1)
        if isinstance(inner_attr := attr.value, ast.Attribute):
            if not isinstance(module_name := inner_attr.value, ast.Name):
                msg = f"Unknown target kind {module_name}"
                raise NotImplementedError(msg)
            return _handle_call(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(inner_attr.attr),
                        attr=attr.attr,
                    ),
                    args=node.args,
                    keywords=[],
                ),
                ctx=ctx.updated(parent_module=imports[module_name.id]),
            )

    msg = f"Unsupported function type {node.func}"
    raise NotImplementedError(msg)
