"""Regression tests for `if`/`elif`/`else` chains followed by more code.

`_handle_fn_body` used to treat an `if`/`else` chain as *being* the
function's return value: each branch contributed only its last assigned
symbol to a `Piecewise`, and any statement coming after the chain (a
combination of branch-local variables, an outer `min`/`max` clamp, ...) was
silently dropped. See `tests/meta/roundtrips/test_jax_codegen_fixes.py` for
the same bug exercised end-to-end through `generate_model_code_jax`.

Assertions substitute concrete numbers and coerce to `float` rather than
comparing sympy objects directly: `Piecewise` structural equality doesn't
reliably canonicalize (e.g. an elif chain nests rather than flattens), and
`sympy.Float(6.0) == 6` is `False` (compares unequal to a plain `int`,
unlike `== 6.0`).
"""

import sympy

from mxlpy.meta import source_tools


def branch_then_combine(cond: float, a: float, b: float) -> float:
    if cond == 0:
        x = a
    else:
        x = b
    return x * 2.0


def branch_multiple_assigns_then_combine(
    cond: float, a: float, b: float, c: float
) -> float:
    if cond == 0:
        x = a
        y = b
    else:
        x = b
        y = a
    return x * y + c


def branch_then_outer_clip(cond: float, a: float, b: float, cap: float) -> float:
    if cond <= 0:
        val = a
    else:
        val = b
    return min(val, cap)


def elif_chain_then_combine(cond: float, a: float, b: float, c: float) -> float:
    if cond == 1:
        x = a
    elif cond == 2:
        x = b
    else:
        x = c
    return x + 1.0


def if_without_else_then_return(cond: float, a: float) -> float:
    if cond > 0:
        return a
    return -a


def test_if_else_assign_then_combine_is_not_dropped() -> None:
    expr = source_tools.fn_to_sympy_expr(branch_then_combine, "test")
    cond, a, b = sympy.symbols("cond a b")
    assert expr is not None
    assert float(expr.subs({cond: 0, a: 3, b: 5})) == 6.0  # branch: a * 2
    assert float(expr.subs({cond: 1, a: 3, b: 5})) == 10.0  # branch: b * 2


def test_if_else_multiple_assigns_all_survive() -> None:
    expr = source_tools.fn_to_sympy_expr(branch_multiple_assigns_then_combine, "test")
    cond, a, b, c = sympy.symbols("cond a b c")
    assert expr is not None
    assert float(expr.subs({cond: 0, a: 2, b: 3, c: 1})) == 7.0  # a * b + c
    assert float(expr.subs({cond: 1, a: 2, b: 3, c: 1})) == 7.0  # b * a + c


def test_statement_after_if_else_is_not_dropped() -> None:
    """`min(val, cap)` after the if/else must survive, not just the branches."""
    expr = source_tools.fn_to_sympy_expr(branch_then_outer_clip, "test")
    cond, a, b, cap = sympy.symbols("cond a b cap")
    assert expr is not None
    # cond <= 0 -> val = a = 1, min(1, 10) = 1
    assert float(expr.subs({cond: 0, a: 1, b: 100, cap: 10})) == 1.0
    # cond > 0 -> val = b = 100, min(100, 10) = 10 (the clip must survive)
    assert float(expr.subs({cond: 1, a: 1, b: 100, cap: 10})) == 10.0


def test_elif_chain_then_combine() -> None:
    expr = source_tools.fn_to_sympy_expr(elif_chain_then_combine, "test")
    cond, a, b, c = sympy.symbols("cond a b c")
    assert expr is not None
    assert float(expr.subs({cond: 1, a: 10, b: 20, c: 30})) == 11.0
    assert float(expr.subs({cond: 2, a: 10, b: 20, c: 30})) == 21.0
    assert float(expr.subs({cond: 3, a: 10, b: 20, c: 30})) == 31.0


def test_if_without_else_followed_by_return() -> None:
    """No trailing statements case: still must not regress."""
    expr = source_tools.fn_to_sympy_expr(if_without_else_then_return, "test")
    cond, a = sympy.symbols("cond a")
    assert expr is not None
    assert float(expr.subs({cond: 1, a: 5})) == 5.0
    assert float(expr.subs({cond: -1, a: 5})) == -5.0
