# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Large functions parse fast and without overflowing the stack (#203)."""

import sympy

from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import compile_fn

x, y, z, a, b = sympy.symbols("x y z a b")


def _mixed_sum(x: float, y: float, z: float) -> float:
    return x - 2.0 * y + z - x * y + 3.0 - z


def _clipped(x: float, y: float) -> float:
    return max(x * y + x, 0.0) + min(y, x - 1.0)


def _numeric_max() -> float:
    return max(1.0, 2.0)


def _scaled_sum(x: float, y: float) -> float:
    return 2.0 * (x - y)


def test_sum_chain_with_subtraction() -> None:
    assert fn_to_sympy_expr(_mixed_sum, origin="v") == (
        x - 2.0 * y + z - x * y + 3.0 - z
    )


def test_long_sum_does_not_overflow_the_stack() -> None:
    xs = sympy.symbols("x0:4")
    terms = [(i + 1) * xs[i % 4] ** (i % 5 + 1) for i in range(1500)]
    fn = compile_fn(sympy.Add(*terms), [str(s) for s in xs], name="long_sum")
    del fn.__mxlpy_expr__  # parse the generated source
    parsed = fn_to_sympy_expr(fn, origin="long_sum")
    assert parsed is not None
    point = dict.fromkeys(xs, 0.5)
    assert float(parsed.subs(point)) == float(sympy.Add(*terms).subs(point))


def test_max_min_over_symbols_are_not_simplified() -> None:
    expr = fn_to_sympy_expr(_clipped, origin="v", model_args=[a, b])
    assert expr == sympy.Max(a * b + a, 0.0, evaluate=False) + sympy.Min(
        b, a - 1.0, evaluate=False
    )


def test_max_over_numbers_is_evaluated() -> None:
    assert fn_to_sympy_expr(_numeric_max, origin="v") == sympy.Float(2.0)


def test_products_still_distribute_numbers() -> None:
    assert fn_to_sympy_expr(_scaled_sum, origin="v", model_args=[a, b]) == (
        2.0 * a - 2.0 * b
    )
