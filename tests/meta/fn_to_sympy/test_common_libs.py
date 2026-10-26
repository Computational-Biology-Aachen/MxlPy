# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math
from math import e, exp

import numpy as np
import sympy

from mxlpy.meta import source_tools


def e_direct() -> float:
    return e


def e_lib() -> float:
    return math.e


def exp_direct() -> float:
    return exp(0.0)


def exp_library_call() -> float:
    return math.exp(0.0)


def test_math_import_attr_l0() -> None:
    assert source_tools.fn_to_sympy_expr(e_direct, "test") == e


def test_math_import_attr_l1() -> None:
    assert source_tools.fn_to_sympy_expr(e_lib, "test") == sympy.E


def test_math_import_fn_l0() -> None:
    assert float(source_tools.fn_to_sympy_expr(exp_direct, "test")) == 1.0  # pyright: ignore[reportArgumentType]


def test_math_import_fn_l1() -> None:
    assert float(source_tools.fn_to_sympy_expr(exp_library_call, "test")) == 1.0  # pyright: ignore[reportArgumentType]


def np_minimum(x: float, y: float) -> float:
    return np.minimum(x, y)  # type: ignore


def np_maximum(x: float, y: float) -> float:
    return np.maximum(x, y)  # type: ignore


def np_greater(x: float, y: float) -> bool:
    return np.greater(x, y)  # type: ignore


def np_less(x: float, y: float) -> bool:
    return np.less(x, y)  # type: ignore


def np_trunc(x: float) -> float:
    return np.trunc(x)  # type: ignore


def math_fmod(x: float, y: float) -> float:
    return math.fmod(x, y)


def test_np_minimum_maximum() -> None:
    x, y = sympy.symbols("x y")
    assert source_tools.fn_to_sympy_expr(
        np_minimum, "test", model_args=[x, y]
    ) == sympy.Min(x, y)
    assert source_tools.fn_to_sympy_expr(
        np_maximum, "test", model_args=[x, y]
    ) == sympy.Max(x, y)


def test_np_strict_comparisons() -> None:
    x, y = sympy.symbols("x y")
    assert source_tools.fn_to_sympy_expr(
        np_greater, "test", model_args=[x, y]
    ) == sympy.Gt(x, y)
    assert source_tools.fn_to_sympy_expr(
        np_less, "test", model_args=[x, y]
    ) == sympy.Lt(x, y)


def test_trunc_and_fmod_round_towards_zero() -> None:
    x, y = sympy.symbols("x y")
    trunc = source_tools.fn_to_sympy_expr(np_trunc, "test", model_args=[x])
    assert [trunc.subs(x, v) for v in (-1.5, 1.5)] == [-1, 1]  # pyright: ignore[reportAttributeAccessIssue]
    fmod = source_tools.fn_to_sympy_expr(math_fmod, "test", model_args=[x, y])
    assert fmod.subs({x: -5, y: 3}) == -2  # pyright: ignore[reportAttributeAccessIssue]
