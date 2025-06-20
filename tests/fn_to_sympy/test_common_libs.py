import math
from math import e, exp

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
    assert source_tools.fn_to_sympy(e_direct, "test") == e


def test_math_import_attr_l1() -> None:
    assert source_tools.fn_to_sympy(e_lib, "test") == sympy.E


def test_math_import_fn_l0() -> None:
    assert source_tools.fn_to_sympy(exp_direct, "test") == 1.0


def test_math_import_fn_l1() -> None:
    assert source_tools.fn_to_sympy(exp_library_call, "test") == 1.0
