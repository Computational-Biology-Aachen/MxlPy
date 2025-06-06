import sympy
from a.b import c

from mxlpy.meta import source_tools


def attr_l1() -> float:
    return c.attr_c


def fn_call_l1() -> float:
    return c.fn_c()


def test_nested_fn_call() -> None:
    assert source_tools.fn_to_sympy(attr_l1, "test") == sympy.Float(3.0)


def test_attr() -> None:
    assert source_tools.fn_to_sympy(fn_call_l1, "test") == sympy.Float(3.0)
