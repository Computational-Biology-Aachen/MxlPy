import sympy

from mxlpy.meta import source_tools


def attr_l1() -> float:
    import a

    return a.attr_a


def attr_l2() -> float:
    import a

    return a.b.attr_b


def attr_l3() -> float:
    import a

    return a.b.c.attr_c


def fn_call_l1() -> float:
    import a

    return a.fn_a()


def fn_call_l2() -> float:
    import a

    return a.b.fn_b()


def fn_call_l3() -> float:
    import a

    return a.b.c.fn_c()


def test_nested_fn_call() -> None:
    assert source_tools.fn_to_sympy(fn_call_l1, "test") == sympy.Float(1.0)
    assert source_tools.fn_to_sympy(fn_call_l2, "test") == sympy.Float(2.0)
    assert source_tools.fn_to_sympy(fn_call_l3, "test") == sympy.Float(3.0)


def test_attr() -> None:
    assert source_tools.fn_to_sympy(attr_l1, "test") == sympy.Float(1.0)
    assert source_tools.fn_to_sympy(attr_l2, "test") == sympy.Float(2.0)
    assert source_tools.fn_to_sympy(attr_l3, "test") == sympy.Float(3.0)
