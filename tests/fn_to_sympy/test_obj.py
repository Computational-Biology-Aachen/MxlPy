from dataclasses import dataclass, field

from mxlpy.meta.source_tools import fn_to_sympy


@dataclass
class L1:
    attr_1: float = 3.0


@dataclass
class L2:
    attr_2: float = 2.0
    inner: L1 = field(default_factory=L1)


@dataclass
class L3:
    attr_3: float = 1.0
    inner: L2 = field(default_factory=L2)


c = L1(attr_1=6.0)
b = L2(attr_2=4.0, inner=c)
a = L3(attr_3=2.0, inner=b)


# Attr 1
def fn_obj_l1_attr() -> float:
    return c.attr_1


def fn_obj_l2_i1() -> float:
    return b.inner.attr_1


def fn_obj_l3_i2() -> float:
    return a.inner.inner.attr_1


def test_class_l1() -> None:
    assert fn_to_sympy(fn_obj_l1_attr, origin="test") == 6.0


def test_class_l2_i1() -> None:
    assert fn_to_sympy(fn_obj_l2_i1, origin="test") == 6.0


def test_class_l3_i2() -> None:
    assert fn_to_sympy(fn_obj_l3_i2, origin="test") == 6.0


# Attr 2


def fn_obj_l2_attr() -> float:
    return b.attr_2


def fn_obj_l3_i1() -> float:
    return a.inner.attr_2


def test_class_l2_attr() -> None:
    assert fn_to_sympy(fn_obj_l2_attr, origin="test") == 4.0


def test_class_l3_i1() -> None:
    assert fn_to_sympy(fn_obj_l3_i1, origin="test") == 4.0


# Attr 3


def fn_obj_l3_attr() -> float:
    return a.attr_3


def test_class_l3_attr() -> None:
    assert fn_to_sympy(fn_obj_l3_attr, origin="test") == 2.0
