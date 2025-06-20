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


# Attr 1


def fn_class_l1_attr() -> float:
    return L1.attr_1


def fn_class_l2_i1() -> float:
    return L2.inner.attr_1


def fn_class_l3_i2() -> float:
    return L3.inner.inner.attr_1


def test_class_l1() -> None:
    assert fn_to_sympy(fn_class_l1_attr, origin="test") == 3.0


def test_class_l2_i1() -> None:
    assert fn_to_sympy(fn_class_l2_i1, origin="test") == 3.0


def test_class_l3_i2() -> None:
    assert fn_to_sympy(fn_class_l3_i2, origin="test") == 3.0


# Attr 2


def fn_class_l2_attr() -> float:
    return L2.attr_2


def fn_class_l3_i1() -> float:
    return L3.inner.attr_2


def test_class_l2_attr() -> None:
    assert fn_to_sympy(fn_class_l2_attr, origin="test") == 2.0


def test_class_l3_i1() -> None:
    assert fn_to_sympy(fn_class_l3_i1, origin="test") == 2.0


# Attr 3


def fn_class_l3_attr() -> float:
    return L3.attr_3


def test_class_l3_attr() -> None:
    assert fn_to_sympy(fn_class_l3_attr, origin="test") == 1.0
