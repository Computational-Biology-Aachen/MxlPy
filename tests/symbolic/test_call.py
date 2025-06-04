"""Test call function of fn_to_sympy."""

import sympy

import mxlpy
from mxlpy import fns
from mxlpy.fns import mass_action_1s
from mxlpy.meta.sympy_tools import fn_to_sympy


def using_inner_l1(x: float, y: float) -> float:
    return mass_action_1s(x, y) + y


def using_inner_l2(x: float, y: float) -> float:
    return fns.mass_action_1s(x, y) + y


def using_inner_l3(x: float, y: float) -> float:
    return mxlpy.fns.mass_action_1s(x, y) + y


def test_call_level1() -> None:
    assert (
        sympy.latex(
            fn_to_sympy(
                using_inner_l1,
                origin="test",
                model_args=sympy.symbols("x y"),
            )
        )
        == "x y + y"
    )


def test_call_level2() -> None:
    assert (
        sympy.latex(
            fn_to_sympy(
                using_inner_l2,
                origin="test",
                model_args=sympy.symbols("x y"),
            )
        )
        == "x y + y"
    )


def test_call_level3() -> None:
    assert (
        sympy.latex(
            fn_to_sympy(
                using_inner_l3,
                origin="test",
                model_args=sympy.symbols("x y"),
            )
        )
        == "x y + y"
    )
