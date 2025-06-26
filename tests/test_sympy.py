import sympy

##############################
# Utility functions
##############################
from mxlpy.fns import mass_action_1s
from mxlpy.meta.sympy_tools import fn_to_sympy


def power(x: float) -> float:
    return x**2


def times(x: float) -> float:
    return x * 2


def add(x: float) -> float:
    return x + 2


def sub(x: float) -> float:
    return x - 2


##############################
# Tests unar & binary ops
##############################


def test_unary_plus() -> None:
    def fn(a: float) -> float:
        return +a

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == a


def test_unary_minus() -> None:
    def fn(a: float) -> float:
        return -a

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == -a


def test_binary_add() -> None:
    def fn(left: float, right: float) -> float:
        return left + right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left + right


def test_binary_sub() -> None:
    def fn(left: float, right: float) -> float:
        return left - right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left - right


def test_binary_mul() -> None:
    def fn(left: float, right: float) -> float:
        return left * right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left * right


def test_binary_div() -> None:
    def fn(left: float, right: float) -> float:
        return left / right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left / right


def test_binary_floordiv() -> None:
    def fn(left: float, right: float) -> float:
        return left // right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left // right


def test_binary_pow() -> None:
    def fn(left: float, right: float) -> float:
        return left**right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left**right


def test_binary_mod() -> None:
    def fn(left: float, right: float) -> float:
        return left % right

    left, right = sympy.symbols("left right")
    assert fn_to_sympy(fn, origin="test", model_args=[left, right]) == left % right


##############################
# Tests ?
##############################


def test_single_assignment() -> None:
    def fn(a: float, b: float) -> float:
        c = a * b
        return c  # noqa: RET504

    a, b = sympy.symbols("a b")
    assert fn_to_sympy(fn, origin="test", model_args=[a, b]) == a * b


def test_two_assignments() -> None:
    def fn(a: float, b: float) -> float:
        c = a * b
        d = c
        return d  # noqa: RET504

    a, b = sympy.symbols("a b")
    assert fn_to_sympy(fn, origin="test", model_args=[a, b]) == a * b


def test_double_assignment() -> None:
    def fn(a: float, b: float) -> float:
        c, d = a, b
        return c * d

    a, b = sympy.symbols("a b")
    assert fn_to_sympy(fn, origin="test", model_args=[a, b]) == a * b


def test_condition() -> None:
    def fn(a: float) -> float:
        if a > 1:
            return a
        return a**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_2() -> None:
    def fn(a: float) -> float:
        if a > 1:
            return a
        else:  # noqa: RET505
            return a**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_assignment_single_return() -> None:
    def fn(a: float) -> float:
        if a > 1:  # noqa: SIM108
            b = a
        else:
            b = a**2
        return b

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_elif() -> None:
    def fn(a: float) -> float:
        if 1 < a < 2:
            return a
        elif a > 2:  # noqa: RET505
            return a / 2
        else:
            return a**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, (a > 1.0) & (a < 2.0)),  # type: ignore
        (a / 2.0, a > 2.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_multiple() -> None:
    def fn(a: float) -> float:
        if 1 < a < 2:
            return a
        if a > 2:
            return a / 2
        return a**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, (a > 1.0) & (a < 2.0)),  # type: ignore
        (a / 2.0, a > 2.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_assignment() -> None:
    def fn(a: float) -> float:
        if a > 1:
            b = a
            return b  # noqa: RET504
        else:  # noqa: RET505
            b = a**2
            return b  # noqa: RET504

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_condition_as_later() -> None:
    def fn(a: float) -> float:
        b = a
        if b > 1:
            return b
        return b**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_conditional_expression() -> None:
    def fn(a: float) -> float:
        return a if a > 1 else a**2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == sympy.Piecewise(
        (a, a > 1.0),  # type: ignore
        (a**2.0, True),
    )


def test_fn_call_power() -> None:
    def fn(a: float) -> float:
        return power(a) * 2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == a**2.0 * 2.0  # type: ignore


def test_fn_call_times() -> None:
    def fn(a: float) -> float:
        return times(a) * 2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == a * 2.0 * 2.0  # type: ignore


def test_fn_call_add() -> None:
    def fn(a: float) -> float:
        return add(a) * 2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == (a + 2.0) * 2.0  # type: ignore


def test_fn_call_sub() -> None:
    def fn(a: float) -> float:
        return sub(a) * 2

    a = sympy.Symbol("a")
    assert fn_to_sympy(fn, origin="test", model_args=[a]) == (a - 2.0) * 2.0  # type: ignore


def test_fn_call_outside_file() -> None:
    def fn(a: float, b: float) -> float:
        return mass_action_1s(a, b)

    a, b = sympy.symbols("a b")
    assert fn_to_sympy(fn, origin="test", model_args=[a, b]) == a * b
