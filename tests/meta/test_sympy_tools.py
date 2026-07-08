import pytest
import sympy

from mxlpy.meta.sympy_tools import sympy_to_inline_jax, sympy_to_inline_py


def test_list_of_symbols() -> None:
    # FIXME: implement this
    assert True


def test_stoichiometries_to_sympy() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_c() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_cxx() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_js() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_julia() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_matlab() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_mxlweb() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_py() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_inline_jax_piecewise_uses_select_not_ternary() -> None:
    x, y = sympy.symbols("x y")
    out = sympy_to_inline_jax(sympy.Piecewise((x, sympy.Gt(x, 0)), (y, True)))
    assert "jnp.select" in out
    # a Python ternary (which fails under jax.jit) must not be produced
    assert " if " not in out
    assert "else" not in out


def test_sympy_to_inline_jax_nested_piecewise_has_one_select() -> None:
    x = sympy.Symbol("x")
    out = sympy_to_inline_jax(
        sympy.Piecewise((1, sympy.Gt(x, 10)), (2, sympy.Gt(x, 0)), (3, True))
    )
    # all branches collapse into a single vectorized select, not nested calls
    assert out.count("jnp.select") == 1
    assert " if " not in out
    assert "else" not in out


def test_sympy_to_inline_jax_piecewise_without_default_uses_nan() -> None:
    x = sympy.Symbol("x")
    out = sympy_to_inline_jax(sympy.Piecewise((x, sympy.Gt(x, 0))))
    assert "jnp.select" in out
    assert "jnp.nan" in out


def test_sympy_to_inline_jax_matches_py_after_normalizing_special_functions() -> None:
    """Anything jax doesn't deliberately special-case must print identically to py."""
    x, y = sympy.symbols("x y")
    expr = x**2 + 2 * x * y + sympy.exp(-x)
    jax_out = sympy_to_inline_jax(expr).replace("jnp.exp", "math.exp")
    assert jax_out == sympy_to_inline_py(expr)


# ---------------------------------------------------------------------------
# Special functions that need jax-specific translation (jnp.* / jax.scipy.special.*
# instead of math.*/bare builtins), and must stay untouched for the plain
# Python/mxlpy export.
# ---------------------------------------------------------------------------

_UNARY_JAX_FUNCTIONS: list[tuple[str, "sympy.FunctionClass", str, str]] = [
    ("exp", sympy.exp, "jnp.exp(x)", "math.exp(x)"),
    ("log", sympy.log, "jnp.log(x)", "math.log(x)"),
    ("sin", sympy.sin, "jnp.sin(x)", "math.sin(x)"),
    ("cos", sympy.cos, "jnp.cos(x)", "math.cos(x)"),
    ("tan", sympy.tan, "jnp.tan(x)", "math.tan(x)"),
    ("asin", sympy.asin, "jnp.arcsin(x)", "math.asin(x)"),
    ("acos", sympy.acos, "jnp.arccos(x)", "math.acos(x)"),
    ("atan", sympy.atan, "jnp.arctan(x)", "math.atan(x)"),
    ("sinh", sympy.sinh, "jnp.sinh(x)", "math.sinh(x)"),
    ("cosh", sympy.cosh, "jnp.cosh(x)", "math.cosh(x)"),
    ("tanh", sympy.tanh, "jnp.tanh(x)", "math.tanh(x)"),
    ("floor", sympy.floor, "jnp.floor(x)", "math.floor(x)"),
    ("ceiling", sympy.ceiling, "jnp.ceil(x)", "math.ceil(x)"),
    ("sign", sympy.sign, "jnp.sign(x)", "math.copysign(1, x)"),
    ("erf", sympy.erf, "jax.scipy.special.erf(x)", "math.erf(x)"),
    ("erfc", sympy.erfc, "jax.scipy.special.erfc(x)", "math.erfc(x)"),
    ("gamma", sympy.gamma, "jax.scipy.special.gamma(x)", "math.gamma(x)"),
    ("loggamma", sympy.loggamma, "jax.scipy.special.gammaln(x)", "math.lgamma(x)"),
    ("factorial", sympy.factorial, "jax.scipy.special.factorial(x)", "math.factorial(x)"),
]


@pytest.mark.parametrize(
    ("name", "fn", "jax_expected", "py_expected"),
    _UNARY_JAX_FUNCTIONS,
    ids=[name for name, *_ in _UNARY_JAX_FUNCTIONS],
)
def test_unary_jax_translation(
    name: str,
    fn: "sympy.FunctionClass",
    jax_expected: str,
    py_expected: str,
) -> None:
    x = sympy.Symbol("x")
    expr = fn(x)
    assert jax_expected in sympy_to_inline_jax(expr), name
    assert py_expected in sympy_to_inline_py(expr), name


def test_sqrt_jax_uses_jnp_sqrt() -> None:
    x = sympy.Symbol("x")
    expr = sympy.sqrt(x)
    assert sympy_to_inline_jax(expr) == "jnp.sqrt(x)"
    assert sympy_to_inline_py(expr) == "math.sqrt(x)"


def test_atan2_jax_uses_jnp_arctan2() -> None:
    x, y = sympy.symbols("x y")
    expr = sympy.atan2(x, y)
    assert sympy_to_inline_jax(expr) == "jnp.arctan2(x, y)"
    assert sympy_to_inline_py(expr) == "math.atan2(x, y)"


def test_mod_jax_uses_jnp_mod() -> None:
    x, y = sympy.symbols("x y")
    expr = sympy.Mod(x, y)
    assert sympy_to_inline_jax(expr) == "jnp.mod(x, y)"
    assert sympy_to_inline_py(expr) == "x % y"


def test_min_jax_uses_jnp_minimum_not_builtin() -> None:
    x, y = sympy.symbols("x y")
    expr = sympy.Min(x, y)
    jax_out = sympy_to_inline_jax(expr)
    assert jax_out == "functools.reduce(jnp.minimum, [x, y])"
    assert sympy_to_inline_py(expr) == "min(x, y)"


def test_max_jax_uses_jnp_maximum_not_builtin() -> None:
    x, y = sympy.symbols("x y")
    expr = sympy.Max(x, y)
    jax_out = sympy_to_inline_jax(expr)
    assert jax_out == "functools.reduce(jnp.maximum, [x, y])"
    assert sympy_to_inline_py(expr) == "max(x, y)"


def test_min_jax_is_nary() -> None:
    x, y, z = sympy.symbols("x y z")
    expr = sympy.Min(x, y, z)
    assert sympy_to_inline_jax(expr) == "functools.reduce(jnp.minimum, [x, y, z])"
    assert sympy_to_inline_py(expr) == "min(x, y, z)"


def test_abs_is_unchanged_builtin_for_both_backends() -> None:
    """abs() already works correctly under jax tracing; no jnp.abs needed."""
    x = sympy.Symbol("x")
    expr = sympy.Abs(x)
    assert sympy_to_inline_jax(expr) == "abs(x)"
    assert sympy_to_inline_py(expr) == "abs(x)"


def test_sympy_to_inline_rust() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_python_fn() -> None:
    # FIXME: implement this
    assert True
