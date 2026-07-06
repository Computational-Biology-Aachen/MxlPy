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


def test_sympy_to_inline_jax_piecewise_uses_where_not_ternary() -> None:
    import sympy

    from mxlpy.meta.sympy_tools import sympy_to_inline_jax

    x, y = sympy.symbols("x y")
    out = sympy_to_inline_jax(sympy.Piecewise((x, sympy.Gt(x, 0)), (y, True)))
    assert "jnp.where" in out
    # a Python ternary (which fails under jax.jit) must not be produced
    assert " if " not in out
    assert "else" not in out


def test_sympy_to_inline_jax_nested_piecewise_nests_where() -> None:
    import sympy

    from mxlpy.meta.sympy_tools import sympy_to_inline_jax

    x = sympy.Symbol("x")
    out = sympy_to_inline_jax(
        sympy.Piecewise((1, sympy.Gt(x, 10)), (2, sympy.Gt(x, 0)), (3, True))
    )
    assert out.count("jnp.where") == 2


def test_sympy_to_inline_jax_piecewise_without_default_uses_nan() -> None:
    import sympy

    from mxlpy.meta.sympy_tools import sympy_to_inline_jax

    x = sympy.Symbol("x")
    out = sympy_to_inline_jax(sympy.Piecewise((x, sympy.Gt(x, 0))))
    assert "jnp.where" in out
    assert "jnp.nan" in out


def test_sympy_to_inline_jax_matches_py_for_non_piecewise() -> None:
    import sympy

    from mxlpy.meta.sympy_tools import sympy_to_inline_jax, sympy_to_inline_py

    x, y = sympy.symbols("x y")
    expr = x**2 + 2 * x * y + sympy.exp(-x)
    assert sympy_to_inline_jax(expr) == sympy_to_inline_py(expr)


def test_sympy_to_inline_rust() -> None:
    # FIXME: implement this
    assert True


def test_sympy_to_python_fn() -> None:
    # FIXME: implement this
    assert True
