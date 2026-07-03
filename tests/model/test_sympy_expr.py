from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import sympy

from mxlpy import KineticModelBuilder

# `add_derived` / `add_reaction` / `add_readout` take an explicit `args` list
# mapping *model* names onto the positional slots of `fn`. For a plain
# function this is unambiguous (the function's own signature order). For a
# `sympy.Expr` there is no declared signature, so mxlpy has to invent a
# canonical order for the expression's free symbols. Using `Expr.free_symbols`
# (a `set`) means that order depends on Python's per-process string hash
# seed, so the *same* model can silently compute different results run to
# run.


def _derived_value_subprocess(seed: str) -> str:
    script = textwrap.dedent(
        """
        import sympy
        from mxlpy import KineticModelBuilder

        k1, k2 = sympy.symbols("k1 k2")
        m = (
            KineticModelBuilder()
            .add_parameters({"k1": 3.0, "k2": 5.0})
            .add_derived("d", k1 - k2, args=["k1", "k2"])
        )
        print(m.get_args()["d"])
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed script, trusted interpreter path
        [sys.executable, "-c", script],
        env={"PYTHONHASHSEED": seed, "PATH": sys.exec_prefix},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_add_derived_expr_result_is_independent_of_hash_seed() -> None:
    """k1 - k2 must evaluate to -2.0 regardless of PYTHONHASHSEED.

    `k1 - k2` has a single unambiguous value (3.0 - 5.0 == -2.0). If the
    order in which mxlpy binds `args` to the expression's free symbols is
    derived from set iteration order, different hash seeds can bind
    k1 <-> k2 the wrong way round and silently produce 2.0 instead.
    """
    results = {_derived_value_subprocess(seed) for seed in ("0", "1", "2", "42")}
    assert results == {"-2.0"}


def test_add_derived_expr_args_are_positional_placeholders() -> None:
    """`args` renames the expression's symbols, like for a plain function.

    For a plain python function, `args` binds model names to the function's
    parameters *by position* - the function's own parameter names are
    irrelevant and don't need to exist in the model. The same must hold for
    a `sympy.Expr`: `k1 - k2` used with `args=["a1", "a2"]` must compute
    `a1 - a2` using the model's `a1`/`a2`, even though the model has no
    `k1`/`k2` at all.
    """
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        KineticModelBuilder()
        .add_parameters({"a1": 3.0, "a2": 5.0})
        .add_derived("d", k1 - k2, args=["a1", "a2"])
    )

    assert m.get_args()["d"] == 3.0 - 5.0


def test_add_reaction_expr_args_are_positional_placeholders() -> None:
    """Same placeholder semantics for `add_reaction`'s `fn`.

    `args` binds to the expression's free symbols in **alphabetical order
    of the symbol names** (km, s, vmax) - not the order they were written
    in the expression, which sympy does not preserve for commutative ops.
    """
    s, vmax, km = sympy.symbols("s vmax km")
    m = (
        KineticModelBuilder()
        .add_parameters({"my_vmax": 2.0, "my_km": 0.5})
        .add_variables({"my_s": 1.0})
        .add_reaction(
            "v1",
            vmax * s / (km + s),
            args=["my_km", "my_s", "my_vmax"],
            stoichiometry={"my_s": -1},
        )
    )

    expected = 2.0 * 1.0 / (0.5 + 1.0)
    assert m.get_fluxes()["v1"] == expected


def test_add_derived_expr_arg_count_mismatch_raises() -> None:
    """A clear error beats a silent, wrong positional binding."""
    k1, k2 = sympy.symbols("k1 k2")
    m = KineticModelBuilder().add_parameters({"a1": 1.0})
    with pytest.raises(ValueError, match="free symbol"):
        m.add_derived("d", k1 - k2, args=["a1"])


def test_update_derived_expr_keeps_existing_args_by_default() -> None:
    """Re-fitting a new expression without `args=` reuses the current args."""
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        KineticModelBuilder()
        .add_parameters({"a1": 3.0, "a2": 5.0})
        .add_derived("d", k1 + k2, args=["a1", "a2"])
    )
    assert m.get_args()["d"] == 3.0 + 5.0

    m.update_derived("d", fn=k1 - k2)  # no args= given, keeps ["a1", "a2"]
    assert m.get_args()["d"] == 3.0 - 5.0
