# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""SymPy expressions as an alternative to python functions + `args` (#157).

Expressions are compiled into regular, introspectable python functions when
they are added, so every downstream operation has to work exactly as for a
hand-written function.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import sympy

import mxlpy
from mxlpy import (
    Derived,
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    SteadyStateModelBuilder,
    fns,
    meta,
    to_symbolic_model,
)
from mxlpy.dsl import from_dsl
from mxlpy.meta.source_tools import fn_to_sympy_expr

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

S = sympy.Symbol("S")
P = sympy.Symbol("P")
b_h = sympy.Symbol("b_H")
kf_1 = sympy.Symbol("kf_1")
km_1 = sympy.Symbol("km_1")
vmax = sympy.Symbol("vmax")

PARAMETERS = {"b_H": 2.0, "kf_1": 1.5, "km_1": 0.5, "vmax": 1.0}
VARIABLES = {"S": 1.0, "P": 0.0}


def _mm(s: float, km: float, vmax: float) -> float:
    return vmax * s / (km + s)


def _four_divided_value(x: float) -> float:
    return 4 / x


def fn_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters(PARAMETERS)
        .add_variables(VARIABLES)
        .add_derived("keq", fns.div, args=["kf_1", "km_1"])
        .add_reaction(
            "v1",
            _mm,
            args=["S", "km_1", "vmax"],
            stoichiometry={
                "S": -1,
                "P": Derived(fn=_four_divided_value, args=["b_H"]),
            },
        )
        .add_reaction(
            "v2",
            fns.mass_action_1s,
            args=["P", "keq"],
            stoichiometry={"P": -1},
        )
        .add_readout("ratio", fns.div, args=["P", "S"])
    )


def expr_model() -> KineticModelBuilder:
    keq = sympy.Symbol("keq")
    return (
        KineticModelBuilder()
        .add_parameters(PARAMETERS)
        .add_variables(VARIABLES)
        .add_derived("keq", kf_1 / km_1)
        .add_reaction(
            "v1",
            vmax * S / (km_1 + S),
            stoichiometry={"S": -1, "P": 4 / b_h},
        )
        .add_reaction("v2", P * keq, stoichiometry={"P": -1})
        .add_readout("ratio", P / S)
    )


def _simulate(model: KineticModelBuilder) -> pd.DataFrame:
    res = Simulator(model).simulate(5).get_result().unwrap_or_err()
    return res.get_combined()


def _assert_same_simulation(
    model: KineticModelBuilder, reference: KineticModelBuilder
) -> None:
    expected = _simulate(reference)
    columns = sorted(expected.columns)
    pd.testing.assert_frame_equal(
        _simulate(model).reindex(columns=columns),
        expected.reindex(columns=columns),
        rtol=1e-8,
    )


def _exec_create_model(source: str) -> KineticModelBuilder:
    namespace: dict[str, Callable[[], KineticModelBuilder]] = {}
    exec(source, namespace)  # noqa: S102
    return namespace["create_model"]()


###############################################################################
# Building
###############################################################################


def test_expr_args_are_free_symbols() -> None:
    model = expr_model()
    assert model.get_raw_derived()["keq"].args == ["kf_1", "km_1"]
    assert model.get_raw_reactions()["v1"].args == ["S", "km_1", "vmax"]
    assert model.get_raw_readouts()["ratio"].args == ["P", "S"]


def test_expr_stoichiometry() -> None:
    stoich = expr_model().get_raw_reactions()["v1"].stoichiometry
    assert stoich["S"] == -1
    assert isinstance(stoich["P"], Derived)
    assert stoich["P"].args == ["b_H"]
    assert stoich["P"].calculate({"b_H": 2.0}) == pytest.approx(2.0)


def test_constant_expr_stoichiometry_is_float() -> None:
    model = KineticModelBuilder().add_variables(VARIABLES)
    model.add_reaction(
        "v", S, stoichiometry={"S": -sympy.Integer(2), "P": sympy.Rational(1, 2)}
    )
    stoich = model.get_raw_reactions()["v"].stoichiometry
    assert stoich == {"S": -2.0, "P": 0.5}
    assert all(type(v) is float for v in stoich.values())


def test_lambda_maps_template_symbols_to_model_names() -> None:
    k1, km = sympy.symbols("k1 km")
    template = sympy.Lambda((km, k1), k1 / km)
    model = (
        KineticModelBuilder()
        .add_parameters(PARAMETERS)
        .add_derived("keq", template(kf_1, km_1))
    )
    # km -> kf_1, k1 -> km_1, so keq = km_1 / kf_1
    assert model.get_args().to_dict()["keq"] == pytest.approx(0.5 / 1.5)


# The two misuses below are static type errors (see `test_static_typing`), the
# runtime checks only guard untyped callers.


def test_expr_with_args_raises() -> None:
    model: Any = KineticModelBuilder()
    with pytest.raises(TypeError, match="can't be combined with a sympy expression"):
        model.add_derived("keq", kf_1 / km_1, args=["kf_1", "km_1"])


def test_fn_without_args_raises() -> None:
    model: Any = KineticModelBuilder()
    with pytest.raises(TypeError, match="`args` are required"):
        model.add_derived("keq", fns.div)


def test_update_with_expr() -> None:
    model = fn_model()
    model.update_derived("keq", kf_1 * km_1)
    model.update_reaction("v2", stoichiometry={"P": -b_h})
    assert model.get_raw_derived()["keq"].args == ["kf_1", "km_1"]
    assert model.get_args().to_dict()["keq"] == pytest.approx(0.75)
    stoich = model.get_raw_reactions()["v2"].stoichiometry["P"]
    assert isinstance(stoich, Derived)
    assert stoich.calculate({"b_H": 2.0}) == pytest.approx(-2.0)


def test_update_without_fn_keeps_fn() -> None:
    model = expr_model()
    fn = model.get_raw_derived()["keq"].fn
    model.update_derived("keq", display_name="Keq")
    assert model.get_raw_derived()["keq"].fn is fn
    assert model.get_raw_derived()["keq"].args == ["kf_1", "km_1"]


def test_ode_builder_expr() -> None:
    x, k = sympy.Symbol("x"), sympy.Symbol("k")
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_derived("kk", 2 * k)
        .add_diff_eq("x", initial_value=2.0, fn=-k * x)
        .add_readout("x2", x**2)
    )
    assert model(0.0, [2.0]) == (-1.0,)
    model.update_diff_eq("x", initial_value=2.0, fn=-sympy.Symbol("kk") * x)
    assert model(0.0, [2.0]) == (-2.0,)


def test_steady_state_builder_expr() -> None:
    k = sympy.Symbol("k")
    model = (
        SteadyStateModelBuilder().add_parameter("k", 2.0).add_derived("d1", 3 * k**2)
    )
    assert model.evaluate().to_dict()["d1"] == pytest.approx(12.0)


###############################################################################
# Downstream operations behave as for python functions
###############################################################################


def test_simulates_like_fn_model() -> None:
    _assert_same_simulation(expr_model(), fn_model())


def test_mxlpy_codegen_roundtrip() -> None:
    source = meta.generate_mxlpy_code(expr_model())
    # every expression got its own, distinct function
    assert "def _keq(" in source
    assert "def _v1(" in source
    assert "def _v1_P(" in source
    _assert_same_simulation(_exec_create_model(source), fn_model())


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    mxlpy.save(expr_model(), path)
    loaded = mxlpy.load(path)
    assert isinstance(loaded, KineticModelBuilder)
    _assert_same_simulation(loaded, fn_model())


def test_cloudpickle_roundtrip() -> None:
    restored = cloudpickle.loads(cloudpickle.dumps(expr_model()))
    _assert_same_simulation(restored, fn_model())


def test_symbolic_model_matches_fn_model() -> None:
    expr_eqs = to_symbolic_model(expr_model()).eqs
    fn_eqs = to_symbolic_model(fn_model()).eqs
    assert [sympy.simplify(a - b) for a, b in zip(expr_eqs, fn_eqs, strict=True)] == [
        0,
        0,
    ]


def _exp_log_model() -> KineticModelBuilder:
    x, k = sympy.symbols("x k")
    return (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v", k * sympy.exp(-x) + sympy.log(k + 1), stoichiometry={"x": -1}
        )
    )


def test_fn_to_sympy_expr_uses_stored_expr() -> None:
    x, k, a, b = sympy.symbols("x k a b")
    fn = _exp_log_model().get_raw_reactions()["v"].fn
    expected = k * sympy.exp(-x) + sympy.log(k + 1)
    assert fn_to_sympy_expr(fn, origin="v") == expected
    assert fn_to_sympy_expr(fn, origin="v", model_args=[a, b]) == expected.xreplace(
        {k: a, x: b}
    )


def test_generated_fn_with_module_calls_parses_back() -> None:
    """Without the stored expression, the source (`math.exp`, `math.log`) is parsed (#202)."""
    x, k = sympy.symbols("x k")
    fn = _exp_log_model().get_raw_reactions()["v"].fn
    del fn.__mxlpy_expr__
    parsed = fn_to_sympy_expr(fn, origin="v")
    assert parsed is not None
    expected = k * sympy.exp(-x) + sympy.log(k + 1)
    point = {x: 0.3, k: 1.7}
    assert float(parsed.subs(point)) == pytest.approx(float(expected.subs(point)))


@pytest.mark.parametrize(
    "codegen",
    [
        meta.generate_model_code_latex,
        meta.generate_model_code_py,
        meta.generate_model_code_jax,
        meta.generate_model_code_ts,
    ],
)
def test_codegen_matches_fn_model(codegen: Callable) -> None:
    # Expressions keep their exact numbers (`4`), parsed source has floats (`4.0`)
    def strip_float_zeros(source: str) -> str:
        return re.sub(r"(\d)\.0\b", r"\1", source)

    assert strip_float_zeros(codegen(expr_model()).full()) == strip_float_zeros(
        codegen(fn_model()).full()
    )


def test_sbml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "m.xml"
    mxlpy.sbml.write(expr_model(), path)
    restored = mxlpy.sbml.read(path)
    expected = _simulate(fn_model())
    actual = _simulate(restored)
    for name in VARIABLES:
        assert np.allclose(actual[name], expected[name], rtol=1e-6)


###############################################################################
# DSL
###############################################################################


@pytest.mark.parametrize("name", ["E", "S", "beta", "N"])
def test_dsl_names_shadowing_sympy_objects(name: str) -> None:
    model = from_dsl(
        KineticModelBuilder(),
        f"k1 * {name}, {name} --> B",
        variables={name: 1.0, "B": 0.0},
        parameters={"k1": 0.5},
    )
    assert sorted(model.get_raw_reactions()["r1"].args) == sorted([name, "k1"])
    assert model.get_args().to_dict()["r1"] == pytest.approx(0.5)


###############################################################################
# Static typing
###############################################################################


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_static_typing(tmp_path: Path) -> None:
    snippet = tmp_path / "snippet.py"
    snippet.write_text(
        textwrap.dedent(
            """
            import sympy
            from mxlpy import KineticModelBuilder, fns

            k = sympy.Symbol("k")
            m = KineticModelBuilder()
            m.add_derived("ok_fn", fns.div, args=["a", "b"])
            m.add_derived("ok_expr", 2 * k)
            m.add_reaction("ok_rxn", 2 * k, stoichiometry={"x": 4 / k})
            m.add_derived("bad_missing_args", fns.div)
            m.add_derived("bad_expr_with_args", 2 * k, args=["k"])
            """
        )
    )
    res = subprocess.run(  # noqa: S603
        [shutil.which("pyright") or "pyright", str(snippet)],
        capture_output=True,
        text=True,
        check=False,
    )
    error_lines = {
        int(line.split(":")[1])
        for line in res.stdout.splitlines()
        if " - error: " in line and str(snippet) in line
    }
    assert error_lines == {10, 11}, res.stdout
