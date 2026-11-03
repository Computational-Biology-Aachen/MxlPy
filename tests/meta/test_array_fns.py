# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Array nodes (mxl-schemas ADR 0014): parsing, codegen and data round trips."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import sympy
from scipy.integrate import simpson

from mxlpy import Model, load, save
from mxlpy.meta import generate_model_code_mxlweb, generate_model_data_mxlweb
from mxlpy.meta.array_fns import ArrayMaxOf, ArraySimpson, ArrayStd, ArraySum
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    mathml_to_sympy,
    sympy_to_inline_mxlweb,
    sympy_to_mathml,
)

if TYPE_CHECKING:
    from pathlib import Path


def _absorbed(k: float, light: pd.Series, wavelength: pd.Series) -> float:
    return k * simpson(light * light, x=wavelength)


def _consumption(absorbed: float, x: float) -> float:
    return absorbed * x


def _spread(a: pd.Series) -> float:
    return np.sum(a) / np.std(a, ddof=1) + np.max(a)


def _population_std(a: pd.Series) -> float:
    return np.std(a)


def _symbols(names: str) -> list[sympy.Symbol]:
    return list(sympy.symbols(names, seq=True))


def test_parse_simpson() -> None:
    expr = fn_to_sympy_expr(_absorbed, origin="t", model_args=_symbols("k y w"))
    k, y, w = _symbols("k y w")
    assert expr == k * ArraySimpson(y**2, w)


def test_parse_reductions() -> None:
    expr = fn_to_sympy_expr(_spread, origin="t", model_args=_symbols("a"))
    (a,) = _symbols("a")
    assert expr == ArraySum(a) / ArrayStd(a) + ArrayMaxOf(a)


def test_population_std_is_not_an_array_node() -> None:
    # The node is the sample std (ddof=1); numpy's default ddof=0 isn't it.
    assert (
        fn_to_sympy_expr(_population_std, origin="t", model_args=_symbols("a")) is None
    )


def test_codegen_and_mathml_round_trip() -> None:
    k, y, w = _symbols("k y w")
    expr = k * ArraySimpson(y * w, w)
    used: set[str] = set()
    assert sympy_to_inline_mxlweb(expr, used, None) == (
        'new Mul([new Name("k"), new Simpson([new Mul([new Name("w"), '
        'new Name("y")]), new Name("w")])])'
    )
    assert "Simpson" in used
    tree = sympy_to_mathml(expr)
    assert tree.to_dict()["children"][1]["type"] == "Simpson"
    assert mathml_to_sympy(tree) == expr
    assert sympy.pycode(expr, fully_qualified_modules=True) == (
        "k*scipy.integrate.simpson(w*y, x=w)"
    )


def _spectral_model() -> Model:
    wavelength = pd.Series(np.arange(400.0, 411.0))
    return (
        Model()
        .add_data("wavelength", wavelength)
        .add_data("light", np.exp(-((wavelength - 405) ** 2) / 8))
        .add_parameter("k", 2.0)
        .add_variable("x", 1.0)
        .add_derived("absorbed", _absorbed, args=["k", "light", "wavelength"])
        .add_reaction(
            "v",
            _consumption,
            args=["absorbed", "x"],
            stoichiometry={"x": -1},
        )
    )


def test_mxlweb_codegen_emits_data() -> None:
    model = _spectral_model()
    code = generate_model_code_mxlweb(model)
    assert 'import data from "./model.data.json";' in code
    assert '.addData("light", data["light"])' in code
    assert "new Simpson([" in code
    data = json.loads(generate_model_data_mxlweb(model) or "")
    assert list(data) == ["wavelength", "light"]
    assert data["wavelength"]["values"][:2] == [400.0, 401.0]


def test_mxlweb_codegen_without_data() -> None:
    model = Model().add_parameter("k", 1.0)
    assert generate_model_data_mxlweb(model) is None
    assert "import data" not in generate_model_code_mxlweb(model)


def test_mxlweb_codegen_rejects_tables() -> None:
    model = Model().add_data("table", pd.DataFrame({"a": [1.0]}))
    with pytest.raises(TypeError, match="only 1-D data"):
        generate_model_data_mxlweb(model)


def test_save_load_round_trip(tmp_path: Path) -> None:
    model = _spectral_model()
    save(model, tmp_path / "spectral.mxl.json")
    doc = json.loads((tmp_path / "spectral.mxl.json").read_text())
    assert doc["spec_version"] == "1.2"
    assert doc["data_ref"] == "spectral.data.json"
    assert (tmp_path / "spectral.data.json").exists()

    loaded = load(tmp_path / "spectral.mxl.json")
    expected = model.get_args()
    actual = loaded.get_args()
    for name in ("absorbed", "v"):
        assert actual[name] == pytest.approx(expected[name], rel=1e-12)
