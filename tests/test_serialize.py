"""Tests for native JSON model serialization (mxlpy.save / mxlpy.load)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sympy
from numpy.polynomial import polynomial

import mxlpy
from example_models import (
    get_linear_chain_2v,
    get_lotka_volterra,
    get_sir,
    get_sird,
    get_tpi_ald_model,
    get_upper_glycolysis,
)
from mxlpy import Derived, InitialAssignment, Model, fns
from mxlpy.meta import _mathml as mml
from mxlpy.meta.sympy_tools import mathml_to_sympy, sympy_to_mathml
from mxlpy.serialize import model_from_dict, model_to_dict
from mxlpy.surrogates._poly import Surrogate
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

EXAMPLE_MODELS = [
    get_linear_chain_2v,
    get_lotka_volterra,
    get_sir,
    get_sird,
    get_tpi_ald_model,
    get_upper_glycolysis,
]


def _combined(model: Model) -> pd.DataFrame:
    res = mxlpy.Simulator(model).simulate(10).get_result().unwrap_or_err()
    return res.get_combined()


###############################################################################
# Node tree <-> dict
###############################################################################


@pytest.mark.parametrize(
    "node",
    [
        mml.Num(value=1.5),
        mml.Name(name="k1"),
        mml.Pi(),
        mml.E(),
        mml.Bool(value=True),
        mml.Bool(value=False),
        mml.Sin(mml.Name(name="x")),
        mml.Pow(left=mml.Name(name="x"), right=mml.Num(value=2.0)),
        mml.Sqrt(child=mml.Name(name="x"), base=mml.Num(value=2.0)),
        mml.Log(child=mml.Name(name="x"), base=mml.Num(value=10.0)),
        mml.Add([mml.Name(name="x"), mml.Num(value=1.0)]),
        mml.Mul([mml.Name(name="x"), mml.Name(name="y"), mml.Num(value=3.0)]),
        mml.Piecewise([mml.Num(value=1.0), mml.Name(name="cond"), mml.Num(value=0.0)]),
    ],
)
def test_node_dict_roundtrip(node: mml.Base) -> None:
    assert mml.node_from_dict(node.to_dict()) == node


def test_node_leaf_dict_shapes() -> None:
    assert mml.Num(value=2.0).to_dict() == {"type": "Num", "value": 2.0}
    assert mml.Name(name="k1").to_dict() == {"type": "Name", "value": "k1"}
    assert mml.Pi().to_dict() == {"type": "Pi"}
    assert mml.Bool(value=True).to_dict() == {"type": "Bool", "value": True}


def test_node_arity_dict_shapes() -> None:
    assert mml.Sin(mml.Name(name="x")).to_dict() == {
        "type": "Sin",
        "child": {"type": "Name", "value": "x"},
    }
    assert mml.Pow(left=mml.Name(name="x"), right=mml.Num(value=2.0)).to_dict() == {
        "type": "Pow",
        "left": {"type": "Name", "value": "x"},
        "right": {"type": "Num", "value": 2.0},
    }
    assert mml.Add([mml.Num(value=1.0)]).to_dict() == {
        "type": "Add",
        "children": [{"type": "Num", "value": 1.0}],
    }


def test_node_from_dict_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown expression node type"):
        mml.node_from_dict({"type": "Nope"})


###############################################################################
# sympy <-> node tree
###############################################################################


@pytest.mark.parametrize(
    "expr",
    [
        sympy.Symbol("k1") * sympy.Symbol("A"),
        sympy.Symbol("k1") * sympy.Symbol("A") - sympy.Symbol("k2"),
        sympy.Symbol("x") ** 2,
        sympy.sqrt(sympy.Symbol("x")),
        sympy.Symbol("a") / sympy.Symbol("b"),
        -sympy.Symbol("x"),
        sympy.exp(sympy.Symbol("x")) + sympy.sin(sympy.Symbol("y")),
        sympy.Max(sympy.Symbol("x"), sympy.Symbol("y")),
        sympy.Piecewise(
            (sympy.Symbol("x"), sympy.Symbol("x") > 0), (sympy.Integer(0), True)
        ),
    ],
)
def test_sympy_tree_roundtrip(expr: sympy.Expr) -> None:
    restored = mathml_to_sympy(sympy_to_mathml(expr))
    assert sympy.simplify(restored - expr) == 0


def test_sympy_constants_roundtrip() -> None:
    assert sympy_to_mathml(sympy.pi) == mml.Pi()
    assert sympy_to_mathml(sympy.E) == mml.E()
    assert mathml_to_sympy(mml.Pi()) == sympy.pi
    assert mathml_to_sympy(mml.E()) == sympy.E


###############################################################################
# Full model round-trip
###############################################################################


@pytest.mark.parametrize("getter", EXAMPLE_MODELS)
def test_example_model_roundtrip_simulates_identically(getter) -> None:  # noqa: ANN001
    model = getter()
    spec = model_to_dict(model, model_id="m")
    restored = model_from_dict(spec)

    original = _combined(model)
    loaded = _combined(restored)
    common = sorted(set(original.columns) & set(loaded.columns))
    assert set(original.columns) == set(loaded.columns)
    assert np.allclose(
        original[common].to_numpy(), loaded[common].to_numpy(), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("getter", EXAMPLE_MODELS)
def test_save_load_save_is_idempotent(getter, tmp_path: Path) -> None:  # noqa: ANN001
    model = getter()
    first = tmp_path / "a.mxl.json"
    second = tmp_path / "b.mxl.json"

    mxlpy.save(model, first, model_id="m")
    mxlpy.save(mxlpy.load(first), second, model_id="m")

    assert first.read_text() == second.read_text()


def test_save_writes_schema_and_default_model_id(tmp_path: Path) -> None:
    path = tmp_path / "glycolysis.mxl.json"
    mxlpy.save(get_lotka_volterra(), path)
    data = json.loads(path.read_text())

    assert data["$schema"].endswith("mxl-model.schema.json")
    assert data["spec_version"] == "1.0"
    assert data["model_id"] == "glycolysis"
    assert set(data["model"]) == {
        "variables",
        "parameters",
        "reactions",
        "derived",
        "readouts",
    }


def test_initial_assignment_roundtrip() -> None:
    model = (
        Model()
        .add_parameters({"p1": 2.0, "p2": 3.0})
        .add_variable("v1", InitialAssignment(fn=fns.mul, args=["p1", "p2"]))
    )
    restored = model_from_dict(model_to_dict(model, model_id="m"))

    raw = restored.get_raw_variables()["v1"]
    assert isinstance(raw.initial_value, InitialAssignment)
    assert restored.get_initial_conditions()["v1"] == pytest.approx(6.0)


def test_dynamic_stoichiometry_roundtrip() -> None:
    model = (
        Model()
        .add_variables({"v1": 1.0, "v2": 1.0})
        .add_parameters({"p1": 1.0, "k": 2.0})
        .add_reaction(
            "r1",
            fn=fns.mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0, "v2": Derived(fn=fns.mul, args=["k", "p1"])},
        )
    )
    restored = model_from_dict(model_to_dict(model, model_id="m"))
    stoich = restored.get_raw_reactions()["r1"].stoichiometry
    assert isinstance(stoich["v2"], Derived)
    assert stoich["v1"] == pytest.approx(-1.0)


def test_readout_roundtrip() -> None:
    model = (
        Model()
        .add_variables({"v1": 2.0, "v2": 4.0})
        .add_readout("ratio", fn=fns.div, args=["v1", "v2"])
    )
    restored = model_from_dict(model_to_dict(model, model_id="m"))
    assert "ratio" in restored.get_raw_readouts()


###############################################################################
# Error handling
###############################################################################


def test_surrogate_raises_serialization_error() -> None:
    surrogate = Surrogate(
        model=polynomial.Polynomial([0, 1]),
        args=["v1"],
        outputs=["o1"],
        stoichiometries={},
    )
    model = Model().add_variable("v1", 1.0).add_surrogate("s1", surrogate)
    with pytest.raises(SerializationError, match="surrogates"):
        model_to_dict(model, model_id="m")
