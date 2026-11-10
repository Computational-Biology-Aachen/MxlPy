# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import mxlpy
from mxlpy import Derived, KineticModelBuilder, Slider, fns
from mxlpy.meta import generate_mxlpy_code

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _exec_create_model(model: KineticModelBuilder) -> KineticModelBuilder:
    namespace: dict[str, Callable[[], KineticModelBuilder]] = {}
    exec(generate_mxlpy_code(model), namespace)  # noqa: S102
    return namespace["create_model"]()


def test_generate_mxlpy_code() -> None:
    # FIXME: implement this
    assert True


def test_generate_mxlpy_code_from_symbolic_repr() -> None:
    # FIXME: implement this
    assert True


def test_generate_mxlpy_code_names() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("k", 1.0, display_name="Rate", tex_name="k_1")
        .add_variable(
            "x",
            2.0,
            display_name="Substrate",
            slider=Slider(min="0", max="5", step="1"),
        )
        .add_variable("y", 3.0)
    )
    restored = _exec_create_model(model)

    parameter = restored.get_raw_parameters()["k"]
    assert (parameter.display_name, parameter.tex_name) == ("Rate", "k_1")
    assert restored.get_raw_variables()["x"].display_name == "Substrate"
    assert restored.get_raw_variables()["x"].slider == Slider(
        min="0", max="5", step="1"
    )
    assert restored.get_raw_variables()["y"].display_name is None


def test_generate_mxlpy_code_distinct_fns_with_same_name(tmp_path: Path) -> None:
    # `mxlpy.load` revives every function under the same `__name__`, which
    # previously made codegen emit only the last body for all of them.
    model = (
        KineticModelBuilder()
        .add_variables({"S": 1.0, "P": 0.0})
        .add_parameters({"k1": 2.0, "km": 0.5, "n": 3.0})
        .add_derived("keq", fns.div, args=["k1", "km"])
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1, "P": Derived(fn=fns.add, args=["n", "km"])},
        )
    )
    path = tmp_path / "m.json"
    mxlpy.save(model, path)
    loaded = mxlpy.load(path)
    assert isinstance(loaded, KineticModelBuilder)
    restored = _exec_create_model(loaded)

    args = restored.get_args().to_dict()
    assert args["keq"] == pytest.approx(4.0)
    assert args["v1"] == pytest.approx(2.0)
    stoich = restored.get_raw_reactions()["v1"].stoichiometry["P"]
    assert isinstance(stoich, Derived)
    assert stoich.calculate({"n": 3.0, "km": 0.5}) == pytest.approx(3.5)
