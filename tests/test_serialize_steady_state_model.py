"""Tests for SteadyStateModelBuilder's .mxl.json serialisation (steady-state-model.schema.json, kind="steady-state").

Mirrors `test_serialize_ode_model.py`'s coverage of `OdeModelBuilder`, but for
the steady-state variant: there are no state variables, reactions, readouts,
NN blocks or time integration at all — only parameters and the algebraic
`derived` quantities computed from them (see
`SteadyStateModelBuilder`'s own docstring and `mxl-schemas`'
`steady-state-model.schema.json`, whose `model` section forbids any other
key).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mxlpy
from mxlpy import InitialAssignment, SteadyStateModelBuilder
from mxlpy.serialize import (
    steady_state_model_from_dict,
    steady_state_model_to_dict,
)

if TYPE_CHECKING:
    from pathlib import Path


def _double(p: float) -> float:
    return 2.0 * p


def _sum(p1: float, p2: float) -> float:
    return p1 + p2


def _make_model() -> SteadyStateModelBuilder:
    return (
        SteadyStateModelBuilder()
        .add_parameter("p", 2.0)
        .add_derived("d", fn=_double, args=["p"])
    )


def test_steady_state_model_to_dict_has_kind_and_only_parameters_and_derived() -> None:
    data = steady_state_model_to_dict(_make_model(), model_id="m")
    assert data["kind"] == "steady-state"
    assert data["spec_version"] == "1.0"
    assert set(data["model"]) == {"parameters", "derived"}
    assert "value" in data["model"]["parameters"]["p"]
    assert "fn" in data["model"]["derived"]["d"]


def test_steady_state_model_from_dict_round_trips_derived() -> None:
    model = _make_model()
    restored = steady_state_model_from_dict(
        steady_state_model_to_dict(model, model_id="m")
    )
    assert isinstance(restored, SteadyStateModelBuilder)

    original_p = model.get_raw_parameters()["p"].value
    restored_p = restored.get_raw_parameters()["p"].value
    assert restored_p == original_p

    original_d = model.get_raw_derived()["d"]
    restored_d = restored.get_raw_derived()["d"]
    args = {"p": 3.0}
    assert restored_d.calculate(args) == original_d.calculate(args)


def test_steady_state_model_from_dict_round_trips_initial_assignment_parameter() -> (
    None
):
    model = (
        SteadyStateModelBuilder()
        .add_parameter("p1", 2.0)
        .add_parameter("p2", InitialAssignment(fn=_double, args=["p1"]))
    )
    restored = steady_state_model_from_dict(
        steady_state_model_to_dict(model, model_id="m")
    )

    original = model.get_raw_parameters()["p2"].value
    reloaded = restored.get_raw_parameters()["p2"].value
    assert isinstance(reloaded, InitialAssignment)
    args = {"p1": 5.0}
    assert isinstance(original, InitialAssignment)
    assert reloaded.calculate(args) == original.calculate(args)


def test_save_load_round_trips_steady_state_model(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")

    data = json.loads(path.read_text())
    assert data["kind"] == "steady-state"

    reloaded = mxlpy.load(path)
    assert isinstance(reloaded, SteadyStateModelBuilder)
    assert (
        reloaded.get_raw_parameters()["p"].value
        == model.get_raw_parameters()["p"].value
    )


def test_save_load_save_is_idempotent_steady_state(tmp_path: Path) -> None:
    model = _make_model()
    first = tmp_path / "a.mxl.json"
    second = tmp_path / "b.mxl.json"

    mxlpy.save(model, first, model_id="m")
    mxlpy.save(mxlpy.load(first), second, model_id="m")

    doc_a = json.loads(first.read_text())
    doc_b = json.loads(second.read_text())
    assert doc_a == doc_b


def test_save_writes_no_weights_sidecar(tmp_path: Path) -> None:
    model = _make_model()
    path = tmp_path / "model.mxl.json"
    mxlpy.save(model, path, model_id="m")

    assert list(tmp_path.iterdir()) == [path]


def test_multi_parameter_derived_round_trips() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("p1", 2.0)
        .add_parameter("p2", 3.0)
        .add_derived("total", fn=_sum, args=["p1", "p2"])
    )
    restored = steady_state_model_from_dict(
        steady_state_model_to_dict(model, model_id="m")
    )
    original_total = model.get_raw_derived()["total"]
    restored_total = restored.get_raw_derived()["total"]
    args = {"p1": 4.0, "p2": 5.0}
    assert restored_total.calculate(args) == original_total.calculate(args)
