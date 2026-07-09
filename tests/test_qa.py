"""Tests for the qa module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mxlpy import KineticModelBuilder, Simulator, fns, qa
from mxlpy.types import IntegrationFailure, Result
from mxlpy.units import mol, second

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def linear_chain() -> KineticModelBuilder:
    """A -> B -> open chain with a source and a sink; not mass-conserving."""
    return (
        KineticModelBuilder()
        .add_variables({"A": 5.0, "B": 2.0})
        .add_parameters({"k1": 1.0, "k2": 0.5, "k_in": 2.0})
        .add_reaction("v_in", fn=fns.constant, args=["k_in"], stoichiometry={"A": 1.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["A", "k1"],
            stoichiometry={"A": -1.0, "B": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["B", "k2"],
            stoichiometry={"B": -1.0},
        )
    )


def closed_chain() -> KineticModelBuilder:
    """A <-> B <-> C closed loop; total A + B + C is conserved."""
    return (
        KineticModelBuilder()
        .add_variables({"A": 1.0, "B": 1.0, "C": 1.0})
        .add_parameters({"k1": 1.0, "k2": 1.0, "k3": 1.0, "k4": 1.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["A", "k1"],
            stoichiometry={"A": -1.0, "B": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["B", "k2"],
            stoichiometry={"B": -1.0, "A": 1.0},
        )
        .add_reaction(
            "v3",
            fn=fns.mass_action_1s,
            args=["B", "k3"],
            stoichiometry={"B": -1.0, "C": 1.0},
        )
        .add_reaction(
            "v4",
            fn=fns.mass_action_1s,
            args=["C", "k4"],
            stoichiometry={"C": -1.0, "B": 1.0},
        )
    )


def unitful_chain() -> KineticModelBuilder:
    """A -> B with consistent mole/second units throughout."""
    return (
        KineticModelBuilder()
        .add_variable("A", 5.0, unit=mol)
        .add_variable("B", 2.0, unit=mol)
        .add_parameter("k1", 1.0, unit=1 / second)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["A", "k1"],
            stoichiometry={"A": -1.0, "B": 1.0},
            unit=mol / second,
        )
    )


###############################################################################
# report structure
###############################################################################


def test_report_returns_expected_metric_names() -> None:
    result = qa.report(linear_chain())
    names = [m.name for m in result.metrics]
    assert names == [
        "State variables",
        "Parameters",
        "Reactions",
        "Parameters / variables",
        "Conservation relations",
        "Stiffness ratio (initial conditions)",
        "Stiffness ratio (steady state)",
        "Unused parameters",
        "Unit coverage",
        "Unit consistency",
    ]


def test_report_counts_are_correct() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["State variables"].value == "2"
    assert by_name["Parameters"].value == "3"
    assert by_name["Reactions"].value == "3"


def test_report_repr_html_renders_table() -> None:
    html = qa.report(linear_chain())._repr_html_()
    assert "<table>" in html
    assert "mxlpy-qa-report" in html


def test_report_write_produces_standalone_document(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    qa.report(linear_chain()).write(path)
    content = path.read_text()
    assert content.startswith("<!DOCTYPE html>")
    assert "<table>" in content


###############################################################################
# size grading
###############################################################################


def test_small_model_counts_are_graded_green() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["State variables"].grade == "green"
    assert by_name["Parameters"].grade == "green"
    assert by_name["Reactions"].grade == "green"


###############################################################################
# conservation relations / stiffness
###############################################################################


def test_conservation_relation_detected_in_closed_chain() -> None:
    result = qa.report(closed_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Conservation relations"].value == "1"
    assert by_name["Conservation relations"].grade == "info"


def test_no_conservation_relation_in_open_chain() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Conservation relations"].value == "0"


def test_stiffness_ratio_at_initial_conditions_is_reported() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    row = by_name["Stiffness ratio (initial conditions)"]
    assert row.grade in {"green", "orange", "red"}
    assert "ratio=" in row.value


def test_steady_state_stiffness_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Simulator, "get_result", lambda _self: Result(IntegrationFailure())
    )
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    row = by_name["Stiffness ratio (steady state)"]
    assert row.grade == "red"
    assert row.value == "did not converge"


###############################################################################
# unused parameters
###############################################################################


def test_unused_parameter_is_flagged_red() -> None:
    model = linear_chain().add_parameter("unused", 1.0)
    result = qa.report(model)
    by_name = {m.name: m for m in result.metrics}
    row = by_name["Unused parameters"]
    assert row.grade == "red"
    assert "unused" in row.value


def test_no_unused_parameters_is_green() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Unused parameters"].grade == "green"


###############################################################################
# units
###############################################################################


def test_unit_coverage_zero_when_no_units_set() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    row = by_name["Unit coverage"]
    assert row.grade == "red"
    assert "0/" in row.value


def test_unit_coverage_full_when_all_units_set() -> None:
    result = qa.report(unitful_chain())
    by_name = {m.name: m for m in result.metrics}
    row = by_name["Unit coverage"]
    assert row.grade == "green"


def test_unit_consistency_not_checked_without_time_unit() -> None:
    result = qa.report(linear_chain())
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Unit consistency"].value == "not checked"
    assert by_name["Unit consistency"].grade == "info"


def test_unit_consistency_green_when_consistent() -> None:
    result = qa.report(unitful_chain(), time_unit=second)
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Unit consistency"].grade == "green"


def test_unit_consistency_red_when_mismatched() -> None:
    model = unitful_chain().update_parameter("k1", unit=1 / (second * second))
    result = qa.report(model, time_unit=second)
    by_name = {m.name: m for m in result.metrics}
    assert by_name["Unit consistency"].grade == "red"
