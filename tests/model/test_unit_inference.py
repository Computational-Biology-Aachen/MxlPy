"""Tests for Model.infer_units() — bidirectional unit constraint propagation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    import pytest

from mxlpy import Conflict, Model, fns
from mxlpy.model import MdText
from mxlpy.units import mmol, second

mmol_per_second = mmol / second


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mass_action_model() -> Model:
    """S → P via v = k * S.  S and k are annotated; P and reaction are not."""
    return (
        Model()
        .add_parameter("k", 1.0, unit=1 / second)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0)
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"S": -1, "P": 1},
            args=["S", "k"],
        )
    )


def _mm_model() -> Model:
    """S → P via Michaelis-Menten.  vmax and S annotated; km not."""
    return (
        Model()
        .add_parameter("vmax", 10.0, unit=mmol_per_second)
        .add_parameter("km", 0.5)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_reaction(
            "v1",
            fns.michaelis_menten_1s,
            stoichiometry={"S": -1, "P": 1},
            args=["S", "vmax", "km"],
        )
    )


# ── Forward inference ─────────────────────────────────────────────────────────


def test_forward_reaction_unit() -> None:
    model = _mass_action_model()
    result = model.infer_units(time_unit=second)
    rxn_unit = result.reactions["v1"]
    assert isinstance(rxn_unit, sympy.Expr)
    assert rxn_unit == mmol_per_second


def test_forward_variable_unit_from_reaction() -> None:
    # P has no explicit unit; inferred from v1 via ODE: unit(P) = unit(v1) * second
    model = _mass_action_model()
    result = model.infer_units(time_unit=second)
    assert result.variables["P"] == mmol


def test_forward_derived_unit() -> None:
    model = (
        Model()
        .add_parameter("k", 1.0, unit=1 / second)
        .add_variable("S", 1.0, unit=mmol)
        .add_derived("flux", fns.mul, args=["k", "S"])
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["flux"] == mmol_per_second


def test_forward_readout_unit() -> None:
    model = (
        Model()
        .add_parameter("k", 2.0, unit=mmol)
        .add_variable("S", 1.0, unit=mmol)
        .add_readout("ratio", fns.div, args=["k", "S"])
    )
    result = model.infer_units(time_unit=second)
    ratio_unit = result.readouts["ratio"]
    assert isinstance(ratio_unit, sympy.Expr)
    assert sympy.simplify(ratio_unit) == sympy.Integer(1)


# ── Backward inference ────────────────────────────────────────────────────────


def test_backward_parameter_unit_simple() -> None:
    """Backward inference works for purely multiplicative rate functions."""
    # v = k * S, unit(v)=mmol/s, unit(S)=mmol → unit(k) = 1/s
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1, "P": 1}, args=["S", "k"]
        )
    )
    result = model.infer_units(time_unit=second)
    k_unit = result.parameters["k"]
    assert isinstance(k_unit, sympy.Expr)
    assert k_unit == 1 / second


def test_backward_parameter_additive_returns_none() -> None:
    """Backward inference via additive terms (Michaelis-Menten km) returns None.

    sympy.solve produces a trivial algebraic solution (km=0) which is filtered
    out.  This is a known limitation — additive constraints require AST-level
    unit analysis beyond what sympy.solve provides.
    """
    model = _mm_model()
    result = model.infer_units(time_unit=second)
    km_unit = result.parameters["km"]
    assert km_unit is None


def test_backward_variable_unit_from_reaction_and_variable() -> None:
    # P unit inferred from ODE: unit(P) = unit(v1) * time_unit
    # v1 unit inferred forward as mmol/second
    model = _mm_model()
    result = model.infer_units(time_unit=second)
    assert result.variables["P"] == mmol


# ── Transitive derived chain ──────────────────────────────────────────────────


def test_transitive_derived_unit() -> None:
    """d2 depends on d1 which depends on k and S."""
    model = (
        Model()
        .add_parameter("k", 1.0, unit=1 / second)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0)
        .add_derived("d1", fns.mul, args=["k", "S"])  # mmol/s
        .add_derived("d2", fns.mul, args=["d1", "S"])  # mmol²/s
        .add_reaction(
            "v1",
            fns.constant,
            stoichiometry={"S": -1, "P": 1},
            args=["d1"],
        )
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["d1"] == mmol_per_second
    assert result.derived["d2"] == mmol**2 / second


# ── Conflict detection ────────────────────────────────────────────────────────


def test_conflict_detection() -> None:
    """Two reactions constrain the same parameter's unit differently.

    k is used as a first-order rate constant in v1 (unit 1/s) and as a
    zero-order flux in v2 (unit mmol/s) — an inconsistent model.
    """
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        # v1 = k * S  → backward gives unit(k) = 1/second
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1}, args=["S", "k"]
        )
        # v2 = k  (constant flux) → backward gives unit(k) = mmol/second
        .add_reaction("v2", fns.constant, stoichiometry={"P": 1}, args=["k"])
    )
    result = model.infer_units(time_unit=second)
    k_result = result.parameters["k"]
    assert isinstance(k_result, Conflict)
    assert len(k_result.constraints) >= 2


def test_has_conflicts() -> None:
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1}, args=["S", "k"]
        )
        .add_reaction("v2", fns.constant, stoichiometry={"P": 1}, args=["k"])
    )
    result = model.infer_units(time_unit=second)
    assert result.has_conflicts()


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_no_units_returns_none() -> None:
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0)
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1}, args=["S", "k"]
        )
    )
    result = model.infer_units(time_unit=second)
    assert result.parameters["k"] is None
    assert result.variables["S"] is None
    assert result.reactions["v1"] is None


def test_all_units_set_returns_existing() -> None:
    rxn_unit = mmol_per_second
    model = (
        Model()
        .add_parameter("k", 1.0, unit=1 / second)
        .add_variable("S", 1.0, unit=mmol)
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"S": -1},
            args=["S", "k"],
            unit=rxn_unit,
        )
    )
    result = model.infer_units(time_unit=second)
    assert result.parameters["k"] == 1 / second
    assert result.variables["S"] == mmol
    assert result.reactions["v1"] == rxn_unit


def test_unparseable_function_returns_none() -> None:
    """A function not parseable by fn_to_sympy yields None for that component."""

    def _unparseable(_x: float) -> float:
        # eval() is not parseable by fn_to_sympy's AST walker
        return eval("_x * 2")  # noqa: S307

    model = (
        Model()
        .add_variable("x", 1.0, unit=mmol)
        .add_derived("d", _unparseable, args=["x"])
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["d"] is None


# ── all_inferred ──────────────────────────────────────────────────────────────


def test_all_inferred_true_when_all_known() -> None:
    # k, S are explicit; v1 and P are inferred — all four should be resolved
    model = _mass_action_model()
    result = model.infer_units(time_unit=second)
    assert result.all_inferred()


def test_all_inferred_false_when_none() -> None:
    model = Model().add_parameter("k", 1.0).add_variable("S", 1.0)
    result = model.infer_units(time_unit=second)
    assert not result.all_inferred()


# ── apply_to ──────────────────────────────────────────────────────────────────


def test_apply_to_fills_nones() -> None:
    model = _mass_action_model()
    assert model.get_raw_variables()["P"].unit is None
    result = model.infer_units(time_unit=second)
    result.apply_to(model)
    assert model.get_raw_variables()["P"].unit == mmol


def test_apply_to_does_not_overwrite_explicit_units() -> None:
    explicit_unit = mmol**2
    model = _mass_action_model()
    # Manually set a (different) explicit unit for P
    model._variables["P"].unit = explicit_unit
    result = model.infer_units(time_unit=second)
    result.apply_to(model)
    # apply_to must NOT overwrite
    assert model.get_raw_variables()["P"].unit == explicit_unit


def test_apply_to_skips_conflicts_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1}, args=["S", "k"]
        )
        .add_reaction("v2", fns.constant, stoichiometry={"P": 1}, args=["k"])
    )
    result = model.infer_units(time_unit=second)
    with caplog.at_level(logging.WARNING, logger="mxlpy.model"):
        result.apply_to(model)
    assert model.get_raw_parameters()["k"].unit is None
    assert any("conflict" in msg.lower() for msg in caplog.messages)


# ── report ────────────────────────────────────────────────────────────────────


def test_report_renders() -> None:
    model = _mass_action_model()
    result = model.infer_units(time_unit=second)
    rpt = result.report()
    assert isinstance(rpt, MdText)
    md = rpt._repr_markdown_()
    assert "## Unit Inference" in md
    assert len(md) > 0


def test_report_contains_conflict_label() -> None:
    model = (
        Model()
        .add_parameter("k", 1.0)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"S": -1}, args=["S", "k"]
        )
        .add_reaction("v2", fns.constant, stoichiometry={"P": 1}, args=["k"])
    )
    result = model.infer_units(time_unit=second)
    md = result.report()._repr_markdown_()
    assert "CONFLICT" in md


# ── time arg ─────────────────────────────────────────────────────────────────


def test_time_arg_uses_time_unit() -> None:
    """A derived quantity that takes 'time' as an arg should use time_unit."""

    def linear_in_time(t: float, k: float) -> float:
        return t * k

    model = (
        Model()
        .add_parameter("k", 1.0, unit=mmol / second)
        .add_variable("S", 1.0)
        .add_derived("slope", linear_in_time, args=["time", "k"])
    )
    result = model.infer_units(time_unit=second)
    slope_unit = result.derived["slope"]
    assert isinstance(slope_unit, sympy.Expr)
    # unit(time * k) = second * (mmol/second) = mmol
    assert slope_unit == mmol
