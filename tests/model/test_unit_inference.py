# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Tests for Model.infer_units() - bidirectional unit constraint propagation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import sympy

if TYPE_CHECKING:
    import pytest

from mxlpy import Conflict, Derived, KineticModelBuilder, fns
from mxlpy._kinetic_builder import MdText
from mxlpy.surrogates import qss
from mxlpy.unit_inference import unit_of_expr
from mxlpy.units import liter, mmol, second, volt

mmol_per_second = mmol / second
per_second = 1 / second  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mass_action_model() -> KineticModelBuilder:
    """S → P via v = k * S.  S and k are annotated; P and reaction are not."""
    return (
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=per_second)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0)
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"S": -1, "P": 1},
            args=["S", "k"],
        )
    )


def _mm_model() -> KineticModelBuilder:
    """S → P via Michaelis-Menten.  vmax and S annotated; km not."""
    return (
        KineticModelBuilder()
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
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=per_second)
        .add_variable("S", 1.0, unit=mmol)
        .add_derived("flux", fns.mul, args=["k", "S"])
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["flux"] == mmol_per_second


def test_forward_readout_unit() -> None:
    model = (
        KineticModelBuilder()
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
        KineticModelBuilder()
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
    assert k_unit == 1 / second  # type: ignore


def test_backward_parameter_additive() -> None:
    """km is inferred from the additive term of Michaelis-Menten (S + km)."""
    model = _mm_model()
    result = model.infer_units(time_unit=second)
    assert result.parameters["km"] == mmol


def test_backward_difference_no_false_conflict() -> None:
    """v = vmax * S / (S + km * (1 + I / ki)) with all units set is consistent."""

    def rate(s: float, i: float, vmax: float, km: float, ki: float) -> float:
        return vmax * s / (s + km * (1 + i / ki))

    model = (
        KineticModelBuilder()
        .add_parameter("vmax", 10.0, unit=mmol_per_second)
        .add_parameter("km", 0.5, unit=mmol)
        .add_parameter("ki", 0.5, unit=mmol)
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("I", 1.0, unit=mmol)
        .add_reaction(
            "v1", rate, stoichiometry={"S": -1}, args=["S", "I", "vmax", "km", "ki"]
        )
    )
    assert not model.infer_units(time_unit=second).has_conflicts()


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
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=per_second)
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
    zero-order flux in v2 (unit mmol/s) - an inconsistent model.
    """
    model = (
        KineticModelBuilder()
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
        KineticModelBuilder()
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
        KineticModelBuilder()
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
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=per_second)
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
    assert result.parameters["k"] == per_second
    assert result.variables["S"] == mmol
    assert result.reactions["v1"] == rxn_unit


def test_unparseable_function_returns_none() -> None:
    """A function not parseable by fn_to_sympy yields None for that component."""

    def _unparseable(_x: float) -> float:
        # eval() is not parseable by fn_to_sympy's AST walker
        return eval("_x * 2")  # noqa: S307

    model = (
        KineticModelBuilder()
        .add_variable("x", 1.0, unit=mmol)
        .add_derived("d", _unparseable, args=["x"])
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["d"] is None


# ── all_inferred ──────────────────────────────────────────────────────────────


def test_all_inferred_true_when_all_known() -> None:
    # k, S are explicit; v1 and P are inferred - all four should be resolved
    model = _mass_action_model()
    result = model.infer_units(time_unit=second)
    assert result.all_inferred()


def test_all_inferred_false_when_none() -> None:
    model = KineticModelBuilder().add_parameter("k", 1.0).add_variable("S", 1.0)
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
        KineticModelBuilder()
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
        KineticModelBuilder()
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
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=mmol / second)
        .add_variable("S", 1.0)
        .add_derived("slope", linear_in_time, args=["time", "k"])
    )
    result = model.infer_units(time_unit=second)
    slope_unit = result.derived["slope"]
    assert isinstance(slope_unit, sympy.Expr)
    # unit(time * k) = second * (mmol/second) = mmol
    assert slope_unit == mmol


# ── Sums ─────────────────────────────────────────────────────────────────────


def test_forward_difference_keeps_unit() -> None:
    model = (
        KineticModelBuilder()
        .add_parameter("total", 1.0, unit=mmol)
        .add_variable("S", 0.5, unit=mmol)
        .add_derived("free", fns.moiety_1s, args=["S", "total"])
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["free"] == mmol


def test_unit_of_expr_mismatched_sum() -> None:
    a, b = sympy.symbols("a b")
    unit = unit_of_expr(a + b, {a: mmol, b: sympy.S.One})
    assert unit is not None
    assert unit not in (mmol, sympy.S.One)


def test_unit_of_expr_missing_unit() -> None:
    a, b = sympy.symbols("a b")
    assert unit_of_expr(a * b, {a: mmol}) is None


# ── Stoichiometries and surrogates ───────────────────────────────────────────


def test_variable_unit_from_derived_stoichiometry() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_variable("B", 0.0)
        .add_parameter("k", 1.0, unit=per_second)
        .add_parameter("scale", 2.0, unit=1 / liter)  # type: ignore[arg-type]
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"A": -1, "B": Derived(fn=fns.constant, args=["scale"])},
            args=["A", "k"],
        )
    )
    result = model.infer_units(time_unit=second)
    assert result.variables["B"] == mmol / liter  # type: ignore[operator]
    assert not result.has_conflicts()


def test_surrogate_output_units_propagate() -> None:
    def rate(a: float, k: float) -> tuple[float]:
        return (k * a,)

    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_parameter("k", 1.0)
        .add_derived("twice", fns.twice, args=["r1"])
        .add_surrogate(
            "s",
            qss.Surrogate(
                model=rate,
                args=["A", "k"],
                outputs=["r1"],
                units={"r1": mmol_per_second},
            ),
        )
    )
    result = model.infer_units(time_unit=second)
    assert result.derived["twice"] == mmol_per_second


def test_no_backward_constraint_through_functions() -> None:
    """pH-like log conversions hide unit constants; they must not cause conflicts."""

    def ph(h: float) -> float:
        return -np.log10(h * 2.5e-4)

    model = (
        KineticModelBuilder()
        .add_variable("H", 1.0, unit=mmol)
        .add_derived("pH", ph, args=["H"])
    )
    assert not model.infer_units(time_unit=second).has_conflicts()


def test_unit_of_expr_float_exponent() -> None:
    a = sympy.Symbol("a")
    assert unit_of_expr(a**2.0, {a: mmol}) == mmol**2


def test_inconsistent_sum_is_not_propagated() -> None:
    """A reaction mixing units must not push its broken unit onto other components."""

    def rate(s: float, k: float) -> float:
        return k * s + k

    model = (
        KineticModelBuilder()
        .add_variable("S", 1.0, unit=mmol)
        .add_parameter("k", 1.0, unit=per_second)
        .add_derived("d", fns.twice, args=["v1"])
        .add_reaction("v1", rate, stoichiometry={"S": -1}, args=["S", "k"])
    )
    result = model.infer_units(time_unit=second)
    assert result.reactions["v1"] == mmol_per_second
    assert result.derived["d"] == mmol_per_second


def test_explicit_derived_unit_does_not_constrain_args() -> None:
    """pmf = dpsi + 0.06 * dpH hides 0.06 V/pH; the explicit unit overrides it."""

    def pmf(dpsi: float, ph_st: float, ph_lu: float) -> float:
        return dpsi + 0.06 * (ph_st - ph_lu)

    model = (
        KineticModelBuilder()
        .add_parameter("dpsi", 0.1, unit=volt)
        .add_parameter("pH_st", 7.8, unit=sympy.S.One)
        .add_parameter("pH_lu", 6.0, unit=sympy.S.One)
        .add_derived("pmf", pmf, args=["dpsi", "pH_st", "pH_lu"], unit=volt)
    )
    assert not model.infer_units(time_unit=second).has_conflicts()


def test_zero_rate_does_not_infer_units() -> None:
    def zero() -> float:
        return 0

    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_reaction("v1", zero, stoichiometry={"A": 1}, args=[])
    )
    result = model.infer_units(time_unit=second)
    assert not result.has_conflicts()


def test_unit_of_expr_numeric_guard_in_max() -> None:
    a = sympy.Symbol("a")
    assert unit_of_expr(sympy.Max(1e-30, a), {a: mmol}) == mmol
    assert unit_of_expr(sympy.Min(1.0, a), {a: mmol}) == mmol
    assert unit_of_expr(a + 1, {a: mmol}) not in (mmol, sympy.S.One)
