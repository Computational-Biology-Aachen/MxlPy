# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Tests for Model.check_units()."""

from __future__ import annotations

from typing import cast

import pytest

from mxlpy import Derived, KineticModelBuilder, fns
from mxlpy._kinetic_builder import Failure
from mxlpy.surrogates import qss
from mxlpy.units import dimensionless, liter, mmol, second


def _reversible(keq_unit: object) -> KineticModelBuilder:
    """S <-> P via v = vmax * (S - P / keq), with vmax = kcat * E0 derived."""

    def rate(s: float, p: float, vmax: float, keq: float) -> float:
        return vmax * (s - p / keq)

    return (
        KineticModelBuilder()
        .add_variable("S", 1.0, unit=mmol)
        .add_variable("P", 0.0, unit=mmol)
        .add_parameter("kcat", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_parameter("E0", 1.0, unit=dimensionless)
        .add_parameter("keq", 2.0, unit=keq_unit)  # type: ignore[arg-type]
        .add_derived("vmax", fns.mul, args=["kcat", "E0"])
        .add_reaction(
            "v1",
            rate,
            stoichiometry={"S": -1, "P": 1},
            args=["S", "P", "vmax", "keq"],
        )
    )


def test_check_units_resolves_derived_and_differences() -> None:
    check = _reversible(dimensionless).check_units(time_unit=second)
    assert check.per_variable == {"S": {"v1": True}, "P": {"v1": True}}


def test_check_units_failure_reports_expected_and_obtained() -> None:
    check = _reversible(dimensionless).check_units(time_unit=second / 60)  # type: ignore[arg-type]
    failure = check.per_variable["S"]["v1"]
    assert isinstance(failure, Failure)
    assert failure.expected == mmol / (second / 60)  # type: ignore[operator]
    assert failure.obtained == mmol / second  # type: ignore[operator]


def test_check_units_mismatched_sum_fails() -> None:
    check = _reversible(mmol).check_units(time_unit=second)
    assert isinstance(check.per_variable["S"]["v1"], Failure)


def test_check_units_uses_explicit_derived_unit() -> None:
    model = _reversible(dimensionless).update_derived("vmax", unit=mmol)
    check = model.check_units(time_unit=second)
    assert isinstance(check.per_variable["S"]["v1"], Failure)


def test_check_units_missing_unit_is_none() -> None:
    check = _reversible(None).check_units(time_unit=second)
    assert check.per_variable["S"]["v1"] is None


def test_check_units_unknown_arg_raises() -> None:
    model = _reversible(dimensionless).add_readout("r", fns.constant, args=["S"])
    model.add_reaction("v2", fns.constant, stoichiometry={"S": -1}, args=["r"])
    with pytest.raises(NotImplementedError):
        model.check_units(time_unit=second)


def _scaled_stoichiometry(scale_unit: object) -> KineticModelBuilder:
    """A -> B where B gains `scale` per unit of flux (e.g. a unit conversion)."""
    return (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_variable("B", 0.0, unit=mmol / liter)  # type: ignore[operator]
        .add_parameter("k", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_parameter("scale", 2.0, unit=scale_unit)  # type: ignore[arg-type]
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"A": -1, "B": Derived(fn=fns.constant, args=["scale"])},
            args=["A", "k"],
        )
    )


def test_check_units_derived_stoichiometry() -> None:
    check = _scaled_stoichiometry(1 / liter).check_units(time_unit=second)
    assert check.per_variable == {"A": {"v1": True}, "B": {"v1": True}}


def test_check_units_derived_stoichiometry_mismatch() -> None:
    check = _scaled_stoichiometry(dimensionless).check_units(time_unit=second)
    assert check.per_variable["A"]["v1"] is True
    assert isinstance(check.per_variable["B"]["v1"], Failure)


def _surrogate_model(units: dict) -> KineticModelBuilder:
    def two_rates(a: float, k: float) -> tuple[float, float]:
        return k * a, 2 * k * a

    return (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_parameter("k", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_reaction("v1", fns.constant, stoichiometry={"A": -1}, args=["r1"])
        .add_surrogate(
            "s",
            qss.Surrogate(
                model=two_rates,
                args=["A", "k"],
                outputs=["r1", "r2"],
                stoichiometries={"r2": {"A": -1}},
                units=units,
            ),
        )
    )


def test_check_units_surrogate_output_units() -> None:
    model = _surrogate_model({"r1": mmol / second, "r2": mmol / second})
    check = model.check_units(time_unit=second)
    assert check.per_variable == {"A": {"v1": True, "r2": True}}


def test_check_units_surrogate_without_units_is_none() -> None:
    check = _surrogate_model({}).check_units(time_unit=second)
    assert check.per_variable == {"A": {"v1": None, "r2": None}}


def test_check_units_reaction_as_argument() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_variable("B", 1.0, unit=mmol)
        .add_parameter("k", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_reaction(
            "v1", fns.mass_action_1s, stoichiometry={"A": -1}, args=["A", "k"]
        )
        .add_reaction("v2", fns.twice, stoichiometry={"B": 1}, args=["v1"])
    )
    check = model.check_units(time_unit=second)
    assert check.per_variable == {"A": {"v1": True}, "B": {"v2": True}}


def test_check_units_zero_branch_is_unitless() -> None:
    def gated(s: float, k: float, light: float) -> float:
        if light > 0:
            return k * s
        return 0

    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_parameter("k", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_parameter("light", 1.0, unit=dimensionless)
        .add_reaction("v1", gated, stoichiometry={"A": -1}, args=["A", "k", "light"])
    )
    assert model.check_units(time_unit=second).per_variable == {"A": {"v1": True}}


def _cast_rate(s: float, k: float) -> float:
    return cast(float, k * s)


def test_check_units_unparseable_rate_is_none() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("A", 1.0, unit=mmol)
        .add_parameter("k", 1.0, unit=1 / second)  # type: ignore[arg-type]
        .add_reaction("v1", _cast_rate, stoichiometry={"A": -1}, args=["A", "k"])
    )
    assert model.check_units(time_unit=second).per_variable == {"A": {"v1": None}}
    assert model.infer_units(time_unit=second).reactions["v1"] is not None
