# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""
FIXME: needs testing
"""

from sympy.physics.units import convert_to
from sympy.physics.units.prefixes import Prefix

from mxlpy import Model, Parameter, Variable, units
from mxlpy.units import bar, mega, mmol, pascal, second

SI_PREFIXES = [
    "yotta",
    "zetta",
    "exa",
    "peta",
    "tera",
    "giga",
    "mega",
    "kilo",
    "hecto",
    "deca",
    "deci",
    "centi",
    "milli",
    "micro",
    "nano",
    "pico",
    "femto",
    "atto",
    "zepto",
    "yocto",
]


def _mul(a: float, b: float) -> float:
    return a * b


def test_mega_pascal_cancels() -> None:
    mpa = pascal * mega
    model = (
        Model()
        .add_variables({"x": Variable(1.0, unit=mmol)})
        .add_parameters(
            {
                "psi": Parameter(1.0, unit=mpa),
                "g": Parameter(1.0, unit=mmol / second / mpa),
            }
        )
        .add_reaction(
            "v",
            fn=_mul,
            args=["psi", "g"],
            stoichiometry={"x": 1},
            unit=mmol / second,
        )
    )
    assert model.check_units(time_unit=second).correct_diff_eqs() == {"x": True}


def test_all_si_prefixes_exported() -> None:
    for name in SI_PREFIXES:
        assert name in units.__all__
        assert isinstance(getattr(units, name), Prefix)


def test_bar() -> None:
    assert convert_to(bar, pascal) == 100000 * pascal


def test_domain_units_exported() -> None:
    for name in ("celsius", "einstein", "mol_chl"):
        assert name in units.__all__
