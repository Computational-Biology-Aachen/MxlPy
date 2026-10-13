# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Unit definitions for MxlPy."""

import sympy
from sympy.physics.units import (
    ampere,
    atto,
    bar,
    becquerel,
    candela,
    centi,
    coulomb,
    deca,
    deci,
    exa,
    farad,
    femto,
    giga,
    gram,
    gray,
    hecto,
    henry,
    hertz,
    hour,
    joule,
    katal,
    kelvin,
    kilo,
    kilogram,
    liter,
    lux,
    mega,
    meter,
    micro,
    milli,
    minute,
    mol,
    nano,
    newton,
    ohm,
    pascal,
    peta,
    pico,
    radian,
    second,
    siemens,
    steradian,
    tera,
    tesla,
    volt,
    watt,
    weber,
    yocto,
    yotta,
    zepto,
    zetta,
)
from sympy.physics.units.quantities import Quantity

__all__ = [
    "ampere",
    "atto",
    "avogadro",
    "bar",
    "becquerel",
    "candela",
    "cbm",
    "celsius",
    "centi",
    "coulomb",
    "deca",
    "deci",
    "dimensionless",
    "einstein",
    "exa",
    "farad",
    "femto",
    "giga",
    "gram",
    "gray",
    "hecto",
    "henry",
    "hertz",
    "hour",
    "item",
    "joule",
    "katal",
    "kelvin",
    "kilo",
    "kilogram",
    "liter",
    "lumen",
    "lux",
    "mega",
    "meter",
    "micro",
    "milli",
    "minute",
    "mmol",
    "mmol_g",
    "mmol_h",
    "mmol_m",
    "mmol_mol_chl",
    "mmol_s",
    "mol",
    "mol_chl",
    "mol_g",
    "mol_h",
    "mol_m",
    "mol_s",
    "mumol",
    "mumol_g",
    "mumol_h",
    "mumol_m",
    "mumol_s",
    "nano",
    "newton",
    "nmol",
    "nmol_g",
    "nmol_h",
    "nmol_m",
    "nmol_s",
    "ohm",
    "pascal",
    "per_hour",
    "per_minute",
    "per_second",
    "peta",
    "pico",
    "pmol",
    "pmol_g",
    "pmol_h",
    "pmol_m",
    "pmol_s",
    "ppfd",
    "radian",
    "second",
    "siemens",
    "sievert",
    "sqm",
    "steradian",
    "tera",
    "tesla",
    "volt",
    "watt",
    "weber",
    "yocto",
    "yotta",
    "zepto",
    "zetta",
]

# time unit
per_second = 1 / second  # type: ignore
per_minute = 1 / minute  # type: ignore
per_hour = 1 / hour  # type: ignore


sqm = meter**2
cbm = meter**3

mol_s = mol / second  # type: ignore
mol_m = mol / minute  # type: ignore
mol_h = mol / hour  # type: ignore
mol_g = mol / gram  # type: ignore

mmol = mol * milli
mmol_s = mmol / second
mmol_m = mmol / minute
mmol_h = mmol / hour
mmol_g = mmol / gram

mumol = mol * micro
mumol_s = mumol / second
mumol_m = mumol / minute
mumol_h = mumol / hour
mumol_g = mumol / gram

nmol = mol * nano
nmol_s = nmol / second
nmol_m = nmol / minute
nmol_h = nmol / hour
nmol_g = nmol / gram

pmol = mol * pico
pmol_s = pmol / second
pmol_m = pmol / minute
pmol_h = pmol / hour
pmol_g = pmol / gram

ppfd = mumol / sqm / second


# SBML units
avogadro = 6.02214076e23
sievert = joule / kilogram  # type: ignore
lumen = candela * steradian  # type: ignore
dimensionless = sympy.S.One
item = 1  # pseudounit for one thing

# Plant units
mol_chl = Quantity("mol_chl", abbrev="mol_chl")
mmol_mol_chl = mmol / mol_chl

# Domain units without a sympy.physics.units equivalent
celsius = Quantity("celsius", abbrev="°C")
einstein = Quantity("einstein", abbrev="E")  # one mole of photons
