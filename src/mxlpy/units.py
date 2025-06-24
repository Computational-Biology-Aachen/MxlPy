"""Unit definitions for MxlPy."""

from sympy.physics.units import (
    ampere,
    becquerel,
    candela,
    coulomb,
    farad,
    gram,
    gray,
    henry,
    hertz,
    hour,
    joule,
    katal,
    kelvin,
    kilogram,
    liter,
    lux,
    meter,
    micro,
    milli,
    minute,
    mol,
    nano,
    newton,
    ohm,
    pascal,
    pico,
    radian,
    second,
    siemens,
    steradian,
    tesla,
    volt,
    watt,
    weber,
)
from sympy.physics.units.quantities import Quantity

__all__ = [
    "ampere",
    "becquerel",
    "coulomb",
    "dimensionless",
    "farad",
    "gram",
    "gray",
    "henry",
    "hertz",
    "hour",
    "item",
    "joule",
    "katal",
    "kelvin",
    "liter",
    "lumen",
    "lux",
    "micro",
    "milli",
    "minute",
    "mmol",
    "mmol_g",
    "mmol_s",
    "mol",
    "nano",
    "newton",
    "nmol",
    "ohm",
    "pascal",
    "ppfd",
    "radian",
    "second",
    "siemens",
    "sievert",
    "sqm",
    "tesla",
    "volt",
    "watt",
    "weber",
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
dimensionless = None
item = 1  # pseudounit for one thing

# Plant units
mol_chl = Quantity("mol_chl", abbrev="mol_chl")
mmol_mol_chl = mmol / mol_chl
