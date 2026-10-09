"""Unit definitions for MxlPy.

Besides the convenience aliases (``mmol_s``, ``ppfd``, ...), this module holds
the shared unit vocabulary of the mxl* tool family (:data:`REGISTRY` and
:data:`PREFIXES`, mirroring ``mxl-schemas/v1/units.json``); :func:`unit_to_json` and
:func:`unit_from_json` convert between sympy unit expressions and the
``.mxl.json`` unit representation (a flat list of ``{kind, prefix?, exponent}``
factors, the SBML unitDefinition model) that mxlweb uses as well.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import sympy
import sympy.physics.units as su
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
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.util import convert_to

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "PREFIXES",
    "REGISTRY",
    "Quantity",
    "ampere",
    "avogadro",
    "becquerel",
    "candela",
    "cbm",
    "celsius",
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
    "kilogram",
    "liter",
    "lumen",
    "lux",
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
    "tesla",
    "unit_from_json",
    "unit_to_json",
    "volt",
    "watt",
    "weber",
]

###############################################################################
# Shared registry — mirrors mxl-schemas/v1/units.json; kept in sync by
# tests/test_units.py. Order matters: it is the canonical factor order.
###############################################################################

#: Registry kind id -> sympy unit expression.
REGISTRY: dict[str, sympy.Expr] = {
    "ampere": su.ampere,
    "becquerel": su.becquerel,
    "candela": su.candela,
    "coulomb": su.coulomb,
    "farad": su.farad,
    "gram": su.gram,
    "gray": su.gray,
    "henry": su.henry,
    "hertz": su.hertz,
    "item": Quantity("item", abbrev="item"),
    "joule": su.joule,
    "katal": su.katal,
    "kelvin": su.kelvin,
    "litre": su.liter,
    "lumen": su.candela * su.steradian,
    "lux": su.lux,
    "metre": su.meter,
    "mole": su.mol,
    "newton": su.newton,
    "ohm": su.ohm,
    "pascal": su.pascal,
    "radian": su.radian,
    "second": su.second,
    "siemens": su.siemens,
    "sievert": su.joule / su.kilogram,
    "steradian": su.steradian,
    "tesla": su.tesla,
    "volt": su.volt,
    "watt": su.watt,
    "weber": su.weber,
    "minute": su.minute,
    "hour": su.hour,
    "celsius": Quantity("celsius", abbrev="°C"),
    "mol_chl": Quantity("mol_chl", abbrev="mol_chl"),
}

#: SI prefix id -> power-of-ten scale.
PREFIXES: dict[str, int] = {
    "yotta": 24,
    "zetta": 21,
    "exa": 18,
    "peta": 15,
    "tera": 12,
    "giga": 9,
    "mega": 6,
    "kilo": 3,
    "hecto": 2,
    "deca": 1,
    "deci": -1,
    "centi": -2,
    "milli": -3,
    "micro": -6,
    "nano": -9,
    "pico": -12,
    "femto": -15,
    "atto": -18,
    "zepto": -21,
    "yocto": -24,
}

# Registry kinds with no `sympy.physics.units` equivalent.
_DOMAIN_KINDS = frozenset({"item", "celsius", "mol_chl"})

_KIND_ORDER = {kind: i for i, kind in enumerate(REGISTRY)}
_PREFIX_BY_SCALE = {scale: name for name, scale in PREFIXES.items()}
# Atomic quantities that map straight back onto a registry kind. Composite
# entries (lumen = candela*steradian) decompose into their constituents.
_KIND_BY_QUANTITY: dict[Quantity, str] = {
    expr: kind for kind, expr in REGISTRY.items() if isinstance(expr, Quantity)
}


def _custom_quantity(kind: str) -> Quantity:
    return Quantity(kind, abbrev=kind)


def _quantity_to_kind(
    q: Quantity, customs: dict[str, dict[str, str]]
) -> tuple[str, sympy.Expr]:
    """Map a sympy quantity to a (kind, scale) pair, registering unknown ones as custom kinds."""
    if (kind := _KIND_BY_QUANTITY.get(q)) is not None:
        return kind, sympy.Integer(1)
    if q == kilogram:
        return "gram", sympy.Integer(1000)
    # Other built-in sympy units (day, kilometer, milliliter, ...): express them
    # as a number times a single registry quantity.
    for target, kind in _KIND_BY_QUANTITY.items():
        if kind in _DOMAIN_KINDS:
            continue
        converted = sympy.sympify(convert_to(q, target))
        coeff, rest = converted.as_coeff_Mul()
        if rest == target and coeff.is_number:
            return kind, sympy.nsimplify(coeff)
    name = str(q.name)
    if name in REGISTRY:
        msg = f"Quantity {q!r} shadows the registry unit {name!r}"
        raise ValueError(msg)
    entry: dict[str, str] = {}
    if str(q.abbrev) != name:
        entry["symbol"] = str(q.abbrev)
    customs.setdefault(name, entry)
    return name, sympy.Integer(1)


def _collect(
    expr: sympy.Expr,
    exp: int,
    kinds: dict[str, int],
    scale: list[sympy.Expr],
    customs: dict[str, dict[str, str]],
) -> None:
    if isinstance(expr, Prefix):
        scale[0] *= sympy.Rational(expr.scale_factor) ** exp
    elif isinstance(expr, Quantity):
        kind, factor = _quantity_to_kind(expr, customs)
        kinds[kind] = kinds.get(kind, 0) + exp
        scale[0] *= factor**exp
    elif expr.is_number:
        scale[0] *= sympy.nsimplify(expr) ** exp
    elif isinstance(expr, sympy.Pow):
        base, power = expr.as_base_exp()
        if not power.is_integer:
            msg = f"Unit exponents must be integers, got {expr}"
            raise ValueError(msg)
        _collect(base, exp * int(power), kinds, scale, customs)
    elif isinstance(expr, sympy.Mul):
        for arg in expr.args:
            _collect(arg, exp, kinds, scale, customs)
    else:
        msg = f"Not a unit expression: {expr!r}"
        raise ValueError(msg)


def _sort_key(kind: str, exponent: int) -> tuple[bool, int, str]:
    return exponent < 0, _KIND_ORDER.get(kind, len(_KIND_ORDER)), kind


def _power_of_ten(value: sympy.Expr) -> int | None:
    if value <= 0:
        return None
    n = round(math.log10(float(value)))
    return n if sympy.Rational(10) ** n == value else None


def unit_to_json(
    unit: sympy.Expr | None,
    customs: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Convert a sympy unit expression into its ``.mxl.json`` representation.

    Prefixes in sympy are plain scale factors (``micro*mol/meter**2`` already
    collapses ``micro`` into ``1/1000000``), so the overall power-of-ten scale is
    re-attached as a prefix to the first factor (positive exponents first, then
    registry order) that can carry it exactly; anything else ends up in
    ``multiplier``.

    Parameters
    ----------
    unit
        Unit expression, or ``None`` for dimensionless.
    customs
        Collects quantities that are not in the shared registry (e.g.
        ``Quantity("OD600")``) as custom unit declarations for ``model.units``.

    Returns
    -------
    dict
        ``{"factors": [...], "multiplier"?: number}``

    Raises
    ------
    ValueError
        If the expression is not a product of integer powers of quantities.

    """
    if customs is None:
        customs = {}
    if unit is None:
        return {"factors": []}
    kinds: dict[str, int] = {}
    scale: list[sympy.Expr] = [sympy.Integer(1)]
    _collect(sympy.sympify(unit), 1, kinds, scale, customs)

    factors: list[dict[str, Any]] = [
        {"kind": kind, "exponent": exp}
        for kind, exp in sorted(kinds.items(), key=lambda kv: _sort_key(*kv))
        if exp != 0
    ]
    total = scale[0]
    if total == 1:
        return {"factors": factors}
    if (n := _power_of_ten(total)) is not None:
        for factor in factors:
            exponent = factor["exponent"]
            if n % exponent == 0 and (p := _PREFIX_BY_SCALE.get(n // exponent)):
                factor["prefix"] = p
                # keep the key order of the schema: kind, prefix, exponent
                factor["exponent"] = factor.pop("exponent")
                return {"factors": factors}
    return {"factors": factors, "multiplier": float(total)}


def unit_from_json(
    data: Mapping[str, Any],
    customs: Mapping[str, Any] | None = None,
) -> sympy.Expr | None:
    """Convert a ``.mxl.json`` unit back into a sympy expression.

    Parameters
    ----------
    data
        ``{"factors": [...], "multiplier"?: number}``
    customs
        The model's ``units`` section declaring custom kinds.

    Returns
    -------
    sympy.Expr | None
        The unit, or ``None`` if dimensionless.

    Raises
    ------
    ValueError
        If a factor uses a kind that is neither in the registry nor declared
        in ``customs``, or an unknown prefix.

    """
    customs = customs or {}
    expr: sympy.Expr = sympy.Integer(1)
    for factor in data["factors"]:
        kind = factor["kind"]
        if kind in REGISTRY:
            base = REGISTRY[kind]
        elif kind in customs:
            base = _custom_quantity(kind)
        else:
            msg = f"Unknown unit kind {kind!r} (not in the registry and not declared in model.units)"
            raise ValueError(msg)
        if (prefix := factor.get("prefix")) is not None:
            if prefix not in PREFIXES:
                msg = f"Unknown unit prefix {prefix!r}"
                raise ValueError(msg)
            # Quantity first: `prefix * quantity` collapses the prefix into a
            # plain rational, `quantity * prefix` keeps it (cf. `mmol` below).
            base = base * getattr(su, prefix)
        expr *= base ** factor["exponent"]
    multiplier = data.get("multiplier", 1)
    if multiplier != 1:
        expr *= sympy.nsimplify(multiplier)
    if expr == 1:
        return None
    return expr


###############################################################################
# Convenience aliases
###############################################################################

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
sievert = REGISTRY["sievert"]
lumen = REGISTRY["lumen"]
dimensionless = None
item = REGISTRY["item"]  # pseudounit for one thing
celsius = REGISTRY["celsius"]

# Plant units
mol_chl = REGISTRY["mol_chl"]
mmol_mol_chl = mmol / mol_chl
