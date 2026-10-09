"""Tests for the shared unit vocabulary and the .mxl.json unit representation."""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import sympy
import sympy.physics.units as su
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity

from mxlpy import KineticModelBuilder, fns, units
from mxlpy.serialize import model_from_dict, model_to_dict
from mxlpy.types import SerializationError
from mxlpy.units import PREFIXES, REGISTRY, unit_from_json, unit_to_json

ASSETS = Path(__file__).parent / "assets"
FIXTURES = json.loads((ASSETS / "units.fixtures.json").read_text(encoding="utf-8"))
UPSTREAM = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/mxl-schemas/main"
)
# Sibling checkout inside the 0-admin-meta repo, when present.
META_SCHEMAS = Path(__file__).resolve().parents[3] / "misc" / "mxl-schemas"


def _strip_prefixes(expr: sympy.Expr) -> sympy.Expr:
    return expr.subs({p: p.scale_factor for p in expr.atoms(Prefix)})  # type: ignore[arg-type]


def _same_unit(a: sympy.Expr | None, b: sympy.Expr | None) -> bool:
    """Equal including scale, comparing prefixes by their numeric value."""
    if a is None or b is None:
        return a is b
    return sympy.simplify(_strip_prefixes(a) / _strip_prefixes(b)) == 1


def _fixture_sympy(case: dict[str, Any]) -> sympy.Expr | None:
    if case["sympy"] is None:
        return None
    return eval(case["sympy"], {"su": su, "Quantity": Quantity})  # noqa: S307


def _canonical(rel: str) -> str:
    local = META_SCHEMAS / rel
    if local.exists():
        return local.read_text(encoding="utf-8")
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/{rel}", timeout=10) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except OSError as exc:
        pytest.skip(f"canonical {rel} unavailable: {exc}")


###############################################################################
# Drift against mxl-schemas
###############################################################################


def test_registry_matches_mxl_schemas() -> None:
    canonical = json.loads(_canonical("v1/units.json"))
    kinds = canonical["kinds"]
    # Same ids in the same (canonical factor) order.
    assert list(REGISTRY) == [k["id"] for k in kinds]
    for k in kinds:
        expected = (
            Quantity(k["id"], abbrev=k["abbrev"])
            if k["sympy"] is None
            else eval(k["sympy"], {"su": su})  # noqa: S307
        )
        assert REGISTRY[k["id"]] == expected, k["id"]
    assert {p["id"]: p["scale"] for p in canonical["prefixes"]} == PREFIXES
    # unit_from_json looks prefixes up on sympy.physics.units by id
    assert all(str(getattr(su, name).name) == name for name in PREFIXES)


def test_vendored_fixtures_match_mxl_schemas() -> None:
    assert json.loads(_canonical("tests/units.fixtures.json")) == FIXTURES


###############################################################################
# Shared fixtures
###############################################################################


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_fixture_from_json(case: dict[str, Any]) -> None:
    got = unit_from_json(case["unit"], case.get("units"))
    assert _same_unit(got, _fixture_sympy(case))


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_fixture_to_json(case: dict[str, Any]) -> None:
    customs: dict[str, dict[str, str]] = {}
    data = unit_to_json(_fixture_sympy(case), customs)
    # Composite registry kinds (lumen) legitimately decompose; compare by value.
    assert _same_unit(
        unit_from_json(data, customs), unit_from_json(case["unit"], case.get("units"))
    )
    assert set(customs) == set(case.get("units", {}))


@pytest.mark.parametrize(
    "name",
    [
        "mmol",
        "mmol_s",
        "mumol_h",
        "nmol_g",
        "pmol_m",
        "ppfd",
        "sqm",
        "cbm",
        "per_hour",
        "sievert",
        "lumen",
        "item",
        "celsius",
        "mol_chl",
        "mmol_mol_chl",
    ],
)
def test_alias_round_trip(name: str) -> None:
    unit = getattr(units, name)
    customs: dict[str, dict[str, str]] = {}
    assert _same_unit(unit_from_json(unit_to_json(unit, customs), customs), unit)
    assert customs == {}


def test_registry_ids_are_unique_and_atomic_kinds_map_back() -> None:
    for kind, expr in REGISTRY.items():
        if isinstance(expr, Quantity):
            assert unit_to_json(expr) == {"factors": [{"kind": kind, "exponent": 1}]}


def test_prefix_is_reattached_after_sympy_folds_it() -> None:
    # sympy turns `micro` into 1/1000000 here; the prefix must come back
    assert unit_to_json(units.ppfd)["factors"][0] == {
        "kind": "mole",
        "prefix": "micro",
        "exponent": 1,
    }


def test_kilogram_maps_to_prefixed_gram() -> None:
    assert unit_to_json(su.kilogram) == {
        "factors": [{"kind": "gram", "prefix": "kilo", "exponent": 1}]
    }


def test_non_registry_sympy_unit_is_converted() -> None:
    assert unit_to_json(su.day) == {
        "factors": [{"kind": "second", "exponent": 1}],
        "multiplier": 86400.0,
    }


def test_dimensionless() -> None:
    assert unit_to_json(None) == {"factors": []}
    assert unit_from_json({"factors": []}) is None


def test_custom_quantity_is_declared() -> None:
    customs: dict[str, dict[str, str]] = {}
    data = unit_to_json(Quantity("cells", abbrev="cells") / su.liter, customs)
    assert customs == {"cells": {}}
    assert unit_from_json(data, customs) == Quantity("cells", abbrev="cells") / su.liter


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="OD600"):
        unit_from_json({"factors": [{"kind": "OD600", "exponent": 1}]})


def test_unknown_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="kibi"):
        unit_from_json({"factors": [{"kind": "mole", "prefix": "kibi", "exponent": 1}]})


def test_non_integer_exponent_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        unit_to_json(su.meter ** sympy.Rational(1, 2))


###############################################################################
# .mxl.json
###############################################################################


def _model_with_units() -> KineticModelBuilder:
    od600 = Quantity("OD600", abbrev="OD600")
    return (
        KineticModelBuilder()
        .add_parameter("k", 1.0, unit=1 / su.second)
        .add_parameter("od", 0.5, unit=od600)
        .add_variable("S", 1.0, unit=units.mmol)
        .add_variable("P", 0.0, unit=units.mmol)
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"S": -1, "P": 1},
            args=["S", "k"],
            unit=units.mmol_s,
        )
    )


def test_model_to_dict_writes_units() -> None:
    data = model_to_dict(_model_with_units(), model_id="m")
    assert data["spec_version"] == "1.1"
    spec = data["model"]
    assert spec["variables"]["S"]["unit"] == {
        "factors": [{"kind": "mole", "prefix": "milli", "exponent": 1}]
    }
    assert spec["parameters"]["od"]["unit"] == {
        "factors": [{"kind": "OD600", "exponent": 1}]
    }
    assert spec["units"] == {"OD600": {}}
    assert "unit" not in spec["variables"]["S"]["value"]


def test_units_round_trip_through_mxl_json() -> None:
    model = _model_with_units()
    loaded = model_from_dict(model_to_dict(model, model_id="m"))
    for name, par in model.get_raw_parameters().items():
        assert _same_unit(loaded.get_raw_parameters()[name].unit, par.unit)
    for name, var in model.get_raw_variables().items():
        assert _same_unit(loaded.get_raw_variables()[name].unit, var.unit)
    assert _same_unit(
        loaded.get_raw_reactions()["v1"].unit, model.get_raw_reactions()["v1"].unit
    )


def test_model_without_units_has_no_units_section() -> None:
    model = KineticModelBuilder().add_parameter("k", 1.0)
    data = model_to_dict(model, model_id="m")
    assert "units" not in data["model"]
    assert "unit" not in data["model"]["parameters"]["k"]


def test_loading_undeclared_custom_kind_fails() -> None:
    data = model_to_dict(_model_with_units(), model_id="m")
    del data["model"]["units"]
    with pytest.raises(SerializationError, match="OD600"):
        model_from_dict(data)


def test_loaded_units_support_inference() -> None:
    loaded = model_from_dict(model_to_dict(_model_with_units(), model_id="m"))
    inferred = loaded.infer_units(su.second)
    assert _same_unit(inferred.reactions["v1"], units.mmol_s)  # type: ignore[arg-type]


###############################################################################
# Cross-language: artefacts generated by mxlweb-core
###############################################################################


def _mxlweb_model() -> KineticModelBuilder:
    """Load the module mxlweb-core's `buildMxlpy()` generated (tests/assets)."""
    path = ASSETS / "mxlweb_units_model.py"
    spec = importlib.util.spec_from_file_location("mxlweb_units_model", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model()


def test_mxlweb_generated_module_units_check_out() -> None:
    model = _mxlweb_model()
    assert _same_unit(model.get_raw_parameters()["light"].unit, units.ppfd)
    check = model.check_units(su.second)
    assert all(check.correct_diff_eqs().values())


def test_mxlweb_generated_module_supports_inference() -> None:
    inferred = _mxlweb_model().infer_units(su.second)
    # v2 has no explicit unit: inferred from k_out [1/s] * P [mmol]
    assert _same_unit(inferred.reactions["v2"], units.mmol_s)  # type: ignore[arg-type]


def test_mxlweb_mxl_json_loads_with_units() -> None:
    data = json.loads(
        (ASSETS / "mxlweb_units_model.mxl.json").read_text(encoding="utf-8")
    )
    loaded = model_from_dict(data)
    generated = _mxlweb_model()
    for name, par in generated.get_raw_parameters().items():
        assert _same_unit(loaded.get_raw_parameters()[name].unit, par.unit), name
    # and writing it back reproduces mxlweb-core's unit JSON exactly
    again = model_to_dict(loaded, model_id="x")["model"]
    for section in ("variables", "parameters", "reactions"):
        for name, entry in data["model"][section].items():
            assert again[section][name].get("unit") == entry.get("unit"), name
    assert again["units"] == {"OD600": {}}
