# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from mxlpy.meta.identifiers import (
    RESERVED,
    SYMBOLS,
    TRANSLITERATE,
    canonical_identifier,
    canonical_identifier_map,
    is_canonical_identifier,
)

ASSETS = Path(__file__).parents[1] / "assets"
FIXTURES = json.loads(
    (ASSETS / "identifiers.fixtures.json").read_text(encoding="utf-8")
)
UPSTREAM = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/mxl-schemas/main"
)
# Sibling checkout inside the 0-admin meta-repo, when present.
META_SCHEMAS = Path(__file__).resolve().parents[4] / "misc" / "mxl-schemas"


def _canonical(rel: str) -> str:
    local = META_SCHEMAS / rel
    if local.exists():
        return local.read_text(encoding="utf-8")
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/{rel}", timeout=10) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except OSError as exc:
        pytest.skip(f"canonical {rel} unavailable: {exc}")


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: repr(c["input"]))
def test_canonical_identifier(case: dict[str, str]) -> None:
    out = canonical_identifier(case["input"])
    assert out == case["output"]
    assert is_canonical_identifier(out)
    assert canonical_identifier(out) == out


@pytest.mark.parametrize(
    "case", FIXTURES["map_cases"], ids=lambda c: "|".join(c["inputs"])
)
def test_canonical_identifier_map(case: dict[str, Any]) -> None:
    out = canonical_identifier_map(case["inputs"])
    assert list(out.values()) == case["outputs"]


@pytest.mark.parametrize("names", FIXTURES["collisions"], ids="|".join)
def test_canonical_identifier_map_collision(names: list[str]) -> None:
    with pytest.raises(ValueError, match="both map to"):
        canonical_identifier_map(names)


def test_canonical_identifier_map_keeps_canonical_names() -> None:
    names = ["x", "y_2", "z"]
    assert canonical_identifier_map(names) == {n: n for n in names}


def test_vendored_rules_match_mxl_schemas() -> None:
    spec = json.loads(_canonical("v1/identifiers.json"))
    assert set(spec["reserved"]) == RESERVED
    assert spec["symbols"] == SYMBOLS
    assert spec["transliterate"] == TRANSLITERATE


def test_vendored_fixtures_match_mxl_schemas() -> None:
    assert json.loads(_canonical("tests/identifiers.fixtures.json")) == FIXTURES
