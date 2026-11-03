# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""The solvers in native/c/vendor must be exactly what mxl-solvers' vendor.sh copied.

Fix solvers in https://github.com/Computational-Biology-Aachen/mxl-solvers and re-run
its scripts/vendor.sh, rather than editing the copies here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

VENDOR = Path(__file__).parents[2] / "native" / "c" / "vendor"


def _manifest() -> dict[str, str]:
    lines = (VENDOR / "VENDORED").read_text().splitlines()
    return {
        name: digest
        for digest, name in (line.split() for line in lines if line and not line.startswith("#"))
    }


@pytest.mark.skipif(not VENDOR.exists(), reason="native sources aren't in this checkout")
def test_vendored_solvers_are_unmodified() -> None:
    manifest = _manifest()
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in VENDOR.iterdir()
        if p.name != "VENDORED"
    }
    assert actual == manifest
