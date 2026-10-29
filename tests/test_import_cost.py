# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""`import mxlpy` must not pull in heavy optional or rarely used dependencies.

Every loky worker of a parallel scan re-imports mxlpy, so each of these costs
memory once per CPU core. They are imported lazily on first use instead.
"""

import subprocess
import sys

import pytest

import mxlpy

HEAVY_MODULES = (
    "diffrax",
    "equinox",
    "jax",
    "keras",
    "libsbml",
    "seaborn",
    "tensorflow",
    "torch",
)


def test_import_does_not_load_heavy_modules() -> None:
    code = (
        f"import sys, mxlpy; print(*(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.split() == []


def test_lazy_diffrax_resolves() -> None:
    pytest.importorskip("diffrax")
    from mxlpy.integrators.int_diffrax import Diffrax  # noqa: PLC0415

    assert mxlpy.Diffrax is Diffrax
    assert mxlpy.integrators.Diffrax is Diffrax


def test_lazy_sbml_resolves() -> None:
    assert callable(mxlpy.sbml.read)
