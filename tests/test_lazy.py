# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Lazy optional-backend submodules (`mxlpy.{nn,npe,surrogates}.{torch,keras,equinox}`).

Each case runs in a fresh interpreter, since what's under test is how the
order of first imports plays out.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec

import pytest

import mxlpy

PACKAGES = ("nn", "npe", "surrogates")
BACKENDS = {"torch": "torch", "keras": "keras", "equinox": "jax"}


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_public_name_after_direct_import(package: str, backend: str) -> None:
    """`pkg.<backend>` works after `pkg._<backend>` was imported directly first."""
    if find_spec(backend) is None:
        pytest.skip(f"needs {backend}")
    res = _run(
        f"import mxlpy.{package}._{backend} as real\n"
        f"from mxlpy import {package}\n"
        f"assert {package}.{backend} is real\n"
    )
    assert res.returncode == 0, res.stderr


def test_nn_after_surrogates() -> None:
    if find_spec("torch") is None:
        pytest.skip("needs torch")
    res = _run(
        "from mxlpy import nn, surrogates\n"
        "surrogates.torch.Trainer\n"
        "assert nn.torch.MLP is surrogates.torch.MLP\n"
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize(("backend", "extra"), sorted(BACKENDS.items()))
def test_missing_backend_names_extra(package: str, backend: str, extra: str) -> None:
    res = _run(
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name.split('.')[0] == {backend!r}:\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import mxlpy\n"
        f"mxlpy.{package}.{backend}\n"
    )
    out = res.stdout + res.stderr
    assert res.returncode != 0
    assert f"mxlpy[{extra}]" in out
    assert "ImportError" in out


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="no attribute 'foo'"):
        _ = mxlpy.nn.foo  # type: ignore[attr-defined]
