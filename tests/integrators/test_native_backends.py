# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Native explains missing, broken and mismatched backends instead of failing obscurely."""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest.importorskip("mxlpy.integrators.int_native", exc_type=ImportError)

from mxlpy import OdeModelBuilder
from mxlpy.integrators._native import backends
from mxlpy.integrators.int_native import _AVAILABLE, Native

HAS_LLVM = importlib.util.find_spec("mxlpy_native") is not None
HAS_WASM = importlib.util.find_spec("wasmtime") is not None


@pytest.fixture(autouse=True)
def _fresh_probes() -> Iterator[None]:
    backends.problem.cache_clear()
    yield
    backends.problem.cache_clear()


def _decay(x: float) -> float:
    return -x


def _model() -> OdeModelBuilder:
    return OdeModelBuilder().add_diff_eq("x", fn=_decay, args=["x"], initial_value=1.0)


def test_unknown_backend_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="'auto', 'llvm' or 'wasm'"):
        Native(_model(), (1.0,), backend="gpu")  # type: ignore[arg-type]


@pytest.mark.skipif(not HAS_LLVM, reason="llvm backend not installed")
@pytest.mark.parametrize(
    ("abi", "hint"),
    [(backends.NATIVE_ABI - 1, "too old"), (backends.NATIVE_ABI + 1, "Upgrade MxlPy")],
)
def test_llvm_abi_mismatch(
    monkeypatch: pytest.MonkeyPatch, abi: int, hint: str
) -> None:
    import mxlpy_native  # noqa: PLC0415

    monkeypatch.setattr(mxlpy_native, "ABI_VERSION", abi)
    with pytest.raises(ImportError, match=hint):
        Native(_model(), (1.0,), backend="llvm")


@pytest.mark.skipif(not (HAS_LLVM and HAS_WASM), reason="needs both backends")
def test_auto_skips_a_broken_backend(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import mxlpy_native  # noqa: PLC0415

    monkeypatch.setattr(mxlpy_native, "ABI_VERSION", 0)
    with caplog.at_level(logging.WARNING):
        integrator = Native(_model(), (1.0,))
    assert integrator.backend == "wasm"
    assert "not the broken llvm backend" in caplog.text


def test_no_working_backend_lists_every_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_llvm_problem", lambda: "llvm is broken")
    monkeypatch.setattr(backends, "_wasm_problem", lambda: "wasm is broken")
    with pytest.raises(ImportError, match="None of Native's installed backends") as e:
        Native(_model(), (1.0,))
    for backend in _AVAILABLE:
        assert f"{backend} is broken" in str(e.value)


def test_unsupported_platform_points_to_wasm(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        backends.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "mxlpy_native" else real(name, *a),
    )
    monkeypatch.setattr(backends.platform, "machine", lambda: "riscv64")
    reason = backends.problem("llvm")
    assert reason is not None
    assert "no wheels" in reason
    assert "mxlpy[native-wasm]" in reason


@pytest.mark.skipif(not HAS_WASM, reason="wasm backend not installed")
def test_shipped_wasm_matches_abi() -> None:
    from mxlpy.integrators._native import wasm_backend  # noqa: PLC0415

    assert wasm_backend.abi_version() == backends.NATIVE_ABI
