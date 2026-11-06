# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Which Native backends are installed, and why one can't be used.

`problem(backend)` returns None if the backend works, and otherwise a message saying
what's wrong and how to fix it. `Native` raises these for an explicitly chosen backend,
and `auto` skips broken backends with a warning.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from functools import cache

__all__ = ["EXTRA", "NATIVE_ABI", "installed", "problem"]

# The solver ABI this MxlPy calls: MXN_ABI_VERSION in mxlpy-native's c/mxn_solvers.c,
# which solvers.wasm is built from too.
NATIVE_ABI = 2

EXTRA = {"llvm": "mxlpy[native]", "wasm": "mxlpy[native-wasm]"}
_MODULES = {"llvm": "mxlpy_native", "wasm": "wasmtime"}
# Where mxlpy-native has wheels, which is where llvmlite has them. Keep in sync with
# the markers on the `native` extra in pyproject.toml.
_LLVM_PLATFORMS = {
    ("linux", "x86_64"),
    ("linux", "aarch64"),
    ("darwin", "arm64"),
    ("win32", "AMD64"),
    ("win32", "ARM64"),
}


def installed() -> tuple[str, ...]:
    """Backends whose packages are installed, fastest first. Cheap: imports nothing."""
    return tuple(
        b for b, m in _MODULES.items() if importlib.util.find_spec(m) is not None
    )


def _this_platform() -> str:
    return f"{sys.platform} {platform.machine()}"


def _abi_problem(package: str, version: str, abi: int) -> str | None:
    if abi == NATIVE_ABI:
        return None
    if abi < NATIVE_ABI:
        return (
            f"{package} {version} is too old for this MxlPy (solver ABI {abi}, needs "
            f'{NATIVE_ABI}). Upgrade it: pip install -U "mxlpy[native]"'
        )
    return (
        f"{package} {version} is newer than this MxlPy supports (solver ABI {abi}, "
        f"needs {NATIVE_ABI}). Upgrade MxlPy: pip install -U mxlpy"
    )


def _llvm_problem() -> str | None:
    if importlib.util.find_spec("mxlpy_native") is None:
        if (sys.platform, platform.machine()) not in _LLVM_PLATFORMS:
            return (
                f"mxlpy-native has no wheels for {_this_platform()}, because llvmlite has none. "
                'Use the wasm backend instead: pip install "mxlpy[native-wasm]"'
            )
        return 'mxlpy-native isn\'t installed: pip install "mxlpy[native]"'
    try:
        import mxlpy_native  # noqa: PLC0415

        version = getattr(mxlpy_native, "__version__", "<0.3")
        if msg := _abi_problem(
            "mxlpy-native", version, getattr(mxlpy_native, "ABI_VERSION", 0)
        ):
            return msg
        mxlpy_native.load()
    except ImportError as e:  # includes mxlpy_native.NativeInstallError
        return str(e)
    except OSError as e:
        return f'mxlpy-native is broken ({e}). Reinstall it: pip install --force-reinstall "mxlpy[native]"'
    try:
        import llvmlite.binding  # noqa: F401, PLC0415
    except ImportError as e:
        return (
            f"llvmlite, which mxlpy-native needs, can't be imported ({e}). "
            'Reinstall it: pip install --force-reinstall "mxlpy[native]"'
        )
    return None


def _wasm_problem() -> str | None:
    if importlib.util.find_spec("wasmtime") is None:
        return 'wasmtime isn\'t installed: pip install "mxlpy[native-wasm]"'
    try:
        wasm_backend = importlib.import_module("mxlpy.integrators._native.wasm_backend")
        abi = wasm_backend.abi_version()
    except FileNotFoundError as e:
        return f"MxlPy's solvers.wasm is missing ({e}). Reinstall MxlPy: pip install --force-reinstall mxlpy"
    except Exception as e:  # noqa: BLE001  wasmtime raises its own error types
        return f"wasmtime can't load MxlPy's solvers.wasm ({type(e).__name__}: {e}). Try: pip install -U wasmtime"
    if abi != NATIVE_ABI:
        return (
            f"MxlPy's solvers.wasm has solver ABI {abi}, but this MxlPy needs {NATIVE_ABI}. "
            "It's stale: rebuild it with scripts/build_native_wasm.sh, or report this at "
            "https://github.com/Computational-Biology-Aachen/MxlPy/issues if it came from PyPI"
        )
    return None


@cache
def problem(backend: str) -> str | None:
    """Return why `backend` can't be used, or None if it works."""
    if backend == "llvm":
        return _llvm_problem()
    if backend == "wasm":
        return _wasm_problem()
    return f"unknown backend {backend!r}, use 'llvm', 'wasm' or 'auto'"
