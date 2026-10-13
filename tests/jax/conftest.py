# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Per-test resource accounting for the JAX tests.

JAX has many sharp edges that silently multiply compilations or memory
(``lax.cond`` over fresh closures outside ``jit``, non-hashable or freshly
created static arguments, shape churn, large ``max_steps`` buffers, ...).
Every test here is therefore checked against a committed baseline,
``resource_baseline.json``:

* **XLA compilations** during the test: deterministic for a given JAX
  version, so they must not exceed the baseline at all (relative slack via
  ``MXLPY_RESOURCE_COMPILE_TOLERANCE``, default 0). Recompilation leaks add
  only one compile per repeated call, so even small increases matter.
* **Peak memory** above the test's starting RSS (Linux only): must stay
  within ``baseline * (1 + MXLPY_RESOURCE_MEM_TOLERANCE) + 64 MB``, with the
  tolerance defaulting to 0.5.

Compiled programs are dropped before and after every test, so each test
starts cold and its numbers don't depend on test order or xdist worker
assignment. Tests
missing from the baseline only emit a warning. After an intentional change,
refresh the baseline (and review its diff) with::

    MXLPY_UPDATE_RESOURCE_BASELINE=1 uv run pytest tests/jax
"""

import contextlib
import ctypes
import ctypes.util
import fcntl
import json
import os
import re
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import pytest

_BASELINE = Path(__file__).with_name("resource_baseline.json")
_UPDATE = os.environ.get("MXLPY_UPDATE_RESOURCE_BASELINE") == "1"
# Compile counts are deterministic for a pinned JAX version, and a per-call
# recompilation leak adds only one compile per repeated call -- a few percent
# of a test's total -- so by default any increase fails.
_COMPILE_REL = float(os.environ.get("MXLPY_RESOURCE_COMPILE_TOLERANCE", "0"))
_COMPILE_ABS = 0
_MEM_REL = float(os.environ.get("MXLPY_RESOURCE_MEM_TOLERANCE", "0.5"))
_MEM_ABS_MB = 64

try:
    _malloc_trim = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim
except (OSError, AttributeError):  # not glibc (e.g. macOS)
    _malloc_trim = None


# --- measurement -------------------------------------------------------------

_n_compiles = [0]


def _count_compile(event: str, duration_secs: float, **kwargs: str | int) -> None:  # noqa: ARG001
    if event == "/jax/core/compile/backend_compile_duration":
        _n_compiles[0] += 1


jax.monitoring.register_event_duration_secs_listener(_count_compile)

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")


def _status_mb(key: str) -> int:
    match = re.search(rf"{key}:\s+(\d+)", _STATUS.read_text())
    assert match is not None
    return int(match.group(1)) // 1024


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter (``VmHWM``); Linux only."""
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:
        return False
    return True


def _release_memory() -> None:
    jax.clear_caches()
    if _malloc_trim is not None:
        _malloc_trim(0)


def _warm_up() -> None:
    """Pay XLA/LLVM's one-time initialisation before any test is measured.

    Otherwise whichever test happens to run first in a process would be
    charged for it, making the numbers depend on test order.
    """

    def loss(y0: jax.Array) -> jax.Array:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),  # noqa: ARG005
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 1.0, 3)),
        )
        return jnp.sum(sol.ys)  # type: ignore[arg-type]

    jax.value_and_grad(loss)(jnp.ones(2))[0].block_until_ready()
    _release_memory()


_warm_up()


# --- baseline ----------------------------------------------------------------

_measured: dict[str, dict[str, int]] = {}


def _load_baseline() -> dict[str, dict[str, int]]:
    try:
        return json.loads(_BASELINE.read_text())
    except FileNotFoundError:
        return {}


_baseline = {} if _UPDATE else _load_baseline()


def _limit(baseline: int, rel: float, abs_: int) -> int:
    return int(baseline * (1 + rel)) + abs_


def _check(item: pytest.Item, compiles: int, peak_mb: int | None) -> None:
    entry = _baseline.get(item.nodeid)
    if entry is None:
        item.warn(
            pytest.PytestWarning(
                f"no resource baseline for this test (used {compiles} compiles, "
                f"{peak_mb} MB peak); add it with "
                "MXLPY_UPDATE_RESOURCE_BASELINE=1 uv run pytest tests/jax"
            )
        )
        return
    problems = []
    max_compiles = _limit(entry["compiles"], _COMPILE_REL, _COMPILE_ABS)
    if compiles > max_compiles:
        problems.append(
            f"{compiles} XLA compilations (baseline {entry['compiles']}, "
            f"limit {max_compiles}) -- likely a JAX recompilation regression: "
            "fresh closures/static args, lax.cond outside jit, shape churn"
        )
    if peak_mb is not None and "peak_mb" in entry:
        max_mb = _limit(entry["peak_mb"], _MEM_REL, _MEM_ABS_MB)
        if peak_mb > max_mb:
            problems.append(
                f"{peak_mb} MB peak memory (baseline {entry['peak_mb']} MB, "
                f"limit {max_mb} MB)"
            )
    if problems:
        pytest.fail(
            "resource regression: "
            + "; ".join(problems)
            + ". If intended, refresh the baseline with "
            "MXLPY_UPDATE_RESOURCE_BASELINE=1 uv run pytest tests/jax",
            pytrace=False,
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> object:
    # Start every test cold: programs compiled at import/collection time or
    # by fixtures would otherwise be free for whichever test runs first in a
    # process, making compile counts depend on test order.
    _release_memory()
    has_peak = _reset_peak_rss()
    start_mb = _status_mb("VmRSS") if has_peak else 0
    start_compiles = _n_compiles[0]
    result = yield  # re-raises if the test itself failed: nothing to check
    compiles = _n_compiles[0] - start_compiles
    peak_mb = _status_mb("VmHWM") - start_mb if has_peak else None
    if _UPDATE:
        _measured[item.nodeid] = {"compiles": compiles}
        if peak_mb is not None:
            _measured[item.nodeid]["peak_mb"] = peak_mb
    else:
        _check(item, compiles, peak_mb)
    return result


def pytest_sessionfinish(session: pytest.Session) -> None:  # noqa: ARG001
    """In update mode, merge this process's measurements into the baseline.

    Each xdist worker merges its own share under a file lock, so this works
    however the run is parallelised. Entries of deleted test files are
    pruned; delete the file and do a full update run to prune everything.
    """
    if not (_UPDATE and _measured):
        return
    root = Path(__file__).parents[2]
    with _BASELINE.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        baseline = _load_baseline()
        baseline.update(_measured)
        baseline = {
            k: v for k, v in baseline.items() if (root / k.split("::")[0]).exists()
        }
        _BASELINE.write_text(
            json.dumps(dict(sorted(baseline.items())), indent=1) + "\n"
        )
    with contextlib.suppress(FileNotFoundError):
        _BASELINE.with_suffix(".lock").unlink()


@pytest.fixture(autouse=True)
def _bound_memory() -> object:
    """Bound each xdist worker's memory by the largest single test.

    JAX keeps every compiled executable for the process lifetime, and each
    distinct model structure compiles its own diffrax solvers (forward,
    gradient, protocol, ...), so a worker's memory would otherwise grow with
    every test it runs. Tests rarely share compiled programs, so dropping
    them costs little recompilation. Freed memory is then handed back to the
    OS: XLA compilation allocates large temporary buffers across many
    threads, which glibc otherwise keeps in its per-thread arenas.
    """
    yield
    _release_memory()
