# ADR 0001: Process-Pool Backend for `parallel.py` (pebble → loky)

**Status:** Implemented
**Scope:** `src/mxlpy/parallel.py`, `src/mxlpy/fit/_joint.py`, `src/mxlpy/fit/_joint_mixed.py`, `src/mxlpy/fuzzy.py`

---

## 0. Background — why a third-party pool at all, and why `pebble` originally

`parallelise()` and friends fan work out across many independent, CPU-bound scientific
runs — e.g. parameter scans, Monte Carlo ensembles, or per-sample steady-state search
during fitting. In this workload it is expected and routine for a single task to
misbehave: a particular parameterisation can start oscillating and never settle, and the
integrator will burn CPU time trying to find a steady state that most likely doesn't
exist for that parameter set. The stdlib `multiprocessing.Pool` /
`concurrent.futures.ProcessPoolExecutor` has no notion of a **per-task timeout** — there
is no built-in way to say "kill this one worker's task after N seconds and move on,"
short of hand-rolling process supervision.

`pebble` was originally chosen specifically because it layers per-task timeouts (with
process-level termination) on top of the standard pool APIs. That capability — not
raw throughput — was the deciding factor: in a batch of thousands of scans, a handful of
stuck/failing tasks should be killed and recorded as failures, not allowed to stall the
whole batch.

`loky` (the replacement, see below) was required to preserve this same timeout/termination
behaviour; a plain `ProcessPoolExecutor` migration would have silently dropped it.

---

## 1. Problem Statement

Python 3.14 changes the default multiprocessing start method on POSIX from `fork` to
`forkserver`. The `forkserver` and `spawn` methods share the same constraint: worker
processes must import `__main__` to reconstruct the execution context, and they rely on
standard `pickle` for task serialisation.

This produces two breakage modes in mxlpy:

1. **Jupyter notebooks** — there is no importable `__main__` module in a notebook kernel.
   Every `pebble.ProcessPool` call in a notebook raises
   `RuntimeError: ... __main__ not found`.
2. **Interactively-defined callables** — functions defined in a notebook cell are not
   importable by name, so `pickle` fails to serialise them when passed as the `fn`
   argument to `parallelise()` or as `worker` in `thompson_sampling()`.

Both failure modes are triggered today on macOS (where `spawn` is already the default
since Python 3.8) and become universal on POSIX with Python 3.14.

---

## 2. Options Considered

### Option A — Explicitly pin `fork` via `multiprocessing.get_context('fork')`

Pass a `fork` context explicitly to `pebble.ProcessPool(context=...)`.

**Pros:** Zero API or dependency change.
**Cons:** `fork` is unsafe in multithreaded programs (numpy, scipy, and the ODE
integrators all use threads). It is deprecated on macOS and being phased out. Pinning
it is the wrong long-term direction and would reintroduce the same deadlock risks that
motivated Python's default change.

### Option B — Switch to `loky` ✓ CHOSEN

`loky` is the process-pool implementation used internally by joblib/scikit-learn. It
uses spawn internally but substitutes **cloudpickle** for serialisation, which handles
interactively-defined functions. It does not require an importable `__main__`, so
Jupyter notebooks work out of the box. Its API is
`concurrent.futures.ProcessPoolExecutor`-compatible.

**Pros:** Solves both breakage modes permanently. Widely deployed in scientific Python.
No public API changes. Windows special-casing can be removed (loky is cross-platform).
**Cons:** Per-task timeout semantics differ from pebble (see §5).

### Option C — `multiprocess` (community fork of `multiprocessing` using `dill`)

Drop-in `multiprocessing` replacement that uses `dill` for serialisation. `dill` is
already a declared dependency.

**Pros:** `dill` already present.
**Cons:** `multiprocess` is poorly maintained; its API is not `concurrent.futures`-
compatible. `loky` is better maintained and better tested in the scipy ecosystem.

---

## 3. Decision and Rationale

Replace `pebble` with `loky` across all four affected files.

- cloudpickle handles notebook-defined callables; loky's bootstrap does not require
  `__main__`.
- `loky` backs `joblib.Parallel` and is deployed at significant scale in scientific
  Python.
- The `concurrent.futures.ProcessPoolExecutor` interface is standard-library-shaped,
  reducing future coupling to third-party pool abstractions.
- `dill` (already declared) is retained for the `Cache` pickle paths in `parallel.py`,
  and separately for source-inspection round-trips in `meta/source_tools.py` and
  `jax/io.py` — unrelated to the pool backend itself.
- The `sys.platform in ["win32", "cygwin"]` guards in `parallel.py` and `fuzzy.py`
  can be removed — loky handles all platforms natively.

---

## 4. File-by-File Migration Plan

### 4.1 `src/mxlpy/parallel.py`

- Replace `import pebble` with `from loky import ProcessPoolExecutor`.
- In `parallelise` and `parallelise_keyless`:
  - Remove the `sys.platform in ["win32", "cygwin"]` guard.
  - Replace `pebble.ProcessPool(max_workers=n)` context manager with
    `ProcessPoolExecutor(max_workers=n)`.
  - Replace the `pool.map(..., timeout=t).result()` iterator pattern with
    `pool.submit(fn, inp)` per input, then `future.result(timeout=t)` per future inside
    the tqdm loop.
  - Catch `concurrent.futures.TimeoutError` instead of pebble's `TimeoutError`.
- Remove `import sys` if no longer needed after the platform guard removal.

### 4.2 `src/mxlpy/fit/_joint.py`

- Replace `import pebble` with the loky import.
- Update the `pool` parameter type annotation on `_sum_of_residuals` from
  `pebble.ProcessPool` to `ProcessPoolExecutor`.
- Replace `pool.map(...).result()` iterator with submit/collect/`future.result()`.
- The pool is created once per outer function call and passed into `_sum_of_residuals`
  via `partial`; this long-lived pool pattern is compatible with loky without structural
  change.
- Catch `concurrent.futures.TimeoutError`.

### 4.3 `src/mxlpy/fit/_joint_mixed.py`

Identical changes to §4.2: update `pool` type on `_mixed_sum_of_residuals`, replace
pool construction, replace the map/result iterator, catch the correct `TimeoutError`.

### 4.4 `src/mxlpy/fuzzy.py`

- Replace `import pebble` with the loky import.
- Remove the `sys.platform in ["win32", "cygwin"]` guard.
- Replace `pebble.ProcessPool` context manager with `ProcessPoolExecutor`.
- Inner loop submits `max_workers` tasks per chunk; replace `pool.map(...,
  timeout=timeout).result()` with list-of-futures + per-future `.result(timeout=timeout)`.
- Catch `concurrent.futures.TimeoutError`.
- Remove `import sys` if no longer needed.

---

## 5. Known Limitations

**Timeout behaviour change.** pebble terminates the OS process for a timed-out task
immediately via the iterator protocol. With loky, `future.result(timeout=t)` raises
`TimeoutError` on the caller but does not kill the worker on its own.

This is handled explicitly: after collecting all results (including timeouts), the pool
is shut down via `pool.shutdown(wait=False, kill_workers=True)`. This sends SIGTERM to
all worker processes, including any still running a timed-out task, and returns
immediately without blocking. The net effect is similar to pebble — timed-out tasks
are terminated — but the termination happens at pool cleanup rather than per-task.

Callers should be aware that tasks that complete _just before_ SIGTERM arrives will
have their results discarded if `shutdown` is called before `future.result()` returns.
In practice this window is negligible for normal scientific workloads.

---

## 6. Dependency Changes (`pyproject.toml`)

| Action | Package |
|--------|---------|
| Remove | `pebble>=5.0.7` |
| Add    | `loky>=3.4` |
| Keep   | `dill>=0.3.9` (used by `Cache` pickle paths) |

- Change `requires-python = ">=3.12,<3.14"` → `">=3.12"`.
- Add Python 3.14 classifier.
- Regenerate `uv.lock` and `pixi.lock` after the change.
