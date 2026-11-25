# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Parallel Execution Module.

This module provides functions and classes for parallel execution and caching of
computation results. It includes functionality for parallel processing and result
caching using multiprocessing and pickle.

Classes:
    Cache: Cache class for storing and retrieving computation results.

Functions:
    parallelise: Execute a function in parallel over a collection of inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loky import ProcessPoolExecutor, cpu_count
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable

__all__ = ["Cache", "content_hash", "parallelise", "parallelise_keyless"]


# (export, install) pairs: export runs in the parent before a pool starts, and
# returns state for the workers (None for nothing), which install receives in each.
_WORKER_STATE: list[tuple[Callable[[], Any], Callable[[Any], None]]] = []


def register_worker_state(
    export: Callable[[], Any], install: Callable[[Any], None]
) -> None:
    """Send state of this process to every worker that `parallelise` starts.

    `export` runs before a pool starts and returns the state, or None if there is
    none. Each worker passes it to `install` when it starts. Both have to be
    module-level functions, so that workers can unpickle them.
    """
    _WORKER_STATE.append((export, install))


def _install_worker_state(states: list[tuple[Callable[[Any], None], Any]]) -> None:
    for install, state in states:
        install(state)


# Rough memory of a worker: importing mxlpy alone takes ~120 MiB, plus the
# model and its results
_WORKER_MEMORY = 256 * 2**20


def _cgroup_headroom() -> int | None:
    """Memory left before the tightest cgroup v2 limit of this process (Linux).

    Limits can be set on any ancestor, e.g. a systemd slice of the session.
    """
    try:
        with open("/proc/self/cgroup") as f:  # noqa: PTH123
            path = next(
                line.split(":", 2)[2].strip() for line in f if line.startswith("0::")
            )
    except (OSError, StopIteration):
        return None
    root = Path("/sys/fs/cgroup")
    headroom = None
    for directory in [root / path.lstrip("/"), *(root / path.lstrip("/")).parents]:
        if directory == root.parent:
            break
        try:
            limit = (directory / "memory.max").read_text().strip()
            current = int((directory / "memory.current").read_text())
        except (OSError, ValueError):
            continue
        if limit != "max":
            left = int(limit) - current
            headroom = left if headroom is None else min(headroom, left)
    return headroom


def _available_memory() -> int | None:
    """Memory available for new processes (Linux), None if unknown."""
    available = None
    try:
        with open("/proc/meminfo") as f:  # noqa: PTH123
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if (headroom := _cgroup_headroom()) is not None:
        available = headroom if available is None else min(available, headroom)
    return available


def _n_workers(max_workers: int | None, n_inputs: int) -> int:
    """Number of workers for `n_inputs` inputs.

    Without `max_workers`, as many as there are inputs, CPUs available to
    this process (affinity, cgroup CPU quotas and the LOKY_MAX_CPU_COUNT
    environment variable, see `loky.cpu_count`) and workers fitting into the
    available memory (Linux), whichever is smallest.
    """
    if max_workers is None:
        max_workers = cpu_count()
        if (memory := _available_memory()) is not None:
            max_workers = min(max_workers, memory // _WORKER_MEMORY)
    return max(1, min(max_workers, n_inputs))


def _pool(max_workers: int | None, n_inputs: int) -> ProcessPoolExecutor:
    states = [
        (install, state)
        for export, install in _WORKER_STATE
        if (state := export()) is not None
    ]
    return ProcessPoolExecutor(
        max_workers=_n_workers(max_workers, n_inputs),
        initializer=_install_worker_state if states else None,
        initargs=(states,),
    )


def _pickle_name(k: Hashable) -> str:
    return f"{k}.p"


def _pickle_load(file: Path) -> Any:
    with file.open("rb") as fp:
        return pickle.load(fp)  # nosec


def _pickle_save(file: Path, data: Any) -> None:
    with file.open("wb") as fp:
        pickle.dump(data, fp)


def _model_bytes(model: Any) -> bytes | None:
    """Canonical bytes of a model builder via its `.mxl.json` form, None for others."""
    from mxlpy import serialize  # noqa: PLC0415 (slow import, only needed here)
    from mxlpy._kinetic_builder import KineticModelBuilder  # noqa: PLC0415
    from mxlpy._ode_builder import OdeModelBuilder  # noqa: PLC0415
    from mxlpy._steady_state_builder import SteadyStateModelBuilder  # noqa: PLC0415

    to_dict: Callable[..., dict[str, Any]]
    if isinstance(model, KineticModelBuilder):
        to_dict = serialize.model_to_dict
    elif isinstance(model, OdeModelBuilder):
        to_dict = serialize.ode_model_to_dict
    elif isinstance(model, SteadyStateModelBuilder):
        to_dict = serialize.steady_state_model_to_dict
    else:
        return None
    try:
        data = to_dict(model, model_id="model")
    except Exception:  # noqa: BLE001 (anything the format can't express)
        return None
    return json.dumps(data, sort_keys=True, default=repr).encode()


def _update_hash(h: Any, obj: Any) -> None:
    """Feed a canonical, type-tagged byte form of `obj` into the hash `h`."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    h.update(type(obj).__qualname__.encode() + b"\0")
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        h.update(repr(obj).encode())
    elif isinstance(obj, (list, tuple)):
        h.update(str(len(obj)).encode())
        for i in obj:
            _update_hash(h, i)
    elif isinstance(obj, (set, frozenset)):
        for i in sorted(obj, key=repr):
            _update_hash(h, i)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _update_hash(h, k)
            _update_hash(h, obj[k])
    elif isinstance(obj, pd.DataFrame):
        _update_hash(h, list(obj.index))
        for name in obj.columns:
            _update_hash(h, name)
            _update_hash(h, obj[name].to_numpy())
    elif isinstance(obj, pd.Series):
        _update_hash(h, list(obj.index))
        _update_hash(h, obj.name)
        _update_hash(h, obj.to_numpy())
    elif isinstance(obj, np.ndarray):
        h.update(f"{obj.dtype}{obj.shape}".encode())
        if obj.dtype.hasobject:
            # Object arrays hold pointers, hash their elements instead
            _update_hash(h, obj.ravel().tolist())
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, np.generic):
        h.update(repr(obj.item()).encode())
    elif (model := _model_bytes(obj)) is not None:
        h.update(model)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _update_hash(h, f.name)
            _update_hash(h, getattr(obj, f.name))
    else:
        import dill  # noqa: PLC0415

        h.update(dill.dumps(obj))


def content_hash(*objs: Any) -> str:
    """Stable sha256 hex digest of the contents of `objs`.

    Containers, numbers, strings, numpy arrays and pandas objects are hashed
    by value (independent of dict order). Models are hashed via their
    `.mxl.json` form, so changing e.g. a rate function changes the hash.
    Models the format can't express and other objects are hashed via their
    pickle, which identifies module-level functions by name only.

    Examples
    --------
        >>> content_hash(model, {"k1": 1.0}) == content_hash(model, {"k1": 1.0})
        True

    """
    h = hashlib.sha256()
    for obj in objs:
        _update_hash(h, obj)
    return h.hexdigest()


@dataclass
class Cache:
    """Cache class for storing and retrieving computation results.

    Attributes
    ----------
    tmp_dir
        Directory to store cache files.
    name_fn
        Function to generate file names from keys.
    load_fn
        Function to load data from files.
    save_fn
        Function to save data to files.
    content_hash
        Add a hash of `salt` and the input to every file name, so changed
        inputs are recomputed instead of loaded, e.g. when resuming with
        different settings.
    salt
        Anything else the results depend on, e.g. the model, hashed with
        `content_hash` (only used with `content_hash=True`).

    """

    tmp_dir: Path = Path(".cache")
    name_fn: Callable[[Any], str] = _pickle_name
    load_fn: Callable[[Path], Any] = _pickle_load
    save_fn: Callable[[Path, Any], None] = _pickle_save
    content_hash: bool = False
    salt: Any = None

    def file(self, key: Hashable, value: Any, salt_digest: str = "") -> Path:
        """File of the result for `key` (and `value` with `content_hash`)."""
        file = self.tmp_dir / self.name_fn(key)
        if not self.content_hash:
            return file
        digest = content_hash(salt_digest, value)[:16]
        return file.with_name(f"{file.stem}-{digest}{file.suffix}")


def _load_or_run[K: Hashable, Tin, Tout](
    inp: tuple[K, Tin],
    fn: Callable[[Tin], Tout],
    cache: Cache | None,
    salt_digest: str = "",
) -> tuple[K, Tout]:
    """Load data from cache or execute function and save result.

    Parameters
    ----------
    inp
        Tuple containing a key and input value.
    fn
        Function to execute if result is not in cache.
    cache
        Optional cache to store and retrieve results.
    salt_digest
        Content hash of the cache's salt, computed once by `parallelise`

    Returns
    -------
    tuple[K, Tout]
        Tuple containing the key and the result of the function.

    """
    k, v = inp
    if cache is None:
        res = fn(v)
    else:
        file = cache.file(k, v, salt_digest)
        if file.exists():
            return k, cast(Tout, cache.load_fn(file))
        res = fn(v)
        cache.save_fn(file, res)
    return k, res


def parallelise[K: Hashable, Tin, Tout](
    fn: Callable[[Tin], Tout],
    inputs: Collection[tuple[K, Tin]],
    *,
    cache: Cache | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    disable_tqdm: bool = False,
    tqdm_desc: str | None = None,
) -> list[tuple[K, Tout]]:
    """Execute a function in parallel over a collection of inputs.

    Examples
    --------
        >>> parallelise(square, [("a", 2), ("b", 3), ("c", 4)])
        {"a": 4, "b": 9, "c": 16}

    Parameters
    ----------
    fn
        Function to execute in parallel. Takes a single input and returns a result.
    inputs
        Collection of (key, input) tuples to process.
    cache
        Optional cache to store and retrieve results.
    parallel
        Whether to execute in parallel (default: True).
    max_workers
        Maximum number of worker processes. By default as many as there are
        inputs, available CPUs and workers fitting into the available memory.
    timeout
        Maximum time (in seconds) to wait for each worker to complete (default: None).
    disable_tqdm
        Whether to disable the tqdm progress bar (default: False).
    tqdm_desc
        Description for the tqdm progress bar (default: None).

    Returns
    -------
    dict[Tin, Tout]
        Dictionary mapping inputs to their corresponding outputs.

    """
    salt_digest = ""
    if cache is not None:
        cache.tmp_dir.mkdir(parents=True, exist_ok=True)
        if cache.content_hash:
            salt_digest = content_hash(cache.salt)

    worker: Callable[[K, Tin], tuple[K, Tout]] = partial(
        _load_or_run,
        fn=fn,
        cache=cache,
        salt_digest=salt_digest,
    )  # type: ignore

    results: list[tuple[K, Tout]]
    if parallel:
        results = []
        pool = _pool(max_workers, len(inputs))
        with tqdm(
            total=len(inputs),
            disable=disable_tqdm,
            desc=tqdm_desc,
        ) as pbar:
            try:
                futures = [pool.submit(worker, inp) for inp in inputs]
                for future in futures:
                    try:
                        key, value = future.result(timeout=timeout)
                        pbar.update(1)
                        results.append((key, value))
                    except FuturesTimeoutError:
                        future.cancel()
                        pbar.update(1)
            finally:
                pool.shutdown(wait=False, kill_workers=True)
    else:
        results = list(
            tqdm(
                map(worker, inputs),  # type: ignore
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            )
        )  # type: ignore

    return results


def parallelise_keyless[Tin, Tout](
    fn: Callable[[Tin], Tout],
    inputs: Collection[Tin],
    *,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    disable_tqdm: bool = False,
    tqdm_desc: str | None = None,
) -> list[Tout]:
    """Execute a function in parallel over a collection of inputs.

    Examples
    --------
        >>> parallelise(square, [("a", 2), ("b", 3), ("c", 4)])
        {"a": 4, "b": 9, "c": 16}

    Parameters
    ----------
    fn
        Function to execute in parallel. Takes a single input and returns a result.
    inputs
        Collection of (key, input) tuples to process.
    parallel
        Whether to execute in parallel (default: True).
    max_workers
        Maximum number of worker processes. By default as many as there are
        inputs, available CPUs and workers fitting into the available memory.
    timeout
        Maximum time (in seconds) to wait for each worker to complete (default: None).
    disable_tqdm
        Whether to disable the tqdm progress bar (default: False).
    tqdm_desc
        Description for the tqdm progress bar (default: None).

    Returns
    -------
    dict[Tin, Tout]
        Dictionary mapping inputs to their corresponding outputs.

    """
    results: list[Tout]
    if parallel:
        results = []
        pool = _pool(max_workers, len(inputs))
        with tqdm(
            total=len(inputs),
            disable=disable_tqdm,
            desc=tqdm_desc,
        ) as pbar:
            try:
                futures = [pool.submit(fn, inp) for inp in inputs]
                for future in futures:
                    try:
                        value = future.result(timeout=timeout)
                        pbar.update(1)
                        results.append(value)
                    except FuturesTimeoutError:
                        future.cancel()
                        pbar.update(1)
            finally:
                pool.shutdown(wait=False, kill_workers=True)
    else:
        results = list(
            tqdm(
                map(fn, inputs),  # type: ignore
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            )
        )  # type: ignore

    return results
