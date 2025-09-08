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

import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pebble
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable

__all__ = ["Cache", "parallelise", "parallelise_keyless"]


def _pickle_name(k: Hashable) -> str:
    return f"{k}.p"


def _pickle_load(file: Path) -> Any:
    with file.open("rb") as fp:
        return pickle.load(fp)  # nosec


def _pickle_save(file: Path, data: Any) -> None:
    with file.open("wb") as fp:
        pickle.dump(data, fp)


@dataclass
class Cache:
    """Cache class for storing and retrieving computation results.

    Attributes:
        tmp_dir: Directory to store cache files.
        name_fn: Function to generate file names from keys.
        load_fn: Function to load data from files.
        save_fn: Function to save data to files.

    """

    tmp_dir: Path = Path(".cache")
    name_fn: Callable[[Any], str] = _pickle_name
    load_fn: Callable[[Path], Any] = _pickle_load
    save_fn: Callable[[Path, Any], None] = _pickle_save


def _load_or_run[K: Hashable, Tin, Tout](
    inp: tuple[K, Tin],
    fn: Callable[[Tin], Tout],
    cache: Cache | None,
) -> tuple[K, Tout]:
    """Load data from cache or execute function and save result.

    Args:
        inp: Tuple containing a key and input value.
        fn: Function to execute if result is not in cache.
        cache: Optional cache to store and retrieve results.

    Returns:
        tuple[K, Tout]: Tuple containing the key and the result of the function.

    """
    k, v = inp
    if cache is None:
        res = fn(v)
    else:
        file = cache.tmp_dir / cache.name_fn(k)
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

    Examples:
        >>> parallelise(square, [("a", 2), ("b", 3), ("c", 4)])
        {"a": 4, "b": 9, "c": 16}

    Args:
        fn: Function to execute in parallel. Takes a single input and returns a result.
        inputs: Collection of (key, input) tuples to process.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).
        max_workers: Maximum number of worker processes (default: None, uses all available CPUs).
        timeout: Maximum time (in seconds) to wait for each worker to complete (default: None).
        disable_tqdm: Whether to disable the tqdm progress bar (default: False).
        tqdm_desc: Description for the tqdm progress bar (default: None).

    Returns:
        dict[Tin, Tout]: Dictionary mapping inputs to their corresponding outputs.

    """
    if cache is not None:
        cache.tmp_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform in ["win32", "cygwin"]:
        parallel = False

    worker: Callable[[K, Tin], tuple[K, Tout]] = partial(
        _load_or_run,
        fn=fn,
        cache=cache,
    )  # type: ignore

    results: list[tuple[K, Tout]]
    if parallel:
        results = []
        max_workers = (
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )

        with (
            tqdm(
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            ) as pbar,
            pebble.ProcessPool(max_workers=max_workers) as pool,
        ):
            future = pool.map(worker, inputs, timeout=timeout)
            it = future.result()
            while True:
                try:
                    key, value = next(it)
                    pbar.update(1)
                    results.append((key, value))
                except StopIteration:
                    break
                except TimeoutError:
                    pbar.update(1)
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

    Examples:
        >>> parallelise(square, [("a", 2), ("b", 3), ("c", 4)])
        {"a": 4, "b": 9, "c": 16}

    Args:
        fn: Function to execute in parallel. Takes a single input and returns a result.
        inputs: Collection of (key, input) tuples to process.
        cache: Optional cache to store and retrieve results.
        parallel: Whether to execute in parallel (default: True).
        max_workers: Maximum number of worker processes (default: None, uses all available CPUs).
        timeout: Maximum time (in seconds) to wait for each worker to complete (default: None).
        disable_tqdm: Whether to disable the tqdm progress bar (default: False).
        tqdm_desc: Description for the tqdm progress bar (default: None).

    Returns:
        dict[Tin, Tout]: Dictionary mapping inputs to their corresponding outputs.

    """
    if sys.platform in ["win32", "cygwin"]:
        parallel = False

    results: list[Tout]
    if parallel:
        results = []
        max_workers = (
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )

        with (
            tqdm(
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            ) as pbar,
            pebble.ProcessPool(max_workers=max_workers) as pool,
        ):
            future = pool.map(fn, inputs, timeout=timeout)
            it = future.result()
            while True:
                try:
                    value = next(it)
                    pbar.update(1)
                    results.append(value)
                except StopIteration:
                    break
                except TimeoutError:
                    pbar.update(1)
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
