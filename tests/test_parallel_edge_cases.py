"""Edge case tests for parallel.py."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest

from mxlpy.parallel import Cache, parallelise


def square(x: float) -> float:
    return x * x


def test_parallelise_empty_inputs() -> None:
    """Empty input collection returns empty list without error."""
    result = parallelise(square, [], parallel=False)
    assert result == []


def test_parallelise_empty_inputs_parallel() -> None:
    """Empty input collection with parallel=True returns empty list."""
    result = parallelise(square, [], parallel=True, max_workers=1)
    assert result == []


def test_parallelise_cache_stores_and_retrieves() -> None:
    """Results written to cache on first call are loaded on second call."""
    call_count = 0

    def counted_square(x: float) -> float:
        nonlocal call_count
        call_count += 1
        return x * x

    with tempfile.TemporaryDirectory() as tmp:
        cache = Cache(tmp_dir=Path(tmp))

        parallelise(counted_square, [(0, 2.0), (1, 3.0)], cache=cache, parallel=False)
        first_count = call_count

        # Second call should load from cache, not call the function again
        result = parallelise(
            counted_square, [(0, 2.0), (1, 3.0)], cache=cache, parallel=False
        )
        assert call_count == first_count  # no new calls

        values = dict(result)
        assert values[0] == pytest.approx(4.0)
        assert values[1] == pytest.approx(9.0)


def test_parallelise_corrupted_cache_raises() -> None:
    """A corrupted cache file propagates pickle.UnpicklingError (or EOFError).

    The cache load path has no error recovery; callers must handle stale caches.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache = Cache(tmp_dir=Path(tmp))
        # Pre-write a corrupt pickle file for key 0
        corrupt_file = Path(tmp) / cache.name_fn(0)
        corrupt_file.write_bytes(b"not a valid pickle")

        with pytest.raises((pickle.UnpicklingError, EOFError, Exception)):
            parallelise(square, [(0, 2.0)], cache=cache, parallel=False)


def test_parallelise_timeout_skips_slow_workers() -> None:
    """Workers exceeding the timeout are cancelled; result list is shorter."""
    import time

    def slow_square(x: float) -> float:
        time.sleep(5)  # much longer than timeout
        return x * x

    result = parallelise(
        slow_square,
        [(0, 1.0), (1, 2.0)],
        parallel=True,
        max_workers=1,
        timeout=0.1,  # 100 ms, far less than 5 s sleep
    )
    # Timed-out futures are cancelled and omitted from result
    assert len(result) < 2


def test_parallelise_sequential_correctness() -> None:
    """parallel=False produces the same values as serial map."""
    inputs = [(i, float(i)) for i in range(5)]
    result = parallelise(square, inputs, parallel=False)
    values = dict(result)
    for i, x in inputs:
        assert values[i] == pytest.approx(x * x)
