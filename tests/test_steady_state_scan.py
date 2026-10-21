# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mxlpy import SteadyStateModelBuilder
from mxlpy.parallel import Cache

if TYPE_CHECKING:
    from pathlib import Path


def _double(value: float) -> float:
    return 2.0 * value


def _make_model() -> SteadyStateModelBuilder:
    return (
        SteadyStateModelBuilder()
        .add_parameter("p", 1.0)
        .add_derived("d", fn=_double, args=["p"])
    )


def test_simulate_scan_applies_parameters_per_row() -> None:
    model = _make_model()
    scan = pd.DataFrame({"p": [1.0, 2.0, 3.0]})

    result = model.simulate_scan(scan, parallel=False)

    expected = pd.DataFrame(
        {"d": [2.0, 4.0, 6.0]},
        index=pd.Index([1.0, 2.0, 3.0], name="p"),
    )
    pd.testing.assert_frame_equal(result, expected)
    assert model.get_raw_parameters()["p"].value == 1.0


def test_simulate_scan_parallel_matches_serial() -> None:
    scan = pd.DataFrame({"p": [1.0, 2.0, 3.0]})

    serial = _make_model().simulate_scan(scan, parallel=False)
    parallel = _make_model().simulate_scan(scan, parallel=True)

    pd.testing.assert_frame_equal(parallel, serial)


def test_simulate_scan_uses_multi_index_for_multiple_scan_parameters() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("p1", 1.0)
        .add_parameter("p2", 2.0)
        .add_derived("d", fn=lambda p1, p2: p1 + p2, args=["p1", "p2"])
    )
    scan = pd.DataFrame({"p1": [1.0, 3.0], "p2": [2.0, 4.0]})

    result = model.simulate_scan(scan, parallel=False)

    expected = pd.DataFrame(
        {"d": [3.0, 7.0]},
        index=pd.MultiIndex.from_tuples(
            [(1.0, 2.0), (3.0, 4.0)],
            names=["p1", "p2"],
        ),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_simulate_scan_cache_reuses_results(tmp_path: Path) -> None:
    scan = pd.DataFrame({"p": [1.0, 2.0]})
    cache = Cache(tmp_dir=tmp_path)

    first = _make_model().simulate_scan(scan, cache=cache, parallel=False)
    second = _make_model().simulate_scan(scan, cache=cache, parallel=False)

    pd.testing.assert_frame_equal(second, first)
    assert sorted(path.name for path in tmp_path.glob("*.p")) == ["0.p", "1.p"]
