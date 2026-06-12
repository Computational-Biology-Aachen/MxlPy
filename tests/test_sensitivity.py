from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mxlpy import scan, sensitivity
from mxlpy.sensitivity import _build_problem, _evaluate
from tests.models import m_2v_2p_1d_1r

if TYPE_CHECKING:
    import pytest

    from mxlpy.model import Model


def _steady_state_output(variable: str) -> sensitivity.OutputFn:
    def output(model: Model, samples: pd.DataFrame) -> np.ndarray:
        return (
            scan.steady_state(model, to_scan=samples, parallel=False)
            .variables[variable]
            .to_numpy()
        )

    return output


def test_build_problem() -> None:
    problem = _build_problem({"k1": (0.1, 10.0), "k2": (0.01, 1.0)})
    assert problem["num_vars"] == 2
    assert problem["names"] == ["k1", "k2"]
    assert problem["bounds"] == [[0.1, 10.0], [0.01, 1.0]]


def test_morris_returns_labelled_dataframe() -> None:
    result = sensitivity.morris(
        m_2v_2p_1d_1r(),
        output=_steady_state_output("v1"),
        param_bounds={"p1": (0.1, 10.0), "p2": (0.01, 1.0)},
        n_trajectories=4,
        seed=0,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == ["p1", "p2"]
    assert list(result.columns) == ["mu", "mu_star", "sigma", "mu_star_conf"]


def test_morris_is_reproducible_with_seed() -> None:
    kwargs = {
        "output": _steady_state_output("v1"),
        "param_bounds": {"p1": (0.1, 10.0), "p2": (0.01, 1.0)},
        "n_trajectories": 4,
        "seed": 42,
    }
    first = sensitivity.morris(m_2v_2p_1d_1r(), **kwargs)
    second = sensitivity.morris(m_2v_2p_1d_1r(), **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_morris_evaluation_count() -> None:
    captured: dict[str, pd.DataFrame] = {}

    def output(_model: Model, samples: pd.DataFrame) -> np.ndarray:
        captured["samples"] = samples
        return np.zeros(len(samples))

    sensitivity.morris(
        m_2v_2p_1d_1r(),
        output=output,
        param_bounds={"p1": (0.1, 10.0), "p2": (0.01, 1.0)},
        n_trajectories=5,
        seed=0,
    )
    # Morris uses N * (k + 1) evaluations
    assert len(captured["samples"]) == 5 * (2 + 1)
    assert list(captured["samples"].columns) == ["p1", "p2"]


def test_evaluate_warns_on_non_finite(caplog: pytest.LogCaptureFixture) -> None:
    def output(_model: Model, samples: pd.DataFrame) -> np.ndarray:
        y = np.ones(len(samples))
        y[0] = np.nan
        return y

    sample_matrix = np.array([[1.0], [2.0], [3.0]])
    with caplog.at_level("WARNING"):
        y = _evaluate(m_2v_2p_1d_1r(), output, sample_matrix, ["p1"])

    assert np.isnan(y[0])
    assert "non-finite" in caplog.text


def test_morris_propagates_failed_runs_as_nan() -> None:
    # A model whose integration fails for every sample yields NaN outputs,
    # which surface as NaN sensitivity indices rather than raising.
    def output(_model: Model, samples: pd.DataFrame) -> np.ndarray:
        return np.full(len(samples), np.nan)

    result = sensitivity.morris(
        m_2v_2p_1d_1r(),
        output=output,
        param_bounds={"p1": (0.1, 10.0), "p2": (0.01, 1.0)},
        n_trajectories=4,
        seed=0,
    )
    assert bool(result["mu_star"].isna().all())
