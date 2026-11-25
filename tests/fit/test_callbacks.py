# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns
from mxlpy.distributions import Normal
from mxlpy.fit.callbacks import FitEvent, JsonlCallback
from mxlpy.minimizers.abstract import Bounds, OptimisationState, Residual
from mxlpy.types import FitFailure, Result

SIGMA = 0.01
NLL = fit.GaussianNll(SIGMA)
P0 = {"k1": 0.8, "k2": 1.5}


def _model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


def _data() -> pd.DataFrame:
    t = np.linspace(0.2, 6.0, 15)
    res = Simulator(_model()).simulate_time_course(t).get_result().unwrap_or_err()
    data = res.get_combined().loc[t, ["x", "y"]]
    return data + np.random.default_rng(0).normal(0, SIGMA, data.shape)


class _FailAbove:
    """Minimizer failing for starts with k1 above a limit (picklable)."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def __call__(
        self, residual_fn: Residual, p0: dict[str, float], bounds: Bounds
    ) -> Result[OptimisationState]:
        if p0["k1"] > self.limit:
            return Result(FitFailure(extra_info=["too large"]))
        return fit.LeastSquaresMinimizer()(residual_fn, p0, bounds)


@pytest.mark.parametrize(
    "minimizer", [fit.LeastSquaresMinimizer(), fit.LocalScipyMinimizer()]
)
def test_single_fit_events(minimizer: fit.LeastSquaresMinimizer) -> None:
    events: list[FitEvent] = []
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=minimizer,
        loss_fn=NLL,
        callback=events.append,
    ).unwrap_or_err()
    evaluations = [e for e in events if e.kind == "evaluation"]
    assert evaluations
    assert events[-1].kind == "finished"
    assert sum(e.kind == "finished" for e in events) == 1
    assert [e.n_evaluations for e in evaluations] == list(
        range(1, len(evaluations) + 1)
    )
    # best is monotone and parameters are only set on improvements
    bests = [e.best for e in evaluations]
    assert all(b2 <= b1 for b1, b2 in itertools.pairwise(bests))
    assert all((e.parameters is not None) == e.improved for e in evaluations)
    assert events[-1].value == pytest.approx(res.objective, rel=1e-9)
    assert events[-1].best == pytest.approx(res.objective, rel=1e-6)
    assert events[-1].parameters == res.best_pars
    assert events[-1].start is None


def test_least_squares_values_are_objectives() -> None:
    events: list[FitEvent] = []
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=NLL,
        callback=events.append,
    ).unwrap_or_err()
    # residual vectors are converted into the objective, Σr² / 2
    last = [e for e in events if e.kind == "evaluation"][-1]
    assert last.value == pytest.approx(res.objective, rel=1e-6)


def test_non_squares_loss_values_are_nan_for_residual_vectors() -> None:
    events: list[FitEvent] = []
    fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=fit.LocalScipyMinimizer(),
        loss_fn=fit.mae,
        callback=events.append,
    ).unwrap_or_err()
    # only scalar evaluations, which are all known
    assert all(math.isfinite(e.value) for e in events if e.kind == "evaluation")


def test_log_transform_reports_fit_values() -> None:
    events: list[FitEvent] = []
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=fit.LocalScipyMinimizer(),
        loss_fn=NLL,
        transform="log",
        callback=events.append,
    ).unwrap_or_err()
    first = next(e for e in events if e.improved)
    assert first.parameters == pytest.approx(P0)
    assert events[-1].parameters == res.best_pars


def test_penalised_least_squares_values() -> None:
    events: list[FitEvent] = []
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=NLL,
        prior={"k1": Normal(0.5, 0.1)},
        callback=events.append,
    ).unwrap_or_err()
    last = [e for e in events if e.kind == "evaluation"][-1]
    assert last.value == pytest.approx(res.objective, rel=1e-6)


def test_failed_fit_event() -> None:
    events: list[FitEvent] = []
    res = fit.time_course(
        _model(),
        p0={"k1": 2.0, "k2": 1.5},
        data=_data(),
        minimizer=_FailAbove(1.0),
        loss_fn=NLL,
        callback=events.append,
    )
    assert isinstance(res.value, FitFailure)
    assert [e.kind for e in events] == ["failed"]
    assert events[0].message == "too large"


def test_group_keeps_failures_by_start() -> None:
    p0 = pd.DataFrame(
        {"k1": [0.3, 2.0, 0.7], "k2": [1.0, 1.0, 3.0]}, index=["a", "b", "c"]
    )
    events: list[FitEvent] = []
    res = fit.group_time_course(
        _model(),
        p0=p0,
        data=_data(),
        minimizer=_FailAbove(1.0),
        loss_fn=NLL,
        parallel=False,
        callback=events.append,
    )
    assert list(res.fits) == ["a", "c"]
    assert res.failures == {"b": "too large"}
    assert res.n_failed == 1
    assert list(res.get_losses().index) == ["a", "c"]
    assert list(res.get_objectives().index) == ["a", "c"]
    assert res.get_best_fit() in res.fits.values()
    assert {e.start for e in events} == {"a", "b", "c"}
    ends = {e.start: e.kind for e in events if e.kind != "evaluation"}
    assert ends == {"a": "finished", "b": "failed", "c": "finished"}


def test_group_all_failed_raises() -> None:
    res = fit.group_time_course(
        _model(),
        p0=pd.DataFrame({"k1": [2.0, 3.0], "k2": [1.0, 1.0]}),
        data=_data(),
        minimizer=_FailAbove(1.0),
        loss_fn=NLL,
        parallel=False,
    )
    assert res.fits == {}
    with pytest.raises(ValueError, match="All 2 starts failed"):
        res.get_best_fit()


def test_group_jsonl_from_parallel_workers(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    p0 = pd.DataFrame({"k1": [0.3, 0.6, 0.9], "k2": [1.0, 2.0, 3.0]})
    res = fit.group_time_course(
        _model(),
        p0=p0,
        data=_data(),
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=NLL,
        callback=JsonlCallback(path, fields={"step": "fit"}),
    )
    lines = [json.loads(i) for i in path.read_text().splitlines()]
    assert all(i["step"] == "fit" for i in lines)
    finished = {i["start"] for i in lines if i["kind"] == "finished"}
    assert finished == {"0", "1", "2"}
    assert len(res.fits) == 3


def test_jsonl_throttles_plain_evaluations(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    cb = JsonlCallback(path, interval=3600.0)
    cb(FitEvent(kind="evaluation", value=1.0, best=1.0, improved=True, time=0.0))
    cb(FitEvent(kind="evaluation", value=2.0, best=1.0, time=1.0))
    cb(FitEvent(kind="evaluation", value=0.5, best=0.5, improved=True, time=2.0))
    cb(FitEvent(kind="finished", value=0.5, best=0.5, time=3.0))
    kinds = [json.loads(i)["value"] for i in path.read_text().splitlines()]
    assert kinds == [1.0, 0.5, 0.5]


def test_event_to_dict_is_json() -> None:
    event = FitEvent(kind="failed", parameters={"k": math.nan})
    out = event.to_dict()
    assert out["value"] is None
    assert out["best"] is None
    assert out["parameters"] == {"k": None}
    json.dumps(out)


def test_joint_fit_events() -> None:
    data = _data()
    events: list[FitEvent] = []
    res = fit.joint_time_course(
        [fit.FitSettings(_model(), data), fit.FitSettings(_model(), data)],
        p0=P0,
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=NLL,
        parallel=False,
        callback=events.append,
    ).unwrap_or_err()
    assert events[-1].kind == "finished"
    assert events[-1].value == pytest.approx(res.objective)


def test_no_callback_leaves_residual_unwrapped() -> None:
    from mxlpy.fit.callbacks import tracked  # noqa: PLC0415

    def residual(updates: dict[str, float]) -> float:  # noqa: ARG001
        return 0.0

    assert tracked(residual, None) is residual


def test_group_rejects_duplicate_starts() -> None:
    with pytest.raises(ValueError, match="unique"):
        fit.group_time_course(
            _model(),
            p0=pd.DataFrame({"k1": [0.3, 0.6], "k2": [1.0, 2.0]}, index=["a", "a"]),
            data=_data(),
            minimizer=fit.LeastSquaresMinimizer(),
            loss_fn=NLL,
            parallel=False,
        )


@pytest.mark.parametrize(
    "minimizer", [fit.LeastSquaresMinimizer(), fit.LocalScipyMinimizer()]
)
def test_fit_to_derived_values(minimizer: fit.LeastSquaresMinimizer) -> None:
    model = _model().add_derived("total", fns.add, args=["x", "y"])
    data = (
        Simulator(model)
        .simulate_time_course(np.linspace(0.2, 6.0, 15))
        .get_result()
        .unwrap_or_err()
        .get_args()
        .loc[:, ["total", "y"]]
    )
    res = fit.time_course(
        model,
        p0=P0,
        data=data,
        minimizer=minimizer,
        loss_fn=NLL,
    ).unwrap_or_err()
    assert res.best_pars == pytest.approx({"k1": 0.5, "k2": 2.0}, rel=1e-3)
