# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns, lab
from mxlpy.fit import diagnostics
from mxlpy.lab.runtime import Estimate, RuntimeModel, fit_evaluations
from mxlpy.lab.store import RunStore, to_json

T = np.linspace(0.25, 8, 20)
SIGMA = 0.01
FAST: dict[str, Any] = {
    "n_samples": 4,
    "n_starts": 3,
    "n_pull_fits": 2,
    "prediction_times": 3,
    "parallel": False,
    "max_hidden": 1,
    "structure_search": False,
}


def _decay(k1: float = 0.5) -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k1": k1, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


def _data(model: KineticModelBuilder, seed: int, x0: float = 2.0) -> pd.DataFrame:
    model = model.update_variable("x", x0)
    res = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    d = res.variables.loc[T, ["x", "y"]]
    return d + np.random.default_rng(seed).normal(0, SIGMA, d.shape)


###############################################################################
# Statistics, runtime, store
###############################################################################


def test_estimate_sigma_from_replicates_and_differences() -> None:
    rng = np.random.default_rng(0)
    t = np.repeat(np.linspace(0, 10, 30), 3)
    clean = pd.DataFrame({"a": np.sin(t), "b": t})
    noisy = clean + rng.normal(0, [0.05, 0.2], clean.shape)
    noisy.index = t
    replicates = diagnostics.estimate_sigma(noisy).to_dict()
    assert replicates["a"] == pytest.approx(0.05, rel=0.25)
    assert replicates["b"] == pytest.approx(0.2, rel=0.25)
    # without replicates: pseudo-residuals, accurate for densely sampled data
    t_dense = np.linspace(0, 10, 200)
    dense = pd.DataFrame(
        {"a": np.sin(t_dense) + rng.normal(0, 0.05, t_dense.size)}, index=t_dense
    )
    assert diagnostics.estimate_sigma(dense).to_dict()["a"] == pytest.approx(
        0.05, rel=0.2
    )
    # exactly smooth data: a fraction of the RMS instead of zero
    linear = pd.DataFrame({"a": np.linspace(1, 2, 10)}, index=np.arange(10.0))
    assert diagnostics.estimate_sigma(linear).to_dict()["a"] > 0


def test_residual_tests() -> None:
    rng = np.random.default_rng(1)
    white = rng.normal(size=200)
    structured = np.sin(np.linspace(0, 6, 200)) + 0.1 * white
    assert diagnostics.runs_test(white) > 0.01
    assert diagnostics.runs_test(structured) < 1e-6
    assert diagnostics.ljung_box(white) > 0.01
    assert diagnostics.ljung_box(structured) < 1e-6
    check = diagnostics.check_residuals(
        {"d": pd.DataFrame({"w": white, "s": structured})}, significance=0.05
    )
    assert not check.passed
    assert list(check.flagged.index) == [("d", "s")]
    gof = diagnostics.chi2_test(nll=50.0, n_data=100, n_parameters=2)
    assert gof.chi2 == 100.0
    assert gof.dof == 98
    assert gof.passed(0.05)


def test_runtime_model() -> None:
    runtime = RuntimeModel()
    runtime.seed(simulation=0.01, sensitivity=0.05, n_reference=4)
    typical, high = runtime.per_evaluation(4)
    assert typical == pytest.approx(0.02 + 0.04)
    assert high > typical
    est = runtime.estimate(fit_evaluations(4), 4)
    assert est.low < est.typical < est.high
    for _ in range(10):
        runtime.observe(0.2, 4)
    assert runtime.per_evaluation(4)[0] == pytest.approx(0.2)
    # other parameter counts are scaled by how far off the seed was
    assert runtime.per_evaluation(2)[0] > runtime._seeded(2)
    restored = RuntimeModel.from_dict(json.loads(json.dumps(runtime.to_dict())))
    assert restored.per_evaluation(4) == runtime.per_evaluation(4)
    total = Estimate(1, 2, 3) + Estimate(1, 1, 1)
    assert (total.low, total.typical, total.high) == (2, 3, 4)
    assert "s" in str(total)


def test_store(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.write_decision("fit", {"values": {"a": 1}, "source": "user"})
    assert store.decisions() == {"fit": {"values": {"a": 1}, "source": "user"}}
    store.drop_decision("fit")
    assert store.decisions() == {}
    store.append_event({"kind": "x"})
    with store.events_file.open("a") as f:
        f.write('{"kind": "partial"')  # a line still being written
    events, offset = store.read_events()
    assert [e["kind"] for e in events] == ["x"]
    again, offset2 = store.read_events(offset)
    assert again == []
    assert offset2 == offset
    assert to_json({"a": math.nan, "b": np.float64(1.5), "c": pd.Series({"x": 1})}) == {
        "a": None,
        "b": 1.5,
        "c": {"x": 1},
    }


def test_parse_duration() -> None:
    from mxlpy.lab._core import parse_duration  # noqa: PLC0415

    assert parse_duration("2h") == 7200
    assert parse_duration("30min") == 1800
    assert parse_duration("90 s") == 90
    assert parse_duration(12) == 12
    assert parse_duration(None) is None
    with pytest.raises(ValueError, match="Can't read"):
        parse_duration("soon")


###############################################################################
# The lab
###############################################################################


@pytest.fixture(scope="module")
def two_sets() -> dict[str, fit.Condition | pd.DataFrame]:
    scaled = _data(_decay(), 2, x0=3.0).rename(columns={"y": "y_au"})
    scaled["y_au"] *= 5
    return {"a": _data(_decay(), 1), "b": fit.Condition(scaled, y0={"x": 3.0})}


def test_lab_runs_to_the_fit(tmp_path: Path, two_sets: dict) -> None:
    events: list[lab.LabEvent] = []
    run = lab.FittingLab(
        _decay(),
        two_sets,
        tmp_path / "run",
        observables={"y_au": lab.Observable("y", scale=True)},
        **FAST,
    )
    run.on_event(events.append)
    setup = run.step0_setup()
    assert setup.summary["targets"] == {"x": "x", "y": "y", "y_au": "y"}
    # y and y_au are different columns, x is pooled over both data sets
    assert setup.summary["sigma_source"] == {
        "a": "estimated (pooled)",
        "b": "estimated (pooled)",
    }
    assert setup.summary["sigma"]["a"]["x"] == setup.summary["sigma"]["b"]["x"]
    assert "scale_y_au" in setup.summary["candidates"]
    # the scale starts at the data's ratio
    current = setup.artifacts["current"]
    assert current.estimates["scale_y_au"] == pytest.approx(5.0, rel=0.1)

    params = run.step1_select_parameters()
    assert set(params.proposal["parameters"]) == {"k1", "k2", "scale_y_au"}
    assert params.summary["rank"] == 3

    res = run.step2_fit()
    assert res.passed
    assert res.proposal["next"] == "identifiability"
    assert res.summary["estimates"]["scale_y_au"] == pytest.approx(5.0, rel=0.05)
    assert run.decisions()["parameters"]["source"] == "implicit"
    assert {e.kind for e in events} >= {"started", "progress", "finished"}
    assert all(e.step in {"setup", "parameters", "fit"} for e in events)

    # the run directory holds everything
    store = run.store
    assert json.loads(store.config_file.read_text())["n_starts"] == 3
    assert (store.step_dir("fit") / "result.json").exists()
    lines = store.events_file.read_text().splitlines()
    assert any(json.loads(i).get("step") == "fit" for i in lines)

    # resuming reuses the results
    resumed = lab.FittingLab.resume(tmp_path / "run")
    assert resumed.is_current("fit")
    assert resumed.step2_fit().summary == res.summary

    # a different decision makes the later steps outdated
    resumed.decide("parameters", parameters=["k1", "scale_y_au"])
    assert resumed.is_current("parameters")
    assert not resumed.is_current("fit")
    refit = resumed.step2_fit()
    assert set(refit.summary["estimates"]) == {"k1", "scale_y_au"}
    with pytest.raises(ValueError, match="decides"):
        resumed.decide("parameters", nope=1)


def test_run_stops_at_open_gates(tmp_path: Path) -> None:
    run = lab.FittingLab(_decay(), _data(_decay(), 3), tmp_path / "run", **FAST)
    first = run.run()
    assert first is not None
    assert first.step == "setup"
    second = run.run()
    assert second is not None
    assert second.step == "setup"  # still undecided
    run.decide("setup")
    assert run.run().step == "parameters"  # type: ignore[union-attr]
    assert "gate: user" in repr(run)


def test_inputs_are_checked(tmp_path: Path) -> None:
    data = _data(_decay(), 4).rename(columns={"y": "unknown"})
    run = lab.FittingLab(_decay(), data, tmp_path / "a", **FAST)
    with pytest.raises(ValueError, match="no model output"):
        run.step0_setup()
    run = lab.FittingLab(_decay(), _data(_decay(), 4).iloc[-1], tmp_path / "b", **FAST)
    with pytest.raises(TypeError, match="time courses"):
        run.step0_setup()
    with pytest.raises(KeyError, match="unknown data sets"):
        lab.FittingLab(_decay(), _data(_decay(), 4), tmp_path / "c", sigma={"x": 1.0})
    with pytest.raises(FileNotFoundError):
        lab.FittingLab.resume(tmp_path / "missing")


def test_identifiability_proposes_fixing(tmp_path: Path) -> None:
    # Only the product k1 * k3 enters the dynamics
    model = (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k3": 2.0})
        .add_variables({"x": 2.0})
        .add_reaction(
            "v", fns.mass_action_2s, args=["x", "k1", "k3"], stoichiometry={"x": -1}
        )
    )
    res = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    data = res.variables.loc[T, ["x"]] + np.random.default_rng(5).normal(
        0, SIGMA, (len(T), 1)
    )
    run = lab.FittingLab(model, data, tmp_path / "run", **FAST)
    run.step1_select_parameters()
    # step 1 drops one of the collinear pair; force both to test step 4
    run.decide("parameters", parameters=["k1", "k3"])
    ident = run.step4_identifiability()
    assert not ident.passed
    assert len(ident.proposal["fix"]) == 1
    assert ident.proposal["refit"]
    assert any("combination" in m for m in ident.messages)
    result = run.step5_uncertainty()
    current = run._state().current("uncertainty")
    assert set(current.fixed) == set(ident.proposal["fix"])
    assert len(current.estimates) == 1
    assert len(result.summary["bands"]) == 1


def test_collinear_parameters_are_dropped(tmp_path: Path) -> None:
    model = (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k3": 2.0})
        .add_variables({"x": 2.0})
        .add_reaction(
            "v", fns.mass_action_2s, args=["x", "k1", "k3"], stoichiometry={"x": -1}
        )
    )
    res = Simulator(model).simulate_time_course(T).get_result().unwrap_or_err()
    run = lab.FittingLab(model, res.variables.loc[T, ["x"]], tmp_path / "run", **FAST)
    params = run.step1_select_parameters()
    assert len(params.proposal["parameters"]) == 1
    statuses = {v["status"] for v in params.summary["parameters"].values()}
    assert any(s.startswith("collinear with") for s in statuses)


def test_condition_specific_parameters(tmp_path: Path) -> None:
    data = {"wt": _data(_decay(0.5), 1), "mut": _data(_decay(1.5), 2)}
    run = lab.FittingLab(_decay(), data, tmp_path / "run", **FAST)
    res = run.step2_fit()
    assert res.proposal["local"] == {"mut": {"k1": "k1__mut"}}
    assert res.passed
    current = run._state().current("identifiability")
    assert current.estimates["k1__mut"] == pytest.approx(1.5, rel=0.05)
    assert current.estimates["k1"] == pytest.approx(0.5, rel=0.05)
    # declining the structure keeps the parameters shared
    run.decide("fit", local={})
    current = run._state().current("identifiability")
    assert "k1__mut" not in current.estimates


def test_diagnose_missing_delay(tmp_path: Path) -> None:
    truth = (
        KineticModelBuilder()
        .add_parameters({"k": 1.0, "kx": 0.7})
        .add_variables({"S": 1.0, "X": 0.0, "P": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["S", "k"], stoichiometry={"S": -1, "X": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["X", "kx"], stoichiometry={"X": -1, "P": 1}
        )
    )
    mech = (
        KineticModelBuilder()
        .add_parameters({"k": 1.0})
        .add_variables({"S": 1.0, "P": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["S", "k"], stoichiometry={"S": -1, "P": 1}
        )
    )
    res = Simulator(truth).simulate_time_course(T).get_result().unwrap_or_err()
    data = res.variables.loc[T, ["S", "P"]]
    data += np.random.default_rng(0).normal(0, SIGMA, data.shape)
    settings: dict[str, Any] = {**FAST, "structure_search": True}
    run = lab.FittingLab(mech, data, tmp_path / "run", **settings)
    assert not run.step2_fit().passed
    diag = run.step3_diagnose()
    assert diag.summary["chosen_hidden"] == 1
    assert diag.proposal["model"] == "delay[P]"
    current = run._state().current("identifiability")
    assert "kdelay_P" in current.estimates
    assert "P_delay1" in current.model.get_variable_names()


def test_report(tmp_path: Path) -> None:
    run = lab.FittingLab(_decay(), _data(_decay(), 6), tmp_path / "run", **FAST)
    run.step2_fit()
    path = run.report()
    page = path.read_text()
    assert "2 · Fit" in page
    assert "data:image/png;base64" in page
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert set(report["steps"]) >= {"setup", "parameters", "fit"}


###############################################################################
# Review regressions
###############################################################################


def test_accepted_proposals_expire_with_their_result(tmp_path: Path) -> None:
    data = {"wt": _data(_decay(0.5), 1), "mut": _data(_decay(1.5), 2)}
    run = lab.FittingLab(_decay(), data, tmp_path / "run", **FAST)
    run._prepare("diagnose", "implicit")
    assert run.decision("fit") == run["fit"].proposal
    run.decide("parameters", parameters=["k2"])
    refit = run.step2_fit()
    # the implicit decision was made on the old result, the new one is open
    assert run.decision("fit") is None
    run.step4_identifiability()
    assert run.decision("fit") == refit.proposal
    current = run._state().current("identifiability")
    assert set(current.estimates) <= {"k2", "k2__mut"}
    assert run.decision(1) == {"parameters": ["k2"]}
    assert run[2] is run["step2_fit"]


def test_negative_offsets_and_user_local_and_terms(tmp_path: Path) -> None:
    model = _decay().add_parameter("kleak", 0.0)
    model.add_term(
        "leak",
        fns.mass_action_1s,
        args=["x", "kleak"],
        target="x",
        mechanism="additive",
    )
    base = _data(_decay(), 1)
    shifted = base.rename(columns={"y": "y_bg"})
    shifted["y_bg"] -= 0.5
    shifted["leak"] = 0.0
    data = {
        "a": fit.Condition(base),
        "b": fit.Condition(shifted, local={"k2": "k2_b"}),
    }
    run = lab.FittingLab(
        model,
        data,
        tmp_path / "run",
        parameters=["k1", "k2"],
        observables={"y_bg": lab.Observable("y", offset=True)},
        **FAST,
    )
    setup = run.step0_setup()
    assert "k2_b" in setup.summary["candidates"]
    run.decide("parameters", parameters=["k1", "k2", "k2_b", "offset_y_bg"])
    res = run.step2_fit()
    assert res.summary["estimates"]["offset_y_bg"] == pytest.approx(-0.5, abs=0.05)
    assert res.summary["estimates"]["k2_b"] == pytest.approx(2.0, rel=0.05)


def test_budget_counts_computing_time(tmp_path: Path) -> None:
    run = lab.FittingLab(
        _decay(), _data(_decay(), 7), tmp_path / "run", time_budget=3600, **FAST
    )
    run.step1_select_parameters()
    assert run["parameters"].summary["limits"]["time_budget"] >= 2
    assert run.active_time < 60
