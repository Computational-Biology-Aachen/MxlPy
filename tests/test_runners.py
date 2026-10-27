# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    SteadyStateModelBuilder,
    fit,
    fns,
    mc,
    mca,
    scan,
)
from mxlpy.runners import EvaluationRunner, KineticRunner, OdeRunner, default_runner
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation
from mxlpy.simulator import Simulator

if TYPE_CHECKING:
    from mxlpy.types import Result


def constant(k: float) -> float:
    return k


def decay(k_in: float, x: float, k: float) -> float:
    return k_in - k * x


def ratio(a: float, b: float) -> float:
    return a / b


def kinetic_model() -> KineticModelBuilder:
    """Steady state x = k_in / k."""
    return (
        KineticModelBuilder()
        .add_parameters({"k_in": 1.0, "k": 0.5})
        .add_variable("x", 0.0)
        .add_reaction("v_in", fn=constant, args=["k_in"], stoichiometry={"x": 1})
        .add_reaction(
            "v_out", fn=fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1}
        )
    )


def ode_model() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameters({"k_in": 1.0, "k": 0.5})
        .add_diff_eq("x", fn=decay, args=["k_in", "x", "k"], initial_value=0.0)
    )


def ss_model() -> SteadyStateModelBuilder:
    return (
        SteadyStateModelBuilder()
        .add_parameters({"a": 1.0, "b": 4.0})
        .add_derived("d", fn=ratio, args=["a", "b"])
    )


class CountingRunner(KineticRunner):
    """Custom runner, the replacement for the removed `worker=` arguments."""

    def steady_state(self, model: KineticModelBuilder) -> Result[Simulation]:
        res = KineticRunner.steady_state(self, model)
        if isinstance(sim := res.value, Simulation):
            sim.raw_variables[-1].loc[:, "x"] = -1.0
        return res


def test_default_runner_per_builder() -> None:
    assert isinstance(default_runner(kinetic_model()), KineticRunner)
    assert isinstance(default_runner(ode_model()), OdeRunner)
    assert isinstance(default_runner(ss_model()), EvaluationRunner)


def test_runners_return_builder_specific_results() -> None:
    assert isinstance(
        KineticRunner().steady_state(kinetic_model()).unwrap_or_err(), Simulation
    )
    assert isinstance(
        OdeRunner().steady_state(ode_model()).unwrap_or_err(), OdeSimulation
    )
    assert isinstance(
        EvaluationRunner().steady_state(ss_model()).unwrap_or_err(), Evaluation
    )


def test_runner_configuration_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}
    original = Simulator.simulate_to_steady_state

    def spy(self: Simulator, **kwargs: Any) -> Simulator:
        received.update(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(Simulator, "simulate_to_steady_state", spy)
    KineticRunner(
        tolerance=1e-3, rel_norm=True, oscillation_detector=None
    ).steady_state(kinetic_model())
    assert received == {
        "tolerance": 1e-3,
        "rel_norm": True,
        "oscillation_detector": None,
    }


def test_runners_are_picklable() -> None:
    for runner in (KineticRunner(tolerance=1e-8), OdeRunner(), EvaluationRunner()):
        assert pickle.loads(pickle.dumps(runner)) == runner


def test_custom_runner_reaches_routines() -> None:
    to_scan = pd.DataFrame({"k": [0.5, 1.0]})
    res = scan.steady_state(
        kinetic_model(), to_scan=to_scan, parallel=False, runner=CountingRunner()
    )
    np.testing.assert_array_equal(res.variables["x"], -1.0)

    res = mc.steady_state(
        kinetic_model(), mc_to_scan=to_scan, parallel=False, runner=CountingRunner()
    )
    np.testing.assert_array_equal(res.variables["x"], -1.0)


def test_failed_evaluation_is_nan_row() -> None:
    res = scan.steady_state(ss_model(), to_scan=pd.DataFrame({"b": [4.0, 0.0]}))
    np.testing.assert_array_equal(res.combined["d"], [0.25, np.nan])


def test_evaluation_accessors() -> None:
    evaluation = EvaluationRunner().steady_state(ss_model()).unwrap_or_err()
    assert list(evaluation.get_combined().columns) == ["d"]
    assert list(evaluation.get_args(include_parameters=True).columns) == ["a", "b", "d"]
    assert Evaluation.default(ss_model()).get_combined().isna().all().all()


def test_routines_dont_mutate_the_callers_model() -> None:
    to_scan = pd.DataFrame({"k": [5.0, 7.0]})
    for model in (kinetic_model(), ode_model()):
        scan.steady_state(model, to_scan=to_scan, y0={"x": 3.0}, parallel=False)
        scan.time_course(
            model, to_scan=to_scan, time_points=np.linspace(0, 1, 3), parallel=False
        )
        assert model.get_parameter_values()["k"] == 0.5
        assert model.get_initial_conditions()["x"] == 0.0

    model = ss_model()
    scan.steady_state(model, to_scan=pd.DataFrame({"a": [5.0, 7.0]}))
    assert model.get_parameter_values()["a"] == 1.0

    model = kinetic_model()
    mca.response_coefficients(model, variables={"x": 3.0}, parallel=False)
    assert model.get_initial_conditions()["x"] == 0.0


def test_fit_with_custom_runner() -> None:
    res = fit.steady_state(
        kinetic_model(),
        p0={"k": 1.0},
        data=pd.Series({"x": 4.0, "v_out": 1.0}),
        minimizer=fit.LocalScipyMinimizer(),
        runner=KineticRunner(tolerance=1e-8),
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(0.25, rel=1e-2)
