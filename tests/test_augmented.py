# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Training hidden-state blocks with `fit.augmented_time_course` (#229)."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns, plot

T = np.linspace(0, 8, 17)
SIGMA = 0.01


def _truth() -> KineticModelBuilder:
    """S -> X -> P: the mechanistic model below misses the delay through X."""
    return (
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


def _mechanistic() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k": 1.0})
        .add_variables({"S": 1.0, "P": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["S", "k"], stoichiometry={"S": -1, "P": 1}
        )
    )


def _data() -> pd.DataFrame:
    res = Simulator(_truth()).simulate_time_course(T).get_result().unwrap_or_err()
    return res.variables.loc[:, ["S", "P"]]


def _mechanistic_fit() -> fit.Fit:
    return fit.time_course(
        _mechanistic(),
        p0={"k": 1.0},
        data=_data(),
        minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
        loss_fn=fit.GaussianNll(sigma=SIGMA),
    ).unwrap_or_err()


@pytest.mark.parametrize("mechanism", ["additive", "relative"])
def test_recovers_missing_delay(mechanism: Literal["additive", "relative"]) -> None:
    mechanistic = _mechanistic_fit()
    # P's production is delayed, while S decays as modelled
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(4,), targets=["P"], mechanism=mechanism
    )
    res = fit.augmented_time_course(
        model, data=_data(), p0=mechanistic.best_pars, sigma=SIGMA
    ).unwrap_or_err()
    assert set(res.stages) == {"weights", "joint"}
    # the hidden state explains most of what the mechanism alone can't
    assert res.loss < 0.2 * mechanistic.loss
    assert list(res.correction.columns) == ["aug_P"]
    assert res.best_pars["k"] > mechanistic.best_pars["k"]  # towards the true 1
    assert list(res.correction.index.names) == ["data", "time"]
    # the fitted model reproduces the data
    sim = Simulator(res.model).simulate_time_course(T).get_result().unwrap_or_err()
    error = (sim.variables.loc[:, ["S", "P"]] - _data()).abs().to_numpy().max()
    assert error < 20 * SIGMA


def test_scales_and_options() -> None:
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(3,), targets=["P"], mechanism="additive"
    )
    data = {"a": _data(), "b": _data().iloc[::2]}
    res = fit.augmented_time_course(
        model,
        data=data,
        p0={"k": 1.0},
        sigma={"a": SIGMA, "b": pd.Series({"S": SIGMA, "P": 2 * SIGMA})},
        joint_refinement=False,
        fit_h0=True,
    ).unwrap_or_err()
    assert set(res.stages) == {"weights"}
    # one h(0) per data set
    assert {"aug_h0_0_a", "aug_h0_0_b"} <= set(res.best_pars)
    assert set(res.models) == set(res.predictions) == set(res.residuals) == {"a", "b"}
    block = res.model.get_raw_hidden_states()["aug"]
    assert block.output_scales[0] == pytest.approx(res.scales["aug_P"])
    assert block.input_scales[0] > 0
    # the given model is unchanged
    assert model.get_raw_hidden_states()["aug"].output_scales[0] == 1.0
    assert set(res.correction.index.get_level_values("data")) == {"a", "b"}


def test_validation() -> None:
    with pytest.raises(ValueError, match="no hidden-state blocks"):
        fit.augmented_time_course(_mechanistic(), data=_data(), p0={"k": 1.0})
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, targets=["P"], mechanism="additive"
    )
    with pytest.raises(KeyError, match="only hold parameters"):
        fit.augmented_time_course(model, data=_data(), p0={"aug_W_0_0": 1.0})
    with pytest.raises(KeyError, match="No protocol"):
        fit.augmented_protocol_time_course(
            model,
            data={"a": _data(), "b": _data()},
            protocol={"a": _data().iloc[:1, :0]},
            p0={"k": 1.0},
        )


def test_protocol_variant_and_plot() -> None:
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(2,), targets=["P"], mechanism="additive"
    )
    protocol = pd.DataFrame({"k": [1.0, 1.0]}, index=pd.Index([4.0, 8.0], name="time"))
    res = fit.augmented_protocol_time_course(
        model,
        data=_data().iloc[::2],
        protocol=protocol,
        p0={"k": 1.0},
        sigma=SIGMA,
        joint_refinement=False,
    ).unwrap_or_err()
    fig, axs = plot.correction(res)
    assert len(cast(list, list(axs))) >= 1
    fig.clf()


def test_review_fixes() -> None:
    from mxlpy.fit._augmented import _scale_blocks  # noqa: PLC0415

    # rescaling keeps the (non-zero) readout
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(3,), targets=["P"], mechanism="relative"
    )
    before = model.get_weight_values()["aug_W_0_0"]
    _scale_blocks(model, ["aug"], {"d": fit.Condition(_data())}, {}, None)
    assert model.get_weight_values()["aug_W_0_0"] == before != 0
    # relative terms aren't scaled by a rate
    assert model.get_raw_hidden_states()["aug"].output_scales[0] == 1.0

    # the loss is the data's alone
    data = _data()
    res = fit.augmented_time_course(
        model,
        data=data,
        p0={"k": 0.6},
        sigma=SIGMA,
        joint_refinement=False,
        correction_penalty=10.0,
    ).unwrap_or_err()
    nll = fit.GaussianNll(sigma=SIGMA)(res.predictions["data"], data)
    assert res.loss == pytest.approx(nll)
    assert res.loss < res.stages["weights"].loss

    # term contributions only appear in combined results on request
    sim = Simulator(res.model).simulate_time_course(T).get_result().unwrap_or_err()
    assert "aug_P" not in sim.get_combined().columns
    assert "aug_P" in sim.get_combined(include_terms=True).columns


def test_conditions_set_initial_values_parameters_and_sigma() -> None:
    truth = _truth().update_variable("S", 2.0)
    res2 = Simulator(truth).simulate_time_course(T).get_result().unwrap_or_err()
    data2 = res2.variables.loc[:, ["S", "P"]].iloc[::2]
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(3,), targets=["P"], mechanism="additive"
    )
    res = fit.augmented_time_course(
        model,
        data={
            "a": fit.Condition(_data(), sigma=SIGMA),
            "b": fit.Condition(data2, y0={"S": 2.0}, sigma=SIGMA),
        },
        p0={"k": 1.0},
        sigma=123.0,  # ignored, every condition has its own
        joint_refinement=False,
        minimizer=fit.FixedNMinimizer(n_steps=3),
    ).unwrap_or_err()
    assert res.models["b"].get_initial_conditions()["S"] == 2.0
    assert res.models["a"].get_initial_conditions()["S"] == 1.0
    # each data set predicted from its own initial value
    assert res.predictions["b"]["S"].iloc[0] == pytest.approx(2.0)
    assert res.predictions["a"]["S"].iloc[0] == pytest.approx(1.0)
    # sigma of the conditions, not the ignored 123: a loss in units of SIGMA
    assert res.loss > 1.0

    with pytest.raises(ValueError, match="set by a condition can't be refined"):
        fit.augmented_time_course(
            model,
            data={"a": fit.Condition(_data(), parameters={"k": 1.0})},
            p0={"k": 1.0},
        )


def test_mixed_protocols_raise() -> None:
    model = _mechanistic().add_hidden_states(
        "aug", n_hidden=1, layers=(3,), targets=["P"], mechanism="additive"
    )
    protocol = pd.DataFrame({"k": [1.0]}, index=pd.Index([8.0], name="time"))
    with pytest.raises(KeyError, match="have a protocol"):
        fit.augmented_time_course(
            model,
            data={"a": fit.Condition(_data(), protocol=protocol), "b": _data()},
            p0={"k": 1.0},
        )
