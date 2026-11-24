# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Hidden-state blocks: latent states driven by a small neural network (#229)."""

from __future__ import annotations

import logging
import math
import time
from typing import cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    adjoint_sensitivities,
    fit,
    fns,
    forward_sensitivities,
)
from mxlpy.composition import compose
from mxlpy.integrators import Native, Scipy
from mxlpy.meta import generate_model_code_py
from mxlpy.types import InitialAssignment

T = np.linspace(0, 4, 9)


def _f(x: object) -> float:
    return float(cast(float, x))


def exp_mechanism(base: float, term: float) -> float:
    return base * math.exp(term)


def _decay(s: float, k: float) -> float:
    return -k * s


def _model() -> KineticModelBuilder:
    """S -> P ->."""
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 1.0, "k2": 0.5})
        .add_variables({"S": 1.0, "P": 0.5})
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1, "P": 1},
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["P", "k2"], stoichiometry={"P": -1}
        )
    )


def _ode_model() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameters({"k": 1.0})
        .add_diff_eq("S", 1.0, _decay, args=["S", "k"])
    )


def _augmented(mechanism: object = "relative") -> KineticModelBuilder:
    return _model().add_hidden_states(
        "aug",
        n_hidden=2,
        layers=(3,),
        targets=["S", "v2"],
        mechanism=mechanism,  # type: ignore[arg-type]
        init_scale=0.5,
    )


def _augmented_ode() -> OdeModelBuilder:
    return _ode_model().add_hidden_states(
        "aug", n_hidden=1, layers=(3,), targets=["S"], mechanism="additive"
    )


def _simulate(model: KineticModelBuilder | OdeModelBuilder, **kwargs) -> pd.DataFrame:  # noqa: ANN003
    res = Simulator(model, **kwargs).simulate_time_course(T).get_result()
    return res.unwrap_or_err().variables


def _zero_readout(model: KineticModelBuilder | OdeModelBuilder) -> None:
    block = model.get_raw_hidden_states(as_copy=False)["aug"]
    model.update_addressable(dict.fromkeys(block.readout_names("aug"), 0.0))


###############################################################################
# Structure
###############################################################################


def test_names() -> None:
    model = _augmented()
    assert model.get_variable_names() == ["S", "P", "aug_h0", "aug_h1"]
    assert model.get_hidden_state_names() == ["aug_h0", "aug_h1"]
    assert model.get_parameter_names() == ["k1", "k2"]
    weights = model.get_weight_names()
    # (2 inputs + 2 states) -> 3 -> 2, readout 2 targets x 2 states
    assert len(weights) == 4 * 3 + 3 + 3 * 2 + 2 + 2 * 2
    assert weights[0] == "aug_b0_0"
    assert weights[1] == "aug_w0_0_0"
    assert weights[-1] == "aug_W_1_1"
    assert model.get_addressable_names() == ["k1", "k2", *weights]
    assert set(model.get_addressable_values()) == {"k1", "k2", *weights}
    assert model.get_term_names() == ["aug_dh0", "aug_dh1", "aug_S", "aug_v2"]
    assert "aug_a0_0" not in model.get_args().index
    assert list(model.parameters.data.index) == ["k1", "k2"]
    assert "aug_W_0_0" not in model.get_raw_parameters()


def test_weights_initialisation() -> None:
    block = _augmented().get_raw_hidden_states()["aug"]
    assert np.all(block.readout != 0)
    assert np.all(np.abs(block.readout) <= 0.5 * math.sqrt(6 / 4))
    assert np.all(block.h0 == 0)
    assert any(np.any(b != 0) for b in block.biases)
    # same seed, same weights
    again = _augmented().get_raw_hidden_states()["aug"]
    np.testing.assert_array_equal(block.readout, again.readout)
    # h moves away from 0
    sim = _simulate(_augmented())
    assert np.abs(sim["aug_h0"].iloc[-1]) > 1e-3


@pytest.mark.parametrize("mechanism", ["additive", "relative", exp_mechanism])
def test_zero_readout_is_bit_identical(mechanism: object) -> None:
    """The derivatives are bit-identical for any state.

    Simulations agree to the solver's tolerance only: the hidden states take
    part in its step-size control.
    """
    model = _augmented(mechanism)
    _zero_readout(model)
    base = _model()
    rng = np.random.default_rng(1)
    for _ in range(5):
        s, p, h0, h1 = rng.uniform(0.1, 2.0, size=4)
        rhs = model.get_right_hand_side({"S": s, "P": p, "aug_h0": h0, "aug_h1": h1})
        expected = base.get_right_hand_side({"S": s, "P": p})
        if mechanism is exp_mechanism:
            np.testing.assert_allclose(
                np.asarray(rhs[["S", "P"]], dtype=float), expected.to_numpy()
            )
        else:
            assert np.array_equal(
                np.asarray(rhs[["S", "P"]], dtype=float), expected.to_numpy()
            )
    pd.testing.assert_frame_equal(
        _simulate(model).loc[:, ["S", "P"]], _simulate(base), rtol=1e-5, atol=1e-7
    )


def test_zero_readout_is_bit_identical_ode() -> None:
    model = _augmented_ode()
    _zero_readout(model)
    rhs = model.get_right_hand_side({"S": 0.7, "aug_h0": 0.3})
    expected = _ode_model().get_right_hand_side({"S": 0.7})
    assert np.array_equal(np.asarray(rhs[["S"]], dtype=float), expected.to_numpy())


@pytest.mark.parametrize("model_fn", [_augmented, _augmented_ode])
def test_remove_restores_the_model(model_fn: object) -> None:
    model = model_fn()  # type: ignore[operator]
    model.remove_hidden_states("aug")
    base = _model() if isinstance(model, KineticModelBuilder) else _ode_model()
    assert model.get_variable_names() == base.get_variable_names()
    assert model.get_weight_names() == []
    assert not any(k.startswith("aug") for k in model.ids)
    assert np.array_equal(_simulate(model).to_numpy(), _simulate(base).to_numpy())


def test_rhs_matches_hand_calculation() -> None:
    model = _augmented("additive")
    block = model.get_raw_hidden_states()["aug"]
    model.update_variables({"aug_h0": 0.3, "aug_h1": -0.2})
    h = np.array([0.3, -0.2])
    x = np.array([1.0, 0.5, *h])
    a = np.tanh(block.weights[0] @ x + block.biases[0])
    dh = block.weights[1] @ a + block.biases[1]
    w_h = block.readout @ h
    rhs = model.get_right_hand_side()
    np.testing.assert_allclose(np.asarray(rhs[["aug_h0", "aug_h1"]], dtype=float), dh)
    v1, v2 = 1.0, 0.5 * 0.5 + w_h[1]
    np.testing.assert_allclose(rhs["S"], -v1 + w_h[0])
    np.testing.assert_allclose(rhs["P"], v1 - v2)


def test_update_addressable_and_initial_values() -> None:
    model = _augmented()
    model.update_addressable({"k1": 2.0, "aug_W_0_1": 0.25})
    assert model.get_parameter_values()["k1"] == 2.0
    block = model.get_raw_hidden_states(as_copy=False)["aug"]
    assert block.readout[0, 1] == 0.25
    assert model.get_addressable_values()["aug_W_0_1"] == 0.25
    model.update_variable("aug_h1", 0.7)
    assert block.h0[1] == 0.7
    assert model.get_initial_conditions()["aug_h1"] == 0.7
    with pytest.raises(KeyError):
        model.update_parameters({"aug_W_0_1": 1.0})
    model.update_hidden_states("aug", input_scales={"S": 10.0})
    assert block.input_scales[0] == 10.0


def test_validation() -> None:
    model = _augmented()
    with pytest.raises(ValueError, match="hidden-state block"):
        model.rename("aug_h0", "h")
    with pytest.raises(ValueError, match="target of the hidden-state blocks"):
        model.remove_reaction("v2")
    with pytest.raises(NameError):
        model.add_hidden_states("aug", n_hidden=1, targets=["S"], mechanism="additive")
    # a failed add leaves the model untouched
    model.add_parameter("b_h0", 1.0)
    ids = dict(model.ids)
    with pytest.raises(NameError):
        model.add_hidden_states(
            "b", n_hidden=1, targets=["S"], mechanism="additive", inputs=["S"]
        )
    assert model.ids == ids
    assert "b" not in model.get_raw_hidden_states()
    with pytest.raises(KeyError, match="unknown inputs"):
        model.add_hidden_states(
            "c", n_hidden=1, targets=["S"], mechanism="additive", inputs=["Q"]
        )
    with pytest.raises(KeyError, match="neither a variable nor a reaction"):
        model.add_hidden_states("c", n_hidden=1, targets=["Q"], mechanism="additive")
    assert "c" not in model.ids


def test_warns_for_large_networks(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _model().add_hidden_states(
            "big", n_hidden=4, layers=(16, 16), targets=["S"], mechanism="additive"
        )
    assert "weights" in caplog.text


###############################################################################
# Lowering
###############################################################################


@pytest.mark.parametrize("model_fn", [_augmented, _augmented_ode])
def test_native_matches_python(model_fn: object) -> None:
    model = model_fn()  # type: ignore[operator]
    pd.testing.assert_frame_equal(
        _simulate(model, integrator=Native),
        _simulate(model, integrator=Scipy),
        rtol=1e-5,
        atol=1e-7,
    )


def test_native_recompiles_nothing_for_new_weights() -> None:
    model = _augmented()
    first = _simulate(model, integrator=Native)
    model.update_addressable({"aug_W_0_0": 3.0})
    second = _simulate(model, integrator=Native)
    assert not np.allclose(first.to_numpy(), second.to_numpy())
    pd.testing.assert_frame_equal(
        second, _simulate(model, integrator=Scipy), rtol=1e-5, atol=1e-7
    )


def test_equations_grow_linearly_with_depth() -> None:
    def code_per_weight(depth: int) -> float:
        model = _model().add_hidden_states(
            "deep",
            n_hidden=2,
            layers=(6,) * depth,
            targets=["S"],
            mechanism="additive",
        )
        code = generate_model_code_py(model).model
        return len(code) / model.get_raw_hidden_states()["deep"].n_weights

    start = time.perf_counter()
    shallow, deep = code_per_weight(1), code_per_weight(6)
    assert time.perf_counter() - start < 60
    # every unit is computed once and referred to by name
    assert deep < 1.5 * shallow


@pytest.mark.parametrize("model_fn", [_augmented, _augmented_ode])
def test_gradients_wrt_weights(model_fn: object) -> None:
    model = model_fn()  # type: ignore[operator]
    wrt = ["aug_W_0_0", "aug_w0_1_0", "aug_b1_0"]
    exact = forward_sensitivities.time_course(model, T, wrt, method="forward")
    fd = forward_sensitivities.time_course(model, T, wrt, method="finite_difference")
    e, f = exact.unwrap_or_err(), fd.unwrap_or_err()
    for k in wrt:
        np.testing.assert_allclose(
            e.derivatives[k].to_numpy(),
            f.derivatives[k].to_numpy(),
            rtol=5e-3,
            atol=1e-5,
        )

    names = model.get_variable_names()
    w = pd.DataFrame(np.ones((len(T), len(names))), index=T, columns=names)
    adj = adjoint_sensitivities.time_course(model, w, wrt, method="adjoint")
    fwd = adjoint_sensitivities.time_course(model, w, wrt, method="forward")
    np.testing.assert_allclose(
        adj.unwrap_or_err().gradient, fwd.unwrap_or_err().gradient, rtol=1e-4
    )


def test_contributions_in_results() -> None:
    res = Simulator(_augmented()).simulate_time_course(T).get_result()
    contributions = cast(pd.DataFrame, res.unwrap_or_err().get_term_contributions())
    assert list(contributions.columns) == ["aug_dh0", "aug_dh1", "aug_S", "aug_v2"]


def test_h0_initial_assignment() -> None:
    model = _augmented().add_parameter("h_start", 0.4)
    model.update_variable(
        "aug_h1", InitialAssignment(fn=fns.constant, args=["h_start"])
    )
    assert model.get_initial_conditions()["aug_h1"] == 0.4
    pd.testing.assert_frame_equal(
        _simulate(model, integrator=Native),
        _simulate(model, integrator=Scipy),
        rtol=1e-5,
        atol=1e-7,
    )
    model.update_variable("aug_h1", 0.1)
    assert model.get_initial_conditions()["aug_h1"] == 0.1


def test_compose_copies_blocks() -> None:
    other = KineticModelBuilder().add_parameters({"kx": 1.0})
    composed = compose(_augmented(), other)
    assert composed.get_weight_names() == _augmented().get_weight_names()
    pd.testing.assert_series_equal(
        composed.get_right_hand_side(), _augmented().get_right_hand_side()
    )
    # a block overridden by a later model is replaced as a whole
    with pytest.warns(UserWarning, match="overridden"):
        again = compose(_augmented(), _augmented(), raise_on_conflict=False)
    assert again.get_weight_names() == _augmented().get_weight_names()


def test_fit_weights() -> None:
    """Fitting the readout recovers the data of a model with known weights."""
    truth = _augmented("additive")
    truth.update_addressable({"aug_W_0_0": 0.3, "aug_W_1_1": -0.2})
    data = _simulate(truth).loc[:, ["S", "P"]]
    model = _augmented("additive")
    readout = model.get_raw_hidden_states()["aug"].readout_names("aug")
    p0 = {k: v for k, v in model.get_addressable_values().items() if k in readout}
    res = fit.time_course(
        model,
        p0=p0,
        data=data,
        minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
        loss_fn=fit.GaussianNll(sigma=0.01),
    ).unwrap_or_err()
    assert res.loss < 1e-3
    assert res.best_pars["aug_W_0_0"] == pytest.approx(0.3, rel=1e-2)


###############################################################################
# Review fixes
###############################################################################


def test_rename_updates_blocks() -> None:
    model = _augmented().rename("S", "S2").rename("k1", "kk").rename("v2", "w2")
    block = model.get_raw_hidden_states()["aug"]
    assert block.inputs == ["S2", "P"]
    assert block.targets == ["S2", "w2"]
    rhs = model.get_right_hand_side()
    base = _augmented().get_right_hand_side()
    assert _f(rhs["S2"]) == _f(base["S"])
    assert _f(rhs["P"]) == _f(base["P"])


def test_inputs_and_states_are_protected() -> None:
    model = _augmented()
    with pytest.raises(ValueError, match="read by the hidden-state blocks"):
        model.remove_variable("P")
    with pytest.raises(ValueError, match="remove the block instead"):
        model.remove_variable("aug_h0")
    model = _model().add_hidden_states(
        "aug", n_hidden=1, targets=["S"], mechanism="additive", inputs=["S", "k2"]
    )
    with pytest.raises(ValueError, match="read by the hidden-state blocks"):
        model.remove_parameter("k2")
    with pytest.raises(ValueError, match="positive"):
        _model().add_hidden_states(
            "b", n_hidden=1, targets=["S"], mechanism="additive", input_scales={"S": 0}
        )
    with pytest.raises(KeyError, match="Unknown parameters or weights"):
        _augmented().update_addressable({"nope": 1.0})


def test_pickle_after_evaluation() -> None:
    import pickle  # noqa: PLC0415

    model = _augmented()
    model.get_right_hand_side()
    model._cache = None
    again = pickle.loads(pickle.dumps(model))
    pd.testing.assert_series_equal(
        again.get_right_hand_side(), _augmented().get_right_hand_side()
    )


def test_h0_update_replaces_assignment() -> None:
    model = _augmented().add_parameter("q", 2.0)
    model.update_variable("aug_h0", InitialAssignment(fn=fns.constant, args=["q"]))
    model.update_hidden_states("aug", h0=[0.7, 0.1])
    assert model.get_initial_conditions()["aug_h0"] == 0.7


def test_ode_args_include_hidden_states() -> None:
    assert "aug_h0" in _augmented_ode().get_args().index


def test_log_transform_leaves_weights_linear() -> None:
    truth = _augmented("additive")
    truth.update_addressable({"aug_W_0_0": -0.3})
    data = _simulate(truth).loc[:, ["S", "P"]]
    model = _augmented("additive")
    res = fit.time_course(
        model,
        p0={"k1": 1.0, "aug_W_0_0": 0.1},
        data=data,
        minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
        loss_fn=fit.GaussianNll(sigma=0.01),
        transform="log",
    ).unwrap_or_err()
    assert res.best_pars["aug_W_0_0"] == pytest.approx(-0.3, rel=1e-2)


def test_h0_assignment_gradients() -> None:
    model = _augmented().add_parameter("q", 0.5)
    model.update_variable("aug_h0", InitialAssignment(fn=fns.constant, args=["q"]))
    exact = forward_sensitivities.time_course(model, T, ["q"], method="forward")
    fd = forward_sensitivities.time_course(model, T, ["q"], method="finite_difference")
    np.testing.assert_allclose(
        exact.unwrap_or_err().derivatives["q"].to_numpy(),
        fd.unwrap_or_err().derivatives["q"].to_numpy(),
        rtol=5e-3,
        atol=1e-5,
    )
