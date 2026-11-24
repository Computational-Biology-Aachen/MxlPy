# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Terms: quantities rewriting a variable's derivative or a reaction's rate (#229)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    OdeModelBuilder,
    Simulator,
    adjoint_sensitivities,
    fns,
    forward_sensitivities,
    sbml,
    serialize,
)
from mxlpy.composition import compose
from mxlpy.integrators import Native, Scipy
from mxlpy.meta import (
    generate_model_code_mxlweb,
    generate_model_code_py,
    generate_mxlpy_code,
)
from mxlpy.types import Derived, Mechanism, SerializationError

T = np.linspace(0, 4, 9)


def _f(x: object) -> float:
    return float(cast(float, x))


def exp_mechanism(base: float, term: float) -> float:
    return base * math.exp(term)


def _model() -> KineticModelBuilder:
    """S -> P ->."""
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 1.0, "k2": 0.5, "kc": 0.1, "kd": 0.2})
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


def _with_terms() -> KineticModelBuilder:
    return (
        _model()
        .add_term("t_rxn", fns.constant, args=["kc"], target="v1", mechanism="relative")
        .add_term(
            "t_add",
            fns.mass_action_1s,
            args=["S", "kd"],
            target="P",
            mechanism="additive",
        )
        .add_term("t_rel", fns.constant, args=["kc"], target="P", mechanism="relative")
        .add_term(
            "t_exp", fns.constant, args=["kd"], target="P", mechanism=exp_mechanism
        )
    )


def _decay(s: float, k: float) -> float:
    return -k * s


def _ode_model() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameters({"k": 1.0, "kc": 0.1})
        .add_diff_eq("S", 1.0, _decay, args=["S", "k"])
        .add_term("t1", fns.constant, args=["kc"], target="S", mechanism="relative")
        .add_term(
            "t2", fns.mass_action_1s, args=["S", "kc"], target="S", mechanism="additive"
        )
    )


def _simulate(model: KineticModelBuilder | OdeModelBuilder, **kwargs) -> pd.DataFrame:  # noqa: ANN003
    res = Simulator(model, **kwargs).simulate_time_course(T).get_result()
    return res.unwrap_or_err().variables


###############################################################################
# Composition and contributions
###############################################################################


def test_composition_matches_hand_calculation() -> None:
    model = _with_terms()
    s, p = 1.0, 0.5
    v1 = s * 1.1  # relative reaction term
    v2 = p * 0.5
    dp = v1 - v2
    dp += s * 0.2  # additive
    dp *= 1.1  # relative
    dp *= math.exp(0.2)  # custom
    rhs = model.get_right_hand_side()
    assert _f(rhs["S"]) == pytest.approx(-v1)
    assert _f(rhs["P"]) == pytest.approx(dp)


def test_rhs_is_base_plus_contributions() -> None:
    model = _with_terms()
    contributions = model.get_term_contributions()
    fluxes = model.get_fluxes()
    stoich = model.get_stoichiometries()
    base = stoich @ fluxes
    # the reaction term fans out through v1's stoichiometry
    expected = base + stoich["v1"] * contributions["t_rxn"]
    expected["P"] += contributions[["t_add", "t_rel", "t_exp"]].sum()
    pd.testing.assert_series_equal(
        model.get_right_hand_side(), expected, check_names=False
    )


def test_reaction_term_corrects_all_touched_variables() -> None:
    model = _model().add_term(
        "t", fns.constant, args=["kc"], target="v1", mechanism="relative"
    )
    rhs = model.get_right_hand_side()
    base = _model().get_right_hand_side()
    assert _f(rhs["S"]) == pytest.approx(base["S"] * 1.1)
    assert _f(rhs["P"] - base["P"]) == pytest.approx(-(rhs["S"] - base["S"]))


def test_reaction_term_with_dynamic_stoichiometry() -> None:
    model = (
        _model()
        .add_reaction(
            "v3",
            fns.constant,
            args=["k2"],
            stoichiometry={"S": Derived(fn=fns.twice, args=["P"])},
        )
        .add_term("t", fns.constant, args=["kc"], target="v3", mechanism="additive")
    )
    assert _f(model.get_right_hand_side()["S"]) == pytest.approx(
        -1.0 + 2 * 0.5 * (0.5 + 0.1)
    )


def test_order_matters_within_a_target() -> None:
    a = (
        _model()
        .add_term("add", fns.constant, args=["kd"], target="P", mechanism="additive")
        .add_term("rel", fns.constant, args=["kc"], target="P", mechanism="relative")
    )
    b = (
        _model()
        .add_term("rel", fns.constant, args=["kc"], target="P", mechanism="relative")
        .add_term("add", fns.constant, args=["kd"], target="P", mechanism="additive")
    )
    base = _model().get_right_hand_side()["P"]
    assert _f(a.get_right_hand_side()["P"]) == pytest.approx((base + 0.2) * 1.1)
    assert _f(b.get_right_hand_side()["P"]) == pytest.approx(base * 1.1 + 0.2)


def test_terms_are_not_in_the_stoichiometric_matrix() -> None:
    model = _with_terms()
    assert list(model.get_stoichiometries().columns) == ["v1", "v2"]
    assert list(model.get_fluxes().index) == ["v1", "v2"]
    corrected = model.get_fluxes(corrected=True)
    assert _f(corrected["v1"]) == pytest.approx(1.1)
    assert _f(corrected["v2"]) == pytest.approx(0.25)


@pytest.mark.parametrize("mechanism", ["additive", "relative", exp_mechanism])
def test_zero_term_is_bit_identical(mechanism: object) -> None:
    def zero(x: float) -> float:
        return 0.0 * x

    model = (
        _model()
        .add_term("t1", zero, args=["kc"], target="v1", mechanism=mechanism)  # type: ignore[arg-type]
        .add_term("t2", zero, args=["kc"], target="P", mechanism=mechanism)  # type: ignore[arg-type]
    )
    expected = _simulate(_model())
    if mechanism is exp_mechanism:
        pd.testing.assert_frame_equal(_simulate(model), expected)
    else:
        assert np.array_equal(_simulate(model).to_numpy(), expected.to_numpy())


def test_contributions_in_results() -> None:
    res = Simulator(_with_terms()).simulate_time_course(T).get_result()
    sim = res.unwrap_or_err()
    contributions = cast(pd.DataFrame, sim.get_term_contributions())
    assert list(contributions.columns) == ["t_rxn", "t_add", "t_rel", "t_exp"]
    assert list(sim.fluxes.columns) == ["v1", "v2"]
    # readouts can read contributions
    model = _with_terms().add_readout("r", fns.twice, args=["t_add"])
    assert _f(model.get_args(include_readouts=True)["r"]) == pytest.approx(0.4)


###############################################################################
# API
###############################################################################


def test_validation() -> None:
    model = _with_terms()
    with pytest.raises(KeyError, match="neither a variable nor a reaction"):
        model.add_term("x", fns.constant, args=["kc"], target="Q", mechanism="additive")
    with pytest.raises(ValueError, match="Unknown mechanism"):
        model.add_term("x", fns.constant, args=["kc"], target="S", mechanism="foo")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two arguments"):
        model.add_term(
            "x",
            fns.constant,
            args=["kc"],
            target="S",
            mechanism=fns.constant,  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(NameError):
        model.add_term(
            "v1", fns.constant, args=["kc"], target="S", mechanism="additive"
        )
    assert "x" not in model.ids


def test_remove_update_and_rename() -> None:
    model = _with_terms()
    for name in model.get_term_names():
        model.remove_term(name)
    pd.testing.assert_series_equal(
        model.get_right_hand_side(), _model().get_right_hand_side()
    )

    model = _with_terms().rename("t_add", "leak").rename("P", "Q")
    assert model.get_term_names() == ["t_rxn", "leak", "t_rel", "t_exp"]
    assert {t.target for t in model.get_raw_terms().values()} == {"v1", "Q"}
    assert _f(model.get_right_hand_side()["Q"]) == pytest.approx(
        _with_terms().get_right_hand_side()["P"]
    )

    model.update_term("leak", mechanism="relative", args=["kc"])
    assert model.get_raw_terms()["leak"].mechanism.name == "relative"
    assert model.get_term_names()[1] == "leak"
    assert "kc" in model.get_raw_terms()["leak"].args


def test_mechanism_presets_are_exact() -> None:
    additive = Mechanism.of("additive")
    relative = Mechanism.of("relative")
    assert additive.contribution(1e16, 1.0) == 1.0
    assert relative.contribution(3.0, 0.5) == 1.5
    assert not additive.needs_base
    assert relative.needs_base
    assert Mechanism.of(exp_mechanism).contribution(2.0, 0.0) == 0.0


def test_ode_builder_terms() -> None:
    model = _ode_model()
    s = 1.0
    assert _f(model.get_right_hand_side()["S"]) == pytest.approx(-s * 1.1 + 0.1 * s)
    contributions = model.get_term_contributions()
    assert _f(contributions["t1"]) == pytest.approx(-0.1)
    assert _f(contributions["t2"]) == pytest.approx(0.1)
    with pytest.raises(KeyError, match="not a diff_eq"):
        model.add_term("x", fns.constant, args=["kc"], target="Q", mechanism="additive")
    bad = _ode_model().add_term(
        "t3", fns.constant, args=["t4"], target="S", mechanism="additive"
    )
    with pytest.raises(KeyError, match="unknown args"):
        bad.get_right_hand_side()


def test_compose_copies_terms() -> None:
    other = KineticModelBuilder().add_parameters({"kx": 1.0})
    composed = compose(_with_terms(), other)
    assert composed.get_term_names() == _with_terms().get_term_names()
    pd.testing.assert_series_equal(
        composed.get_right_hand_side(), _with_terms().get_right_hand_side()
    )


###############################################################################
# Lowering
###############################################################################


@pytest.mark.parametrize("model_fn", [_with_terms, _ode_model])
def test_native_matches_python(model_fn: object) -> None:
    model = model_fn()  # type: ignore[operator]
    pd.testing.assert_frame_equal(
        _simulate(model, integrator=Native),
        _simulate(model, integrator=Scipy),
        rtol=1e-5,
        atol=1e-7,
    )


def test_native_outputs_include_contributions() -> None:
    res = Simulator(_with_terms(), integrator=Native).simulate_time_course(T)
    native = cast(
        pd.DataFrame, res.get_result().unwrap_or_err().get_term_contributions()
    )
    res = Simulator(_with_terms(), integrator=Scipy).simulate_time_course(T)
    python = cast(
        pd.DataFrame, res.get_result().unwrap_or_err().get_term_contributions()
    )
    pd.testing.assert_frame_equal(native, python, rtol=1e-5, atol=1e-7)


def test_generated_python_code_matches() -> None:
    model = _with_terms()
    code = generate_model_code_py(model)
    ns: dict = {}
    exec(code.full(), ns)  # noqa: S102
    y = model.get_initial_conditions()
    rhs = ns["model"](0.0, [y["S"], y["P"]])
    np.testing.assert_allclose(rhs, model.get_right_hand_side().to_numpy())
    # terms act outside the stoichiometry, so there is no nv
    assert "nv" not in ns


def test_generated_mxlpy_code_round_trips() -> None:
    model = _with_terms()
    ns: dict = {"math": math}  # imports of user functions aren't generated
    exec(generate_mxlpy_code(model), ns)  # noqa: S102
    pd.testing.assert_series_equal(
        ns["create_model"]().get_right_hand_side(), model.get_right_hand_side()
    )


def test_lambda_mechanism_is_symbolic() -> None:
    model = _model().add_term(
        "t", fns.constant, args=["kd"], target="P", mechanism=lambda b, t: b * t
    )
    pd.testing.assert_frame_equal(
        _simulate(model, integrator=Native), _simulate(model), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("model_fn", [_with_terms, _ode_model])
def test_sensitivities_through_terms(model_fn: object) -> None:
    model = model_fn()  # type: ignore[operator]
    wrt = ["kc"]
    exact = forward_sensitivities.time_course(model, T, wrt, method="forward")
    fd = forward_sensitivities.time_course(model, T, wrt, method="finite_difference")
    e, f = exact.unwrap_or_err(), fd.unwrap_or_err()
    np.testing.assert_allclose(
        e.derivatives["kc"].to_numpy(),
        f.derivatives["kc"].to_numpy(),
        rtol=5e-3,
        atol=1e-4,
    )

    names = model.get_variable_names()
    w = pd.DataFrame(np.ones((len(T), len(names))), index=T, columns=names)
    adj = adjoint_sensitivities.time_course(model, w, wrt, method="adjoint")
    fwd = adjoint_sensitivities.time_course(model, w, wrt, method="forward")
    np.testing.assert_allclose(
        adj.unwrap_or_err().gradient, fwd.unwrap_or_err().gradient, rtol=1e-4
    )


def test_unsupported_exports_raise(tmp_path: object) -> None:
    model = _with_terms()
    with pytest.raises(NotImplementedError, match="terms"):
        generate_model_code_mxlweb(model)
    with pytest.raises(NotImplementedError, match="terms"):
        sbml.write(model, Path(str(tmp_path)) / "m.xml")
    with pytest.raises(SerializationError, match="terms"):
        serialize.save(model, Path(str(tmp_path)) / "m.mxl.json")


###############################################################################
# Review fixes
###############################################################################


def test_removal_drops_or_keeps_terms() -> None:
    model = _with_terms().remove_reaction("v1")
    assert "t_rxn" not in model.get_term_names()
    kept = _with_terms().remove_reaction("v1", remove_terms=False)
    assert "t_rxn" in kept.get_term_names()
    model = _with_terms().make_variable_static("P")
    assert model.get_term_names() == ["t_rxn"]
    # composition keeps the terms of the overridden components
    other = (
        KineticModelBuilder()
        .add_variable("P", 0.5)
        .add_reaction(
            "v1", fns.mass_action_1s, args=["S", "k1"], stoichiometry={"S": -1, "P": 1}
        )
    )
    with pytest.warns(UserWarning, match="overridden"):
        composed = compose(_with_terms(), other, raise_on_conflict=False)
    assert composed.get_term_names() == _with_terms().get_term_names()
    assert _f(composed.get_right_hand_side()["P"]) == pytest.approx(
        _f(_with_terms().get_right_hand_side()["P"])
    )


def test_ode_readouts_and_terms_read_contributions() -> None:
    model = (
        _ode_model()
        .add_readout("r", fns.twice, args=["t1"])
        .add_term("t3", fns.constant, args=["t2"], target="S", mechanism="additive")
    )
    args = model.get_args(include_readouts=True)
    assert _f(args["r"]) == pytest.approx(-0.2)
    assert _f(args["t3"]) == pytest.approx(0.1)
    assert _f(model.get_right_hand_side()["S"]) == pytest.approx(-1.1 + 0.1 + 0.1)
    bad = _ode_model().add_derived("d", fns.constant, args=["t1"])
    with pytest.raises(ValueError, match="computes after the derivatives"):
        bad.get_right_hand_side()


def test_update_term_is_atomic() -> None:
    model = _with_terms()
    with pytest.raises(KeyError):
        model.update_term("t_add", fn=fns.constant, args=["kc"], target="nope")
    assert model.get_raw_terms()["t_add"].args == ["S", "kd"]
    with pytest.raises(ValueError, match="two arguments"):
        model.update_term("t_add", mechanism=fns.constant)  # pyright: ignore[reportArgumentType]


def test_corrected_fluxes_of_simulations() -> None:
    sim = Simulator(_with_terms()).simulate_time_course(T).get_result().unwrap_or_err()
    fluxes = cast(pd.DataFrame, sim.get_fluxes())
    corrected = cast(pd.DataFrame, sim.get_fluxes(corrected=True))
    contributions = cast(pd.DataFrame, sim.get_term_contributions())
    np.testing.assert_array_equal(
        corrected["v1"].to_numpy(),
        (fluxes["v1"] + contributions["t_rxn"]).to_numpy(),
    )
    np.testing.assert_array_equal(corrected["v2"].to_numpy(), fluxes["v2"].to_numpy())


def test_lambda_mechanism_codegen() -> None:
    model = _model().add_term(
        "t", fns.constant, args=["kd"], target="P", mechanism=lambda b, t: b * t
    )
    ns: dict = {}
    exec(generate_mxlpy_code(model), ns)  # noqa: S102
    pd.testing.assert_series_equal(
        ns["create_model"]().get_right_hand_side(), model.get_right_hand_side()
    )
