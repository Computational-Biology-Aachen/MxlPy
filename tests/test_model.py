# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from mxlpy import Derived, InitialAssignment, Model


def test_failure_difference() -> None:
    # FIXME: implement this
    assert True


def test_model_add_data() -> None:
    # FIXME: implement this
    assert True


def test_model_add_derived() -> None:
    # FIXME: implement this
    assert True


def test_model_add_parameter() -> None:
    # FIXME: implement this
    assert True


def test_model_add_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_add_reaction() -> None:
    # FIXME: implement this
    assert True


def test_model_add_readout() -> None:
    # FIXME: implement this
    assert True


def test_model_add_surrogate() -> None:
    # FIXME: implement this
    assert True


def test_model_add_variable() -> None:
    # FIXME: implement this
    assert True


def test_model_add_variables() -> None:
    # FIXME: implement this
    assert True


def test_model_check_units() -> None:
    # FIXME: implement this
    assert True


def test_model_derived() -> None:
    # FIXME: implement this
    assert True


def test_model_get_arg_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_args() -> None:
    # FIXME: implement this
    assert True


def test_model_get_args_time_course() -> None:
    # FIXME: implement this
    assert True


def test_model_get_derived_parameter_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_derived_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_get_derived_variable_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_derived_variables() -> None:
    # FIXME: implement this
    assert True


def test_model_get_fluxes() -> None:
    # FIXME: implement this
    assert True


def test_model_get_fluxes_time_course() -> None:
    # FIXME: implement this
    assert True


def test_model_get_initial_conditions() -> None:
    # FIXME: implement this
    assert True


def test_model_get_parameter_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_parameter_values() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_derived() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_reactions() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_readouts() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_stoichiometries_of_variable() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_surrogates() -> None:
    # FIXME: implement this
    assert True


def test_model_get_raw_variables() -> None:
    # FIXME: implement this
    assert True


def test_model_get_reaction_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_readout_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_right_hand_side() -> None:
    # FIXME: implement this
    assert True


def test_model_get_right_hand_side_time_course() -> None:
    # FIXME: implement this
    assert True


def test_model_get_stoichiometries() -> None:
    # FIXME: implement this
    assert True


def test_model_get_stoichiometries_of_variable() -> None:
    # FIXME: implement this
    assert True


def test_model_get_surrogate_output_names() -> None:
    # FIXME: implement this
    assert True


def test_model_get_surrogate_reaction_names() -> None:
    # FIXME: implement this
    assert True


def _rate(x: float, k: float) -> float:
    return k * x


def _inverse(b: float) -> float:
    return 1 / b


def _double(c: float) -> float:
    return 2 * c


def test_model_get_unused_parameters() -> None:
    m = (
        Model()
        .add_variables({"x": 1.0, "y": 0.0})
        .add_parameters({"k": 1.0, "b": 10.0, "c": 3.0, "r": 2.0, "unused": 0.0})
        .add_parameter("ia", InitialAssignment(fn=_double, args=["c"]))
        .add_reaction(
            "v",
            fn=_rate,
            args=["x", "k"],
            stoichiometry={"x": -1, "y": Derived(fn=_inverse, args=["b"])},
        )
        .add_readout("ro", fn=_double, args=["r"])
    )
    assert m.get_unused_parameters() == {"unused", "ia"}


def test_model_get_unused_parameters_removal_keeps_model_valid() -> None:
    m = (
        Model()
        .add_variables({"x": 1.0, "y": 0.0})
        .add_parameters({"k": 1.0, "b": 10.0})
        .add_reaction(
            "v",
            fn=_rate,
            args=["x", "k"],
            stoichiometry={"x": -1, "y": Derived(fn=_inverse, args=["b"])},
        )
    )
    m.remove_parameters(m.get_unused_parameters())
    assert m.get_right_hand_side()["y"] == 0.1


def test_model_get_variable_names() -> None:
    # FIXME: implement this
    assert True


def test_model_ids() -> None:
    # FIXME: implement this
    assert True


def test_model_infer_units() -> None:
    # FIXME: implement this
    assert True


def test_model_make_parameter_dynamic() -> None:
    # FIXME: implement this
    assert True


def test_model_make_variable_static() -> None:
    # FIXME: implement this
    assert True


def test_model_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_reactions() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_data() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_derived() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_parameter() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_reaction() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_readout() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_surrogate() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_variable() -> None:
    # FIXME: implement this
    assert True


def test_model_remove_variables() -> None:
    # FIXME: implement this
    assert True


def test_model_scale_parameter() -> None:
    # FIXME: implement this
    assert True


def test_model_scale_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_update_data() -> None:
    # FIXME: implement this
    assert True


def test_model_update_derived() -> None:
    # FIXME: implement this
    assert True


def test_model_update_parameter() -> None:
    # FIXME: implement this
    assert True


def test_model_update_parameters() -> None:
    # FIXME: implement this
    assert True


def test_model_update_reaction() -> None:
    # FIXME: implement this
    assert True


def test_model_update_surrogate() -> None:
    # FIXME: implement this
    assert True


def test_model_update_variable() -> None:
    # FIXME: implement this
    assert True


def test_model_update_variables() -> None:
    # FIXME: implement this
    assert True


def test_model_variables() -> None:
    # FIXME: implement this
    assert True


def test_unitcheck_correct_diff_eqs() -> None:
    # FIXME: implement this
    assert True


def test_unitcheck_report() -> None:
    # FIXME: implement this
    assert True
