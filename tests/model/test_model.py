from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mxlpy.model import Model


def no_arguments() -> float:
    return 1.0


def one_argument(x: float) -> float:
    return x


def one_argument_v2(x: float) -> float:
    return x + 1


def two_arguments(x: float, y: float) -> float:
    return x + y


def two_arguments_v2(x: float, y: float) -> float:
    return x + y


def test_add_parameter() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 1.0


def test_add_parameter_existing_name() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    with pytest.raises(NameError):
        model.add_parameter("param1", 2.0)


def test_add_parameter_protected_name() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.add_parameter("time", 1.0)


def test_add_parameters() -> None:
    model = Model()
    model.add_parameters({"param1": 1.0, "param2": 2.0})

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 1.0
    assert parameters["param2"] == 2.0


def test_add_parameters_existing_name() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    with pytest.raises(NameError):
        model.add_parameters({"param1": 2.0, "param2": 2.0})


def test_add_parameters_protected_name() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.add_parameters({"time": 1.0, "param2": 2.0})


def test_parameters() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 1.0
    assert parameters["param2"] == 2.0


def test_parameters_empty() -> None:
    model = Model()
    parameters = model.get_parameter_values()
    assert parameters == {}


def test_get_parameter_names() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)
    parameter_names = model.get_parameter_names()
    assert "param1" in parameter_names
    assert "param2" in parameter_names
    assert len(parameter_names) == 2


def test_get_parameter_names_empty() -> None:
    model = Model()
    parameter_names = model.get_parameter_names()
    assert parameter_names == []


def test_remove_parameter() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.remove_parameter("param1")

    parameters = model.get_parameter_values()
    assert "param1" not in parameters
    assert "param1" not in model._ids


def test_remove_parameter_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_parameter("param1")


def test_remove_parameters() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)
    model.remove_parameters(["param1", "param2"])

    parameters = model.get_parameter_values()
    assert "param1" not in parameters
    assert "param2" not in parameters
    assert "param1" not in model._ids
    assert "param2" not in model._ids


def test_remove_parameters_partial() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)
    model.add_parameter("param3", 3.0)
    model.remove_parameters(["param1", "param3"])

    parameters = model.get_parameter_values()
    assert "param1" not in parameters
    assert "param3" not in parameters
    assert "param2" in parameters
    assert parameters["param2"] == 2.0
    assert "param1" not in model._ids
    assert "param3" not in model._ids
    assert "param2" in model._ids


def test_remove_parameters_nonexistent() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    with pytest.raises(KeyError):
        model.remove_parameters(["param1", "param2"])


def test_update_parameter() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.update_parameter("param1", 2.0)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 2.0


def test_update_parameter_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.update_parameter("param1", 2.0)


def test_update_parameters() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)
    update_params = {"param1": 1.5, "param2": 2.5}
    model.update_parameters(update_params)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 1.5
    assert parameters["param2"] == 2.5


def test_update_parameters_nonexistent() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    update_params = {"param1": 1.5, "param2": 2.5}
    with pytest.raises(KeyError):
        model.update_parameters(update_params)


def test_scale_parameter() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.scale_parameter("param1", 2.0)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 2.0


def test_scale_parameter_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.scale_parameter("param1", 2.0)


def test_scale_parameters() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_parameter("param2", 2.0)
    scale_factors = {"param1": 2.0, "param2": 0.5}
    model.scale_parameters(scale_factors)

    parameters = model.get_parameter_values()
    assert parameters["param1"] == 2.0
    assert parameters["param2"] == 1.0


def test_scale_parameters_nonexistent() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    scale_factors = {"param1": 2.0, "param2": 0.5}
    with pytest.raises(KeyError):
        model.scale_parameters(scale_factors)


def test_make_parameter_dynamic() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.make_parameter_dynamic("param1")

    parameters = model.get_parameter_values()
    variables = model.get_initial_conditions()
    assert "param1" not in parameters
    assert "param1" in variables
    assert variables["param1"] == 1.0
    assert model._ids["param1"] == "variable"


def test_make_parameter_dynamic_with_initial_value() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.make_parameter_dynamic("param1", initial_value=2.0)

    parameters = model.get_parameter_values()
    variables = model.get_initial_conditions()
    assert "param1" not in parameters
    assert "param1" in variables
    assert variables["param1"] == 2.0
    assert model._ids["param1"] == "variable"


def test_make_parameter_dynamic_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.make_parameter_dynamic("param1")


def test_variables() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.add_variable("var2", 2.0)

    variables = model.get_initial_conditions()
    assert "var1" in variables
    assert variables["var1"] == 1.0
    assert "var2" in variables
    assert variables["var2"] == 2.0


def test_variables_empty() -> None:
    model = Model()
    variables = model.get_initial_conditions()
    assert variables == {}


def test_add_variable() -> None:
    model = Model()
    model.add_variable("var1", 1.0)

    variables = model.get_initial_conditions()
    assert "var1" in variables
    assert variables["var1"] == 1.0
    assert model._ids["var1"] == "variable"


def test_add_variable_existing_name() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    with pytest.raises(NameError):
        model.add_variable("var1", 2.0)


def test_add_variable_protected_name() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.add_variable("time", 1.0)


def test_add_variables() -> None:
    model = Model()
    model.add_variables({"var1": 1.0, "var2": 2.0})

    variables = model.get_initial_conditions()
    assert "var1" in variables
    assert variables["var1"] == 1.0
    assert model._ids["var1"] == "variable"
    assert "var2" in variables
    assert variables["var2"] == 2.0
    assert model._ids["var2"] == "variable"


def test_add_variables_existing_name() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    with pytest.raises(NameError):
        model.add_variables({"var1": 2.0, "var2": 2.0})


def test_add_variables_protected_name() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.add_variables({"time": 1.0, "var2": 2.0})


def test_remove_variable() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.remove_variable("var1", remove_stoichiometries=True)

    variables = model.get_initial_conditions()
    assert "var1" not in variables
    assert "var1" not in model._ids


def test_remove_variable_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_variable("var1", remove_stoichiometries=True)


def test_remove_variables() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.add_variable("var2", 2.0)
    model.remove_variables(["var1", "var2"], remove_stoichiometries=True)

    variables = model.get_initial_conditions()
    assert "var1" not in variables
    assert "var2" not in variables
    assert "var1" not in model._ids
    assert "var2" not in model._ids


def test_remove_variables_partial() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.add_variable("var2", 2.0)
    model.add_variable("var3", 3.0)
    model.remove_variables(["var1", "var3"], remove_stoichiometries=True)

    variables = model.get_initial_conditions()
    assert "var1" not in variables
    assert "var3" not in variables
    assert "var2" in variables
    assert variables["var2"] == 2.0
    assert "var1" not in model._ids
    assert "var3" not in model._ids
    assert "var2" in model._ids


def test_remove_variables_nonexistent() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    with pytest.raises(KeyError):
        model.remove_variables(["var1", "var2"], remove_stoichiometries=True)


def test_update_variable() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.update_variable("var1", 2.0)

    variables = model.get_initial_conditions()
    assert variables["var1"] == 2.0


def test_update_variable_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.update_variable("var1", 2.0)


def test_get_variable_names() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.add_variable("var2", 2.0)
    variable_names = model.get_variable_names()
    assert "var1" in variable_names
    assert "var2" in variable_names
    assert len(variable_names) == 2


def test_get_variable_names_empty() -> None:
    model = Model()
    variable_names = model.get_variable_names()
    assert variable_names == []


def test_get_initial_conditions() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.add_variable("var2", 2.0)
    initial_conditions = model.get_initial_conditions()
    assert "var1" in initial_conditions
    assert initial_conditions["var1"] == 1.0
    assert "var2" in initial_conditions
    assert initial_conditions["var2"] == 2.0


def test_get_initial_conditions_empty() -> None:
    model = Model()
    initial_conditions = model.get_initial_conditions()
    assert initial_conditions == {}


def test_make_variable_static() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.make_variable_static("var1")

    parameters = model.get_parameter_values()
    variables = model.get_initial_conditions()
    assert "var1" not in variables
    assert "var1" in parameters
    assert parameters["var1"] == 1.0
    assert model._ids["var1"] == "parameter"


def test_make_variable_static_with_value() -> None:
    model = Model()
    model.add_variable("var1", 1.0)
    model.make_variable_static("var1", value=2.0)

    parameters = model.get_parameter_values()
    variables = model.get_initial_conditions()
    assert "var1" not in variables
    assert "var1" in parameters
    assert parameters["var1"] == 2.0
    assert model._ids["var1"] == "parameter"


def test_make_variable_static_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.make_variable_static("var1")


def test_derived_variables() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_variable("x", 1.0)
    model.add_derived("derived1", derived_fn, args=["x"])
    derived_vars = model.get_derived_variables()
    assert "derived1" in derived_vars
    assert derived_vars["derived1"].fn == derived_fn
    assert derived_vars["derived1"].args == ["x"]


def test_derived_variables_empty() -> None:
    model = Model()
    derived_vars = model.get_derived_variables()
    assert derived_vars == {}


def test_derived_parameters() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_parameter("param1", 1.0)
    model.add_derived("derived_param1", derived_fn, args=["param1"])
    derived_params = model.get_derived_parameters()
    assert "derived_param1" in derived_params
    assert derived_params["derived_param1"].fn == derived_fn
    assert derived_params["derived_param1"].args == ["param1"]


def test_derived_parameters_empty() -> None:
    model = Model()
    derived_params = model.get_derived_parameters()
    assert derived_params == {}


def test_add_derived() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_derived("derived1", derived_fn, args=["x"])

    derived = model.get_raw_derived()
    assert "derived1" in derived
    assert derived["derived1"].fn == derived_fn
    assert derived["derived1"].args == ["x"]
    assert model._ids["derived1"] == "derived"


def test_add_derived_existing_name() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_derived("derived1", derived_fn, args=["x"])
    with pytest.raises(NameError):
        model.add_derived("derived1", derived_fn, args=["x"])


def test_add_derived_protected_name() -> None:
    model = Model()
    derived_fn = one_argument
    with pytest.raises(KeyError):
        model.add_derived("time", derived_fn, args=["x"])


def test_get_derived_parameter_names() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_parameter("param1", 1.0)
    model.add_derived("derived_param1", derived_fn, args=["param1"])
    derived_param_names = model.get_derived_parameter_names()
    assert "derived_param1" in derived_param_names
    assert len(derived_param_names) == 1


def test_get_derived_parameter_names_empty() -> None:
    model = Model()
    derived_param_names = model.get_derived_parameter_names()
    assert derived_param_names == []


def test_get_derived_variable_names() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_variable("x", 1.0)
    model.add_derived("derived1", derived_fn, args=["x"])
    derived_var_names = model.get_derived_variable_names()
    assert "derived1" in derived_var_names
    assert len(derived_var_names) == 1


def test_get_derived_variable_names_empty() -> None:
    model = Model()
    derived_var_names = model.get_derived_variable_names()
    assert derived_var_names == []


def test_update_derived_fn() -> None:
    model = Model()
    derived_fn = one_argument
    new_derived_fn = one_argument_v2
    model.add_variable("x", 1.0)
    model.add_derived("derived1", derived_fn, args=["x"])
    model.update_derived("derived1", fn=new_derived_fn)

    derived = model.get_raw_derived()
    assert derived["derived1"].fn == new_derived_fn
    assert derived["derived1"].args == ["x"]


def test_update_derived_args() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_variable("x", 1.0)
    model.add_variable("y", 2.0)
    model.add_derived("derived1", derived_fn, args=["x"])
    model.update_derived("derived1", args=["y"])

    derived = model.get_raw_derived()
    assert derived["derived1"].fn == derived_fn
    assert derived["derived1"].args == ["y"]


def test_update_derived_fn_and_args() -> None:
    model = Model()
    derived_fn = one_argument
    new_derived_fn = one_argument_v2
    model.add_variable("x", 1.0)
    model.add_variable("y", 2.0)
    model.add_derived("derived1", derived_fn, args=["x"])
    model.update_derived("derived1", fn=new_derived_fn, args=["y"])

    derived = model.get_raw_derived()
    assert derived["derived1"].fn == new_derived_fn
    assert derived["derived1"].args == ["y"]


def test_update_derived_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.update_derived("derived1", fn=one_argument, args=["x"])


def test_remove_derived() -> None:
    model = Model()
    derived_fn = one_argument
    model.add_derived("derived1", derived_fn, args=["x"])
    model.remove_derived("derived1")

    derived = model.get_raw_derived()
    assert "derived1" not in derived
    assert "derived1" not in model._ids


def test_remove_derived_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_derived("derived1")


def test_reactions() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    reactions = model.get_raw_reactions()
    assert "reaction1" in reactions
    assert reactions["reaction1"].fn == reaction_fn
    assert reactions["reaction1"].stoichiometry == stoichiometry
    assert reactions["reaction1"].args == ["A", "B"]


def test_reactions_empty() -> None:
    model = Model()
    reactions = model.get_raw_reactions()
    assert reactions == {}


def test_add_reaction() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )

    reactions = model.get_raw_reactions()
    assert "reaction1" in reactions
    assert reactions["reaction1"].fn == reaction_fn
    assert reactions["reaction1"].stoichiometry == stoichiometry
    assert reactions["reaction1"].args == ["A", "B"]
    assert model._ids["reaction1"] == "reaction"


def test_add_reaction_existing_name() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    with pytest.raises(NameError):
        model.add_reaction(
            "reaction1",
            fn=reaction_fn,
            stoichiometry=stoichiometry,
            args=["A", "B"],
        )


def test_add_reaction_protected_name() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    with pytest.raises(KeyError):
        model.add_reaction(
            "time",
            fn=reaction_fn,
            stoichiometry=stoichiometry,
            args=["A", "B"],
        )


def test_get_reaction_names() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    reaction_names = model.get_reaction_names()
    assert "reaction1" in reaction_names
    assert len(reaction_names) == 1


def test_get_reaction_names_empty() -> None:
    model = Model()
    reaction_names = model.get_reaction_names()
    assert reaction_names == []


def test_update_reaction_fn() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    new_reaction_fn = two_arguments_v2
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction("reaction1", fn=new_reaction_fn)

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == new_reaction_fn
    assert reactions["reaction1"].stoichiometry == stoichiometry
    assert reactions["reaction1"].args == ["A", "B"]


def test_update_reaction_stoichiometry() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_stoichiometry = {"A": -2.0, "B": 2.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction("reaction1", stoichiometry=new_stoichiometry)

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == reaction_fn
    assert reactions["reaction1"].stoichiometry == new_stoichiometry
    assert reactions["reaction1"].args == ["A", "B"]


def test_update_reaction_args() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_args = ["B", "A"]
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction("reaction1", args=new_args)

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == reaction_fn
    assert reactions["reaction1"].stoichiometry == stoichiometry
    assert reactions["reaction1"].args == new_args


def test_update_reaction_fn_and_stoichiometry() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    new_reaction_fn = two_arguments_v2
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_stoichiometry = {"A": -2.0, "B": 2.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction(
        "reaction1", fn=new_reaction_fn, stoichiometry=new_stoichiometry
    )

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == new_reaction_fn
    assert reactions["reaction1"].stoichiometry == new_stoichiometry
    assert reactions["reaction1"].args == ["A", "B"]


def test_update_reaction_fn_and_args() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    new_reaction_fn = two_arguments_v2
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_args = ["B", "A"]
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction("reaction1", fn=new_reaction_fn, args=new_args)

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == new_reaction_fn
    assert reactions["reaction1"].stoichiometry == stoichiometry
    assert reactions["reaction1"].args == new_args


def test_update_reaction_stoichiometry_and_args() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_stoichiometry = {"A": -2.0, "B": 2.0}
    new_args = ["B", "A"]
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction("reaction1", stoichiometry=new_stoichiometry, args=new_args)

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == reaction_fn
    assert reactions["reaction1"].stoichiometry == new_stoichiometry
    assert reactions["reaction1"].args == new_args


def test_update_reaction_fn_stoichiometry_and_args() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    new_reaction_fn = two_arguments_v2
    stoichiometry = {"A": -1.0, "B": 1.0}
    new_stoichiometry = {"A": -2.0, "B": 2.0}
    new_args = ["B", "A"]
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.update_reaction(
        "reaction1", fn=new_reaction_fn, stoichiometry=new_stoichiometry, args=new_args
    )

    reactions = model.get_raw_reactions()
    assert reactions["reaction1"].fn == new_reaction_fn
    assert reactions["reaction1"].stoichiometry == new_stoichiometry
    assert reactions["reaction1"].args == new_args


def test_update_reaction_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.update_reaction("reaction1")


def test_remove_reaction() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    model.remove_reaction("reaction1")

    reactions = model.get_raw_reactions()
    assert "reaction1" not in reactions
    assert "reaction1" not in model._ids


def test_remove_reaction_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_reaction("reaction1")


def test_add_readout() -> None:
    model = Model()
    readout_fn = one_argument
    model.add_readout("readout1", readout_fn, args=["x"])

    readouts = model.get_raw_readouts()
    assert "readout1" in readouts
    assert readouts["readout1"].fn == readout_fn
    assert readouts["readout1"].args == ["x"]
    assert model._ids["readout1"] == "readout"


def test_add_readout_existing_name() -> None:
    model = Model()
    readout_fn = one_argument
    model.add_readout("readout1", readout_fn, args=["x"])
    with pytest.raises(NameError):
        model.add_readout("readout1", readout_fn, args=["x"])


def test_add_readout_protected_name() -> None:
    model = Model()
    readout_fn = one_argument
    with pytest.raises(KeyError):
        model.add_readout("time", readout_fn, args=["x"])


def test_get_readout_names() -> None:
    model = Model()
    readout_fn = one_argument
    model.add_readout("readout1", readout_fn, args=["x"])
    model.add_readout("readout2", readout_fn, args=["x"])

    readout_names = model.get_readout_names()
    assert "readout1" in readout_names
    assert "readout2" in readout_names
    assert len(readout_names) == 2


def test_get_readout_names_empty() -> None:
    model = Model()
    readout_names = model.get_readout_names()
    assert readout_names == []


def test_remove_readout() -> None:
    model = Model()
    readout_fn = one_argument
    model.add_readout("readout1", readout_fn, args=["x"])
    model.remove_readout("readout1")

    readouts = model.get_raw_readouts()
    assert "readout1" not in readouts
    assert "readout1" not in model._ids


def test_remove_readout_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_readout("readout1")


def test_get_args() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    args = model.get_args(
        {"var1": 2.0},
        include_parameters=False,
        include_time=False,
        include_readouts=False,
    )
    assert args["var1"] == 2.0
    assert args["derived1"] == 2.0


def test_get_args_without_readouts() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    args = model.get_args({"var1": 2.0})
    assert args["param1"] == 1.0
    assert args["var1"] == 2.0
    assert args["derived1"] == 2.0
    assert "readout1" not in args


def test_get_args_with_multiple_concs() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_variable("var2", 3.0)
    model.add_derived("derived1", two_arguments, args=["var1", "var2"])
    model.add_readout("readout1", two_arguments, args=["var1", "var2"])

    args = model.get_args({"var1": 2.0, "var2": 3.0})
    assert args["param1"] == 1.0
    assert args["var1"] == 2.0
    assert args["var2"] == 3.0
    assert args["derived1"] == 5.0
    assert "readout1" not in args
    assert args["time"] == 0.0


def test_get_args_with_empty_concs() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    with pytest.raises(KeyError):
        model.get_args({})


def test_get_args_time_course() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    args_time_course = model.get_args_time_course(
        pd.DataFrame(
            {"var1": [2.0, 3.0]},
            index=[0.0, 1.0],
        )
    )
    assert args_time_course["param1"].iloc[0] == 1.0
    assert args_time_course["param1"].iloc[1] == 1.0
    assert args_time_course["var1"].iloc[0] == 2.0
    assert args_time_course["var1"].iloc[1] == 3.0
    assert args_time_course["derived1"].iloc[0] == 2.0
    assert args_time_course["derived1"].iloc[1] == 3.0
    assert "readout1" not in args_time_course.columns


def test_get_args_time_course_with_readouts() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    args_time_course = model.get_args_time_course(
        pd.DataFrame(
            {"var1": [2.0, 3.0]},
            index=[0.0, 1.0],
        ),
        include_readouts=True,
    )

    assert args_time_course["param1"].iloc[0] == 1.0
    assert args_time_course["param1"].iloc[1] == 1.0
    assert args_time_course["var1"].iloc[0] == 2.0
    assert args_time_course["var1"].iloc[1] == 3.0
    assert args_time_course["derived1"].iloc[0] == 2.0
    assert args_time_course["derived1"].iloc[1] == 3.0
    assert args_time_course["readout1"].iloc[0] == 2.0
    assert args_time_course["readout1"].iloc[1] == 3.0


def test_get_args_time_course_with_multiple_concs() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_variable("var2", 3.0)
    model.add_derived("derived1", two_arguments, args=["var1", "var2"])
    model.add_readout("readout1", two_arguments, args=["var1", "var2"])

    concs = pd.DataFrame({"var1": [2.0, 3.0], "var2": [3.0, 4.0]}, index=[0.0, 1.0])
    args_time_course = model.get_args_time_course(concs, include_readouts=True)

    assert args_time_course["param1"].iloc[0] == 1.0
    assert args_time_course["param1"].iloc[1] == 1.0
    assert args_time_course["var1"].iloc[0] == 2.0
    assert args_time_course["var1"].iloc[1] == 3.0
    assert args_time_course["var2"].iloc[0] == 3.0
    assert args_time_course["var2"].iloc[1] == 4.0
    assert args_time_course["derived1"].iloc[0] == 5.0
    assert args_time_course["derived1"].iloc[1] == 7.0
    assert args_time_course["readout1"].iloc[0] == 5.0
    assert args_time_course["readout1"].iloc[1] == 7.0


def test_get_args_time_course_with_empty_concs() -> None:
    model = Model()
    model.add_parameter("param1", 1.0)
    model.add_variable("var1", 2.0)
    model.add_derived("derived1", one_argument, args=["var1"])
    model.add_readout("readout1", one_argument, args=["var1"])

    with pytest.raises(KeyError):
        model.get_args_time_course(pd.DataFrame({}, index=[]))


def test_get_full_args() -> None:
    model = Model()
    model.add_variable("var1", 2.0)
    model.add_variable("var2", 3.0)
    model.add_derived("derived1", two_arguments, args=["var1", "var2"])
    model.add_readout("readout1", two_arguments, args=["var1", "var2"])

    concs = {"var1": 2.0, "var2": 3.0}
    full_concs = model.get_args(concs, include_readouts=True)

    assert full_concs["var1"] == 2.0
    assert full_concs["var2"] == 3.0
    assert full_concs["derived1"] == 5.0
    assert full_concs["readout1"] == 5.0


def test_get_full_args_without_readouts() -> None:
    model = Model()
    model.add_variable("var1", 2.0)
    model.add_variable("var2", 3.0)
    model.add_derived("derived1", two_arguments, args=["var1", "var2"])
    model.add_readout("readout1", two_arguments, args=["var1", "var2"])

    concs = {"var1": 2.0, "var2": 3.0}
    full_concs = model.get_args(concs, include_readouts=False)

    assert full_concs["var1"] == 2.0
    assert full_concs["var2"] == 3.0
    assert full_concs["derived1"] == 5.0
    assert "readout1" not in full_concs


def test_get_full_args_with_empty_concs() -> None:
    model = Model()
    model.add_variable("var1", 2.0)
    model.add_variable("var2", 3.0)
    model.add_derived("derived1", two_arguments, args=["var1", "var2"])
    model.add_readout("readout1", two_arguments, args=["var1", "var2"])

    with pytest.raises(KeyError):
        model.get_args({})


def test_get_fluxes() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    concs = {"A": 1.0, "B": 2.0}
    fluxes = model.get_fluxes(concs)
    assert "reaction1" in fluxes
    assert fluxes["reaction1"] == 3.0


def test_get_fluxes_empty_reactions() -> None:
    model = Model()
    concs = {"A": 1.0, "B": 2.0}
    fluxes = model.get_fluxes(concs)
    assert fluxes.empty


def test_get_fluxes_empty_concs() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    with pytest.raises(KeyError):
        model.get_fluxes({})


def test_get_fluxes_time_course() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    concs = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]}, index=[0.0, 1.0])
    args_time_course = model.get_args_time_course(concs)
    fluxes_time_course = model.get_fluxes_time_course(args_time_course)

    assert fluxes_time_course["reaction1"].iloc[0] == 3.0
    assert fluxes_time_course["reaction1"].iloc[1] == 5.0


def test_get_fluxes_time_course_empty_reactions() -> None:
    model = Model()
    concs = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]}, index=[0.0, 1.0])
    args_time_course = model.get_args_time_course(concs)
    fluxes_time_course = model.get_fluxes_time_course(args_time_course)

    assert fluxes_time_course.empty


def test_get_fluxes_time_course_empty_args() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    concs = pd.DataFrame({}, index=[])
    with pytest.raises(KeyError):
        model.get_fluxes_time_course(concs)


def test_call() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    time = 0.0
    concs = np.array([1.0, 2.0])
    result = model(time, concs)
    assert result[0] == -3.0
    assert result[1] == 3.0


def test_call_empty_reactions() -> None:
    model = Model()
    time = 0.0
    concs = np.array([1.0, 2.0])
    with pytest.raises(Exception):  # noqa: B017
        model(time, concs)


def test_call_empty_concs() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    time = 0.0
    with pytest.raises(ValueError):
        model(time, np.array([]))


def test_get_right_hand_side() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    concs = {"A": 1.0, "B": 2.0}
    rhs = model.get_right_hand_side(concs)
    assert rhs["A"] == -3.0
    assert rhs["B"] == 3.0


def test_get_right_hand_side_empty_concs() -> None:
    model = Model()
    model.add_variables({"A": 1.0, "B": 2.0})
    reaction_fn = two_arguments
    stoichiometry = {"A": -1.0, "B": 1.0}
    model.add_reaction(
        "reaction1",
        fn=reaction_fn,
        stoichiometry=stoichiometry,
        args=["A", "B"],
    )
    with pytest.raises(KeyError):
        model.get_right_hand_side({})
