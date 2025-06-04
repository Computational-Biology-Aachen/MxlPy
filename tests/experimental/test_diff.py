# from __future__ import annotations

# import pytest

# from mxlpy import Model, fns
# from mxlpy.experimental.diff import (
#     DerivedDiff,
#     ModelDiff,
#     ReactionDiff,
#     model_diff,
#     soft_eq,
# )
# from mxlpy.types import Derived


# @pytest.fixture
# def basic_model1() -> Model:
#     model = Model()
#     model.add_parameters({"k1": 1.0, "k2": 2.0})
#     model.add_variables({"S": 10.0, "P": 0.0})

#     model.add_reaction(
#         "v1",
#         fn=fns.mass_action_1s,
#         args=["S", "k1"],
#         stoichiometry={"S": -1.0, "P": 1.0},
#     )

#     model.add_reaction(
#         "v2",
#         fn=fns.mass_action_1s,
#         args=["P", "k2"],
#         stoichiometry={"P": -1.0},
#     )

#     model.add_derived(
#         "D1",
#         fn=fns.add,
#         args=["S", "P"],
#     )

#     return model


# @pytest.fixture
# def basic_model2() -> Model:
#     model = Model()
#     model.add_parameters({"k1": 1.0, "k2": 3.0})  # k2 is different
#     model.add_variables({"S": 10.0, "P": 0.0})

#     model.add_reaction(
#         "v1",
#         fn=fns.mass_action_1s,
#         args=["S", "k1"],
#         stoichiometry={"S": -1.0, "P": 1.0},
#     )

#     model.add_reaction(
#         "v2",
#         fn=fns.mass_action_1s,
#         args=["P", "k2"],
#         stoichiometry={"P": -1.0},
#     )

#     model.add_derived("D1", fn=fns.add, args=["S", "P"])

#     return model


# @pytest.fixture
# def model_with_derived_stoichiometry() -> Model:
#     model = Model()
#     model.add_parameters({"k1": 1.0})
#     model.add_variables({"S": 10.0, "P": 0.0})

#     def stoich_fn(s: float) -> float:
#         return -s / 10

#     model.add_reaction(
#         "v1",
#         fn=fns.mass_action_1s,
#         args=["S", "k1"],
#         stoichiometry={"S": Derived(fn=stoich_fn, args=["S"]), "P": 1.0},
#     )

#     return model


# def test_derived_diff() -> None:
#     diff = DerivedDiff(args1=["a", "b"], args2=["c", "d"])
#     assert diff.args1 == ["a", "b"]
#     assert diff.args2 == ["c", "d"]


# def test_reaction_diff() -> None:
#     diff = ReactionDiff(
#         args1=["a", "b"],
#         args2=["c", "d"],
#         stoichiometry1={"X": 1.0},
#         stoichiometry2={"Y": 2.0},
#     )
#     assert diff.args1 == ["a", "b"]
#     assert diff.args2 == ["c", "d"]
#     assert diff.stoichiometry1 == {"X": 1.0}
#     assert diff.stoichiometry2 == {"Y": 2.0}


# def test_model_diff_empty() -> None:
#     diff = ModelDiff()
#     assert diff.missing_parameters == set()
#     assert diff.missing_variables == set()
#     assert diff.missing_reactions == set()
#     assert diff.different_parameters == {}
#     assert diff.different_variables == {}
#     assert diff.different_reactions == {}


# def test_model_diff_str() -> None:
#     diff = ModelDiff()
#     diff.missing_parameters = {"k1", "k2"}
#     diff.different_parameters = {"k3": (1.0, 2.0)}
#     diff.missing_variables = {"S", "P"}
#     diff.different_variables = {"E": (0.1, 0.2)}
#     diff.missing_derived = {"D1"}
#     diff.missing_reactions = {"v1"}
#     diff.different_reactions = {"v2": ReactionDiff(args1=["a"], args2=["b"])}

#     result = str(diff)
#     assert "Missing Parameters: " in result
#     assert "k1" in result
#     assert "k2" in result
#     assert "Different Parameters:" in result
#     assert "k3: 1.0 != 2.0" in result
#     assert "Missing Variables: " in result
#     assert "S" in result
#     assert "P" in result
#     assert "Different Variables:" in result
#     assert "E: 0.1 != 0.2" in result
#     assert "Missing Derived: " in result
#     assert "D1" in result
#     assert "Missing Reactions: " in result
#     assert "v1" in result
#     assert "Different Reactions:" in result
#     assert "v2:" in result
#     assert "Args: ['a'] != ['b']" in result


# def test_soft_eq_identical_models(basic_model1: Model) -> None:
#     model1 = basic_model1
#     model2 = basic_model1
#     assert soft_eq(model1, model2)


# def test_soft_eq_different_models(basic_model1: Model, basic_model2: Model) -> None:
#     assert not soft_eq(basic_model1, basic_model2)


# def test_model_diff_identical(basic_model1: Model) -> None:
#     diff = model_diff(basic_model1, basic_model1)
#     assert not diff.missing_parameters
#     assert not diff.missing_variables
#     assert not diff.missing_reactions
#     assert not diff.missing_derived
#     assert not diff.different_parameters
#     assert not diff.different_variables
#     assert not diff.different_reactions
#     assert not diff.different_derived


# def test_model_diff_different_parameters(
#     basic_model1: Model, basic_model2: Model
# ) -> None:
#     diff = model_diff(basic_model1, basic_model2)
#     assert not diff.missing_parameters
#     assert not diff.missing_variables
#     assert not diff.missing_reactions
#     assert not diff.missing_derived
#     assert "k2" in diff.different_parameters
#     assert diff.different_parameters["k2"] == (2.0, 3.0)
#     assert not diff.different_variables
#     assert not diff.different_reactions
#     assert not diff.different_derived


# def test_model_diff_missing_reaction(basic_model1: Model) -> None:
#     model1 = basic_model1
#     model2 = Model()
#     model2.add_parameters({"k1": 1.0, "k2": 2.0})
#     model2.add_variables({"S": 10.0, "P": 0.0})

#     diff = model_diff(model1, model2)
#     assert not diff.missing_parameters
#     assert not diff.missing_variables
#     assert "v1" in diff.missing_reactions
#     assert "v2" in diff.missing_reactions
#     assert "D1" in diff.missing_derived


# def test_model_diff_different_stoichiometry(basic_model1: Model) -> None:
#     model1 = basic_model1
#     model2 = Model()
#     model2.add_parameters({"k1": 1.0, "k2": 2.0})
#     model2.add_variables({"S": 10.0, "P": 0.0})

#     model2.add_reaction(
#         "v1",
#         fn=fns.mass_action_1s,
#         args=["S", "k1"],
#         stoichiometry={"S": -2.0, "P": 1.0},  # Changed -1.0 to -2.0
#     )

#     model2.add_reaction(
#         "v2",
#         fn=fns.mass_action_1s,
#         args=["P", "k2"],
#         stoichiometry={"P": -1.0},
#     )

#     model2.add_derived("D1", fn=fns.add, args=["S", "P"])

#     diff = model_diff(model1, model2)
#     assert "v1" in diff.different_reactions
#     assert diff.different_reactions["v1"].stoichiometry1 == {"S": -1.0, "P": 1.0}
#     assert diff.different_reactions["v1"].stoichiometry2 == {"S": -2.0, "P": 1.0}


# def test_model_diff_derived_stoichiometry(
#     model_with_derived_stoichiometry: Model,
# ) -> None:
#     model1 = model_with_derived_stoichiometry
#     model2 = Model()
#     model2.add_parameters({"k1": 1.0})
#     model2.add_variables({"S": 10.0, "P": 0.0})

#     def different_stoich_fn(s: float) -> float:
#         return -s / 5  # Changed from 10 to 5

#     model2.add_reaction(
#         "v1",
#         fn=fns.mass_action_1s,
#         args=["S", "k1"],
#         stoichiometry={
#             "S": Derived(fn=different_stoich_fn, args=["S"]),
#             "P": 1.0,
#         },
#     )

#     # Here we can't directly check stoichiometry content since Derived instances
#     # will have different function objects, but we can still test that
#     # the reaction is detected as different
#     diff = model_diff(model1, model2)
#     assert "v1" in diff.different_reactions
