import numpy as np
import pandas as pd
import pytest

from mxlpy.model import Model
from mxlpy.types import MockSurrogate
from tests.model.test_model import two_arguments


def test_add_surrogate() -> None:
    model = Model()
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["x"],
            outputs=["v1"],
            stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
        ),
    )

    surrogates = model.get_raw_surrogates()
    assert "surrogate1" in surrogates
    assert model._ids["surrogate1"] == "surrogate"


def test_add_surrogate_existing_name() -> None:
    mock_surrogate = MockSurrogate(
        fn=lambda x: (x,),
        args=["x"],
        outputs=["v1"],
        stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
    )

    model = Model()
    model.add_surrogate("surrogate1", mock_surrogate)
    with pytest.raises(NameError):
        model.add_surrogate("surrogate1", mock_surrogate)


def test_add_surrogate_protected_name() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.add_surrogate(
            "time",
            MockSurrogate(
                fn=lambda x: (x,),
                args=["x"],
                outputs=["v1"],
                stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
            ),
        )


def test_update_surrogate() -> None:
    model = Model()
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["x"],
            outputs=["v1"],
            stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
        ),
    )
    new_surrogate = MockSurrogate(
        fn=lambda x: x,
        args=["x"],
        outputs=["v1"],
        stoichiometries={"v1": {"x": 1.0}},
    )
    model.update_surrogate("surrogate1", new_surrogate)

    surrogates = model.get_raw_surrogates()
    assert surrogates["surrogate1"] == new_surrogate


def test_update_surrogate_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.update_surrogate(
            "surrogate1",
            MockSurrogate(
                fn=lambda x: (x,),
                args=["x"],
                outputs=["v1"],
                stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
            ),
        )


def test_remove_surrogate() -> None:
    model = Model()
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["x"],
            outputs=["v1"],
            stoichiometries={"v1": {"x": -1.0, "y": 1.0}},
        ),
    )
    model.remove_surrogate("surrogate1")

    surrogates = model.get_raw_surrogates()
    assert "surrogate1" not in surrogates
    assert "surrogate1" not in model._ids


def test_remove_surrogate_nonexistent() -> None:
    model = Model()
    with pytest.raises(KeyError):
        model.remove_surrogate("surrogate1")


def test_get_fluxes_with_surrogate() -> None:
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
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["A"],
            outputs=["flux1"],
            stoichiometries={"flux1": {"A": 1.0, "B": 1.0}},
        ),
    )
    concs = {"A": 1.0, "B": 2.0}
    fluxes = model.get_fluxes(concs)
    assert "reaction1" in fluxes
    assert fluxes["reaction1"] == 3.0
    assert "flux1" in fluxes
    assert fluxes["flux1"] == 1.0


def test_get_fluxes_time_course_with_surrogate() -> None:
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
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["A"],
            outputs=["flux1"],
            stoichiometries={"flux1": {"A": 1.0, "B": 1.0}},
        ),
    )
    concs = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]}, index=[0.0, 1.0])
    args_time_course = model.get_args_time_course(concs)
    fluxes_time_course = model.get_fluxes_time_course(args_time_course)

    assert fluxes_time_course["reaction1"].iloc[0] == 3.0
    assert fluxes_time_course["reaction1"].iloc[1] == 5.0
    assert fluxes_time_course["flux1"].iloc[0] == 1.0
    assert fluxes_time_course["flux1"].iloc[1] == 2.0


def test_call_with_surrogate() -> None:
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
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["A"],
            outputs=["flux1"],
            stoichiometries={"flux1": {"A": 1.0, "B": 1.0}},
        ),
    )
    time = 0.0
    concs = np.array([1.0, 2.0])
    result = model(time, concs)
    assert result[0] == -2.0
    assert result[1] == 4.0


def test_get_right_hand_side_with_surrogate() -> None:
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
    model.add_surrogate(
        "surrogate1",
        MockSurrogate(
            fn=lambda x: (x,),
            args=["A"],
            outputs=["flux1"],
            stoichiometries={"flux1": {"A": 1.0, "B": 1.0}},
        ),
    )
    concs = {"A": 1.0, "B": 2.0}
    rhs = model.get_right_hand_side(concs)
    assert rhs["A"] == -2.0
    assert rhs["B"] == 4.0
