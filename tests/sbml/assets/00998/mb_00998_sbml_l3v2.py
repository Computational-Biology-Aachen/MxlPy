from mxlpy import InitialAssignment, Model


def init_S1(comp: float) -> float:
    return 1.0 * comp


def J0(S1: float) -> float:
    return 10 / S1


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["comp"]),
        )
        .add_parameter("comp", value=5.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=["S1"],
            stoichiometry={"S1": 1.0},
        )
    )
