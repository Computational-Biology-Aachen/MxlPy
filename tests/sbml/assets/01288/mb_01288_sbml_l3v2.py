from mxlpy import Model


def J0() -> float:
    return False


def J1() -> float:
    return True


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": -1.0},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
