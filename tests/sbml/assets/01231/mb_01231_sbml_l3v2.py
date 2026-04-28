from mxlpy import Model


def J0(k1: float) -> float:
    return k1


def J1(J0: float) -> float:
    return J0 + 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("c", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=["J0"],
            stoichiometry={"S2": 1.0},
        )
    )
