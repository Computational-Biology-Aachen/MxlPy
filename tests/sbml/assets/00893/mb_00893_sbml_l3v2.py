from mxlpy import Model


def dP3(time: float, k1: float) -> float:
    return k1 * time


def dP1(P1: float, P3: float) -> float:
    return -P1 * P3


def dP2(P1: float, P3: float) -> float:
    return P1 * P3


def create_model() -> Model:
    return (
        Model()
        .add_variable("P1", initial_value=0.0015)
        .add_variable("P2", initial_value=0.0)
        .add_variable("P3", initial_value=0.001)
        .add_parameter("k1", value=0.5)
        .add_reaction(
            "dP3",
            fn=dP3,
            args=["time", "k1"],
            stoichiometry={"P3": 1.0},
        )
        .add_reaction(
            "dP1",
            fn=dP1,
            args=["P1", "P3"],
            stoichiometry={"P1": 1.0},
        )
        .add_reaction(
            "dP2",
            fn=dP2,
            args=["P1", "P3"],
            stoichiometry={"P2": 1.0},
        )
    )
