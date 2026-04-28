import math

from mxlpy import Model


def dP1(time: float, k2: float, P2: float) -> float:
    return P2 * k2 * math.exp(-time)


def dP2(P1: float, time: float, k1: float) -> float:
    return P1 * k1 * math.exp(-time)


def create_model() -> Model:
    return (
        Model()
        .add_variable("P1", initial_value=1.5)
        .add_variable("P2", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("k2", value=0.75)
        .add_reaction(
            "dP1",
            fn=dP1,
            args=["time", "k2", "P2"],
            stoichiometry={"P1": 1.0},
        )
        .add_reaction(
            "dP2",
            fn=dP2,
            args=["P1", "time", "k1"],
            stoichiometry={"P2": 1.0},
        )
    )
