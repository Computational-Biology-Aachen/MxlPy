import math

from mxlpy import Model


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def reaction2(S2: float, k2: float) -> float:
    return S2**2 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.00015)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k1", value=0.35)
        .add_parameter("k2", value=180.0)
        .add_parameter("compartment", value=math.nan)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 2.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S2", "k2"],
            stoichiometry={"S2": -2.0, "S1": 1.0},
        )
    )
