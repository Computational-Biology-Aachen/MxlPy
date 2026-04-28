import math

from mxlpy import Model


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.15)
        .add_parameter("k1", value=1.023)
        .add_parameter("compartment", value=math.nan)
        .add_parameter("S2", value=0.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0},
        )
    )
