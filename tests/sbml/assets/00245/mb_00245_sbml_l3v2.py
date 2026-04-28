import math

from mxlpy import Model


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, S4: float, k2: float) -> float:
    return S3 * S4 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S3", initial_value=2.0)
        .add_variable("S4", initial_value=0.5)
        .add_parameter("k1", value=0.6)
        .add_parameter("k2", value=0.13)
        .add_parameter("compartment", value=math.nan)
        .add_parameter("S1", value=1.0)
        .add_parameter("S2", value=1.5)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1", "S2"],
            stoichiometry={"S3": 1.0, "S4": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "S4", "k2"],
            stoichiometry={"S3": -1.0, "S4": -1.0},
        )
    )
