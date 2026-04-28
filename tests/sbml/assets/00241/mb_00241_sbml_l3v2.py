import math

from mxlpy import Model


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0e-6)
        .add_variable("S2", initial_value=2.0e-6)
        .add_parameter("k1", value=9.8)
        .add_parameter("k2", value=0.57)
        .add_parameter("compartment", value=math.nan)
        .add_parameter("S3", value=1.0e-6)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1", "S2"],
            stoichiometry={"S1": -1.0, "S2": -1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "k2"],
            stoichiometry={"S1": 1.0, "S2": 1.0},
        )
    )
