import math

from mxlpy import Model


def dS3(k1: float) -> float:
    return 0.5 * k1


def dS4(k2: float) -> float:
    return -0.5 * k2


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S3", initial_value=1.5)
        .add_variable("S4", initial_value=4.0)
        .add_variable("S1", initial_value=1.5)
        .add_variable("S2", initial_value=2.0)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("compartment", value=math.nan)
        .add_reaction(
            "dS3",
            fn=dS3,
            args=["k1"],
            stoichiometry={"S3": 1.0},
        )
        .add_reaction(
            "dS4",
            fn=dS4,
            args=["k2"],
            stoichiometry={"S4": 1.0},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
