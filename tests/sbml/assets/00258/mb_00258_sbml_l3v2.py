import math

from mxlpy import Model


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, S4: float, k2: float) -> float:
    return S3 * S4 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.001)
        .add_variable("S2", initial_value=0.0015)
        .add_variable("S3", initial_value=0.00075)
        .add_parameter("k1", value=1680.0)
        .add_parameter("k2", value=270.0)
        .add_parameter("compartment", value=math.nan)
        .add_parameter("S4", value=0.00125)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1", "S2"],
            stoichiometry={"S1": -1.0, "S2": -1.0, "S3": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "S4", "k2"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S2": 1.0},
        )
    )
