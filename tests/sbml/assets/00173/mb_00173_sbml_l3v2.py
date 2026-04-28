import math

import scipy

from mxlpy import Model


def dS1(S1: float, p1: float, p2: float) -> float:
    return -scipy.special.factorial(math.ceil(S1 * p1)) / p2


def dS2(S1: float, p1: float, p2: float) -> float:
    return scipy.special.factorial(math.ceil(S1 * p1)) / p2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("p1", value=4.0)
        .add_parameter("p2", value=25.0)
        .add_reaction(
            "dS1",
            fn=dS1,
            args=["S1", "p1", "p2"],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S1", "p1", "p2"],
            stoichiometry={"S2": 1.0},
        )
    )
