import math

from mxlpy import InitialAssignment, Model


def init_P() -> float:
    return math.inf


def init_Q() -> float:
    return float("-inf")


def init_R() -> float:
    return math.nan


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "P",
            value=InitialAssignment(fn=init_P, args=[]),
        )
        .add_parameter(
            "Q",
            value=InitialAssignment(fn=init_Q, args=[]),
        )
        .add_parameter(
            "R",
            value=InitialAssignment(fn=init_R, args=[]),
        )
    )
