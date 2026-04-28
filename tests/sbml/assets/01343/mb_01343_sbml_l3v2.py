import math

from mxlpy import InitialAssignment, Model


def init_P1() -> float:
    return 0.291312612451591


def P2() -> float:
    return -0.379948962255225


def P3(time: float) -> float:
    return math.tanh(time)


def P4(time: float) -> float:
    return -math.tanh(time)


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "P1",
            value=InitialAssignment(fn=init_P1, args=[]),
        )
        .add_derived(
            "P2",
            fn=P2,
            args=[],
        )
        .add_derived(
            "P3",
            fn=P3,
            args=["time"],
        )
        .add_derived(
            "P4",
            fn=P4,
            args=["time"],
        )
    )
