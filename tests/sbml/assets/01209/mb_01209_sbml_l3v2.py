from mxlpy import InitialAssignment, Model


def init_x() -> float:
    return 3


def create_model() -> Model:
    return Model().add_parameter(
        "x",
        value=InitialAssignment(fn=init_x, args=[]),
    )
