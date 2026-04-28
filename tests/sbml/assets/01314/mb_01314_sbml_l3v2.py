from mxlpy import InitialAssignment, Model


def init_p1() -> float:
    return 10


def create_model() -> Model:
    return Model().add_parameter(
        "p1",
        value=InitialAssignment(fn=init_p1, args=[]),
    )
