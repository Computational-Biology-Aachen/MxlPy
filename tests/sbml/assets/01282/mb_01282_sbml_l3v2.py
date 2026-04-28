from mxlpy import InitialAssignment, Model


def init_p1() -> float:
    return 0


def init_p2() -> float:
    return False


def init_p3() -> float:
    return False


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "p1",
            value=InitialAssignment(fn=init_p1, args=[]),
        )
        .add_parameter(
            "p2",
            value=InitialAssignment(fn=init_p2, args=[]),
        )
        .add_parameter(
            "p3",
            value=InitialAssignment(fn=init_p3, args=[]),
        )
    )
