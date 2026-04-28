from mxlpy import InitialAssignment, Model


def init_p1() -> float:
    return False


def init_p2() -> float:
    return True


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
    )
