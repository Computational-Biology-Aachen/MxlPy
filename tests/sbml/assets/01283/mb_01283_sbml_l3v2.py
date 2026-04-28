from mxlpy import InitialAssignment, Model


def init_p1() -> float:
    return False


def init_p2() -> float:
    return True


def init_p3() -> float:
    return False


def init_p4() -> float:
    return 4.0


def init_p5() -> float:
    return -4.0


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
        .add_parameter(
            "p4",
            value=InitialAssignment(fn=init_p4, args=[]),
        )
        .add_parameter(
            "p5",
            value=InitialAssignment(fn=init_p5, args=[]),
        )
    )
