from mxlpy import InitialAssignment, Model


def init_a() -> float:
    return 1


def init_b() -> float:
    return 2


def init_c() -> float:
    return 1


def init_d() -> float:
    return 1


def init_e() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "a",
            value=InitialAssignment(fn=init_a, args=[]),
        )
        .add_parameter(
            "b",
            value=InitialAssignment(fn=init_b, args=[]),
        )
        .add_parameter(
            "c",
            value=InitialAssignment(fn=init_c, args=[]),
        )
        .add_parameter(
            "d",
            value=InitialAssignment(fn=init_d, args=[]),
        )
        .add_parameter(
            "e",
            value=InitialAssignment(fn=init_e, args=[]),
        )
    )
