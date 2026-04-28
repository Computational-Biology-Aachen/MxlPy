from mxlpy import InitialAssignment, Model


def init_a() -> float:
    return 1


def init_b(a: float) -> float:
    return (1) if (a > 0) else (2)


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "a",
            value=InitialAssignment(fn=init_a, args=[]),
        )
        .add_parameter(
            "b",
            value=InitialAssignment(fn=init_b, args=["a"]),
        )
    )
