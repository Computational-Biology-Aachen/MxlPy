from mxlpy import InitialAssignment, Model


def init_p2() -> float:
    return 1.602214179


def create_model() -> Model:
    return (
        Model()
        .add_parameter("p1", value=5.0)
        .add_parameter(
            "p2",
            value=InitialAssignment(fn=init_p2, args=[]),
        )
    )
