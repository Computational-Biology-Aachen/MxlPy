from mxlpy import InitialAssignment, Model


def init_k1(time: float) -> float:
    return time


def init_k2() -> float:
    return 6.02214179e23


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "k1",
            value=InitialAssignment(fn=init_k1, args=["time"]),
        )
        .add_parameter(
            "k2",
            value=InitialAssignment(fn=init_k2, args=[]),
        )
    )
