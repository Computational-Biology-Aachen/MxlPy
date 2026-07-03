from mxlpy import InitialAssignment, KineticModelBuilder


def init_k1(k2: float) -> float:
    return 2.0 * k2


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter(
            "k1",
            value=InitialAssignment(fn=init_k1, args=["k2"]),
        )
        .add_parameter("k2", value=0.3)
    )
