from mxlpy import InitialAssignment, KineticModelBuilder


def init_x() -> float:
    return 3


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter(
        "x",
        value=InitialAssignment(fn=init_x, args=[]),
    )
