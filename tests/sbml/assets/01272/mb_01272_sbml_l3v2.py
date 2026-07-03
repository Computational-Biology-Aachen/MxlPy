from mxlpy import InitialAssignment, KineticModelBuilder


def init_p1() -> float:
    return 4


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_parameter(
        "p1",
        value=InitialAssignment(fn=init_p1, args=[]),
    )
