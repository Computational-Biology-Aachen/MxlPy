from mxlpy import InitialAssignment, KineticModelBuilder


def init_p1() -> float:
    return False


def init_p2() -> float:
    return True


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter(
            "p1",
            value=InitialAssignment(fn=init_p1, args=[]),
        )
        .add_parameter(
            "p2",
            value=InitialAssignment(fn=init_p2, args=[]),
        )
    )
