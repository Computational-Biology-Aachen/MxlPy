from mxlpy import InitialAssignment, KineticModelBuilder


def init_p1(J0: float) -> float:
    return J0


def J0() -> float:
    return 3


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable(
            "p1",
            initial_value=InitialAssignment(fn=init_p1, args=["J0"]),
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={},
        )
    )
