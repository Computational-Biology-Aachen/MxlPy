from mxlpy import InitialAssignment, KineticModelBuilder


def init_S1(c: float) -> float:
    return 0.6 * c


def dS1() -> float:
    return 1


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["c"]),
        )
        .add_parameter("c", value=2.0)
        .add_reaction(
            "dS1",
            fn=dS1,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
