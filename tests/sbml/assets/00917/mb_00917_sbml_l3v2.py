from mxlpy import InitialAssignment, KineticModelBuilder


def init_c() -> float:
    return 0.733333333333333


def dc(c: float) -> float:
    return 0.5 * c


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable(
            "c",
            initial_value=InitialAssignment(fn=init_c, args=[]),
        )
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
