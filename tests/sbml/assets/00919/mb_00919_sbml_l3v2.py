from mxlpy import InitialAssignment, KineticModelBuilder


def init_c() -> float:
    return 0.525


def dc(c: float) -> float:
    return 1.25 * c


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
