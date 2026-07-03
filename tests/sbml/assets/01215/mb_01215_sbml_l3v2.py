from mxlpy import KineticModelBuilder


def dx() -> float:
    return 0


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("x", initial_value=3.0)
        .add_reaction(
            "dx",
            fn=dx,
            args=[],
            stoichiometry={"x": 1.0},
        )
    )
