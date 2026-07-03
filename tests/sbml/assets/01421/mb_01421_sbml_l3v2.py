from mxlpy import KineticModelBuilder


def J0() -> float:
    return -1


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("A", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"A": -3.0},
        )
    )
