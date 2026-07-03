from mxlpy import KineticModelBuilder


def dC() -> float:
    return 1


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("C", initial_value=1.0)
        .add_parameter("S1", value=1.0)
        .add_reaction(
            "dC",
            fn=dC,
            args=[],
            stoichiometry={"C": 1.0},
        )
    )
