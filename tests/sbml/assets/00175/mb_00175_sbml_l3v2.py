from mxlpy import KineticModelBuilder


def dS1() -> float:
    return 7


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("S1", initial_value=0.0)
        .add_reaction(
            "dS1",
            fn=dS1,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
