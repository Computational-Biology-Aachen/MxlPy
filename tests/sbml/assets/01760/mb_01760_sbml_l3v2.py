from mxlpy import KineticModelBuilder


def J0(J0_avogadro: float) -> float:
    return J0_avogadro


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("S1", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_parameter("J0_avogadro", value=0.1)
        .add_reaction(
            "J0",
            fn=J0,
            args=["J0_avogadro"],
            stoichiometry={"S1": 1.0},
        )
    )
