from mxlpy import Model


def J0(J0_avogadro: float) -> float:
    return 6.02214179e23 / J0_avogadro


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_parameter("J0_avogadro", value=1.0e24)
        .add_reaction(
            "J0",
            fn=J0,
            args=["J0_avogadro"],
            stoichiometry={"S1": 1.0},
        )
    )
