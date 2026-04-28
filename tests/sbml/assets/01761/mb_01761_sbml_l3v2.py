from mxlpy import Model


def J0(avogadro: float) -> float:
    return avogadro


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_parameter("avogadro", value=0.1)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=["avogadro"],
            stoichiometry={"S1": 1.0},
        )
    )
