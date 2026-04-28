from mxlpy import Model


def dS1(S1: float, k1: float) -> float:
    return -S1 * k1


def dS2(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.015)
        .add_variable("S2", initial_value=0.015)
        .add_parameter("k1", value=1.0)
        .add_reaction(
            "dS1",
            fn=dS1,
            args=["S1", "k1"],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S1", "k1"],
            stoichiometry={"S2": 1.0},
        )
    )
