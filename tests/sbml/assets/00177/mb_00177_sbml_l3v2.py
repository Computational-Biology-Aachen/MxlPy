from mxlpy import Model


def dk1() -> float:
    return 0.5


def dS1(S1: float, k1: float) -> float:
    return -S1 * k1


def dS2(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.15)
        .add_variable("S2", initial_value=0.0)
        .add_variable("k1", initial_value=0.1)
        .add_reaction(
            "dk1",
            fn=dk1,
            args=[],
            stoichiometry={"k1": 1.0},
        )
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
