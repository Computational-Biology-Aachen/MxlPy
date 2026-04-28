from mxlpy import Model


def dC1(C1: float, C2: float) -> float:
    return -0.2 * C1 + 0.5 * C2


def dC2(C1: float, C2: float) -> float:
    return 0.75 * C1 - 2.1 * C2


def create_model() -> Model:
    return (
        Model()
        .add_variable("C1", initial_value=1.5)
        .add_variable("C2", initial_value=1.5)
        .add_reaction(
            "dC1",
            fn=dC1,
            args=["C1", "C2"],
            stoichiometry={"C1": 1.0},
        )
        .add_reaction(
            "dC2",
            fn=dC2,
            args=["C1", "C2"],
            stoichiometry={"C2": 1.0},
        )
    )
