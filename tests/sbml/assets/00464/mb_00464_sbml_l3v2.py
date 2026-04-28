from mxlpy import Model


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def reaction2(k2: float, S2: float) -> float:
    return S2 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0015)
        .add_variable("S2", initial_value=0.0015)
        .add_parameter("k1", value=0.5)
        .add_parameter("k2", value=0.5)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["k2", "S2"],
            stoichiometry={"S2": -1.0, "S1": 1.0},
        )
    )
