from mxlpy import Model


def reaction1(S1: float, kr: float, S2: float, kf: float) -> float:
    return S1 * kf - S2 * kr


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("kf", value=0.9)
        .add_parameter("kr", value=0.075)
        .add_parameter("C", value=0.5)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "kr", "S2", "kf"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
