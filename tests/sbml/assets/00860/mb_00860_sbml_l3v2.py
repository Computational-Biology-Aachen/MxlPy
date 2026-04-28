from mxlpy import Model


def reaction1(time: float, S1: float, k1: float) -> float:
    return S1 * k1 * time


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0015)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("C", value=0.9)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["time", "S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
