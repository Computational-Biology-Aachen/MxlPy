from mxlpy import Model


def dk1(k3: float, k2: float) -> float:
    return k2 + k3


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=1.0)
        .add_variable("S1", initial_value=0.0015)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k2", value=0.2)
        .add_parameter("k3", value=0.3)
        .add_parameter("C", value=2.5)
        .add_reaction(
            "dk1",
            fn=dk1,
            args=["k3", "k2"],
            stoichiometry={"k1": 1.0},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
