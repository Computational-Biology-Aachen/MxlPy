from mxlpy import InitialAssignment, Model


def init_k1(k2: float) -> float:
    return (1 / 100) * k2


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "k1",
            initial_value=InitialAssignment(fn=init_k1, args=["k2"]),
        )
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=1.5)
        .add_parameter("k2", value=50.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
