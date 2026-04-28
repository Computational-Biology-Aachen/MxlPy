from mxlpy import InitialAssignment, Model


def init_S1(k1: float) -> float:
    return 0.133333333333333 * k1


def k1() -> float:
    return 0.75


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["k1"]),
        )
        .add_variable("S2", initial_value=0.15)
        .add_parameter("C", value=6.6)
        .add_derived(
            "k1",
            fn=k1,
            args=[],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
