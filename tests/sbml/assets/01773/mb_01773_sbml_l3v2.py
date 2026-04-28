from mxlpy import Derived, Model


def J0(S1_stoich: float) -> float:
    return (1 / 200) * S1_stoich


def J0_stoich_S1(S1_stoich: float) -> float:
    return -1.0 * S1_stoich


def J1(J1_S1_stoich: float) -> float:
    return J1_S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_stoich", initial_value=2.0)
        .add_variable("S1", initial_value=2.0)
        .add_variable("S2", initial_value=3.0)
        .add_parameter("C", value=2.0)
        .add_parameter("J1_S1_stoich", value=0.01)
        .add_reaction(
            "J0",
            fn=J0,
            args=["S1_stoich"],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["S1_stoich"])},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=["J1_S1_stoich"],
            stoichiometry={"S2": 1.0},
        )
    )
