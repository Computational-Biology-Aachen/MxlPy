from mxlpy import Derived, InitialAssignment, Model


def init_k0(S1_stoich: float) -> float:
    return 2 * S1_stoich


def J0() -> float:
    return 0.01


def J0_stoich_S1(S1_stoich: float) -> float:
    return -1.0 * S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_stoich", initial_value=2.0)
        .add_variable("S1", initial_value=2.0)
        .add_variable("S2", initial_value=3.0)
        .add_parameter(
            "k0",
            value=InitialAssignment(fn=init_k0, args=["S1_stoich"]),
        )
        .add_parameter("C", value=2.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={
                "S2": 1.0,
                "S1": Derived(fn=J0_stoich_S1, args=["S1_stoich"]),
            },
        )
    )
