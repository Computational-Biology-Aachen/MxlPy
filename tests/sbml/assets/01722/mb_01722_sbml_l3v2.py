from mxlpy import Derived, InitialAssignment, Model


def init_S1_stoich() -> float:
    return 6.02214179


def J0() -> float:
    return 0.1


def J0_stoich_S1(S1_stoich: float) -> float:
    return 1.0 * S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1_stoich",
            initial_value=InitialAssignment(fn=init_S1_stoich, args=[]),
        )
        .add_variable("S1", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["S1_stoich"])},
        )
    )
