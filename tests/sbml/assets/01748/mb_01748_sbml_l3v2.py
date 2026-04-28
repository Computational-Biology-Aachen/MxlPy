from mxlpy import Derived, Model


def S1_stoich(time: float) -> float:
    return 0.1 * time


def J0() -> float:
    return 0.1


def J0_stoich_S1(S1_stoich: float) -> float:
    return 1.0 * S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=2.0)
        .add_parameter("C", value=2.0)
        .add_derived(
            "S1_stoich",
            fn=S1_stoich,
            args=["time"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["S1_stoich"])},
        )
    )
