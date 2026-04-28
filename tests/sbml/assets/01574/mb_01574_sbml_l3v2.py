from mxlpy import Model


def S1_stoich() -> float:
    return 2


def J0(S1: float) -> float:
    return 0.1 * S1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S2", initial_value=3.0)
        .add_parameter("C", value=1.0)
        .add_parameter("S1", value=3.0)
        .add_derived(
            "S1_stoich",
            fn=S1_stoich,
            args=[],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["S1"],
            stoichiometry={"S2": 1.0},
        )
    )
