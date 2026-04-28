from mxlpy import Derived, Model


def J0() -> float:
    return 1


def J0_stoich_S1(S1_sr: float) -> float:
    return 1.0 * S1_sr


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_sr", initial_value=2.0)
        .add_variable("S1", initial_value=0.0)
        .add_parameter("C1", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["S1_sr"])},
        )
    )
