from mxlpy import Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def dS1_stoich() -> float:
    return 1


def J0() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_stoich", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_parameter("S1", value=2.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_reaction(
            "dS1_stoich",
            fn=dS1_stoich,
            args=[],
            stoichiometry={"S1_stoich": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={},
        )
    )
