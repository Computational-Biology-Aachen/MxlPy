from mxlpy import Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def dS1(S1: float, k1: float) -> float:
    return -S1 * k1


def dS2(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.015)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("compartment", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["compartment", "S1"],
        )
        .add_derived(
            "S2_conc",
            fn=S2_conc,
            args=["compartment", "S2"],
        )
        .add_reaction(
            "dS1",
            fn=dS1,
            args=["S1", "k1"],
            stoichiometry={"S1": "compartment"},
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S1", "k1"],
            stoichiometry={"S2": "compartment"},
        )
    )
