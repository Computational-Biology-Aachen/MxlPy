from mxlpy import Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def reaction1(compartment: float, S1_conc: float, k1: float) -> float:
    return S1_conc * k1 / compartment


def reaction2(compartment: float, S2_conc: float, k2: float) -> float:
    return S2_conc * k2 / compartment


def create_model() -> Model:
    return (
        Model()
        .add_parameter("k1", value=0.5)
        .add_parameter("k2", value=0.25)
        .add_parameter("compartment", value=1.0)
        .add_parameter("S1", value=0.0015)
        .add_parameter("S2", value=0.001)
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
            "reaction1",
            fn=reaction1,
            args=["compartment", "S1_conc", "k1"],
            stoichiometry={},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["compartment", "S2_conc", "k2"],
            stoichiometry={},
        )
    )
