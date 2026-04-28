from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S2(compartment: float) -> float:
    return 1.0 * compartment


def reaction2(S2_conc: float, k2: float) -> float:
    return S2_conc * k2


def reaction2_stoich_S2(compartment: float) -> float:
    return -1.0 * compartment


def reaction2_stoich_S1(compartment: float) -> float:
    return 1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0015)
        .add_variable("S2", initial_value=0.0015)
        .add_parameter("k1", value=0.5)
        .add_parameter("k2", value=0.5)
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
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["compartment"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S2_conc", "k2"],
            stoichiometry={
                "S2": Derived(fn=reaction2_stoich_S2, args=["compartment"]),
                "S1": Derived(fn=reaction2_stoich_S1, args=["compartment"]),
            },
        )
    )
