from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def S3_conc(S3: float, compartment: float) -> float:
    return S3 / compartment


def S4_conc(compartment: float, S4: float) -> float:
    return S4 / compartment


def reaction1(S2_conc: float, S1_conc: float, k1: float) -> float:
    return S1_conc * S2_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S2(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S3(compartment: float) -> float:
    return 1.0 * compartment


def reaction1_stoich_S4(compartment: float) -> float:
    return 2.0 * compartment


def reaction2(S4_conc: float, S3_conc: float, k2: float) -> float:
    return S3_conc * S4_conc * k2


def reaction2_stoich_S3(compartment: float) -> float:
    return -1.0 * compartment


def reaction2_stoich_S4(compartment: float) -> float:
    return -1.0 * compartment


def reaction2_stoich_S1(compartment: float) -> float:
    return 1.0 * compartment


def reaction2_stoich_S2(compartment: float) -> float:
    return 1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.001)
        .add_variable("S2", initial_value=0.001)
        .add_variable("S3", initial_value=0.002)
        .add_variable("S4", initial_value=0.001)
        .add_parameter("k1", value=750.0)
        .add_parameter("k2", value=250.0)
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
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "compartment"],
        )
        .add_derived(
            "S4_conc",
            fn=S4_conc,
            args=["compartment", "S4"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S2_conc", "S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["compartment"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["compartment"]),
                "S4": Derived(fn=reaction1_stoich_S4, args=["compartment"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S4_conc", "S3_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["compartment"]),
                "S4": Derived(fn=reaction2_stoich_S4, args=["compartment"]),
                "S1": Derived(fn=reaction2_stoich_S1, args=["compartment"]),
                "S2": Derived(fn=reaction2_stoich_S2, args=["compartment"]),
            },
        )
    )
