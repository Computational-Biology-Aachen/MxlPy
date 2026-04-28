from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def S3_conc(S3: float, compartment: float) -> float:
    return S3 / compartment


def S4_conc(compartment: float, S4: float) -> float:
    return S4 / compartment


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction2(compartment: float, S2_conc: float, k2: float) -> float:
    return S2_conc * k2 / compartment


def reaction2_stoich_S1(compartment: float) -> float:
    return 1.0 * compartment


def reaction3(S2_conc: float, k3: float) -> float:
    return S2_conc * k3


def reaction3_stoich_S3(compartment: float) -> float:
    return 1.0 * compartment


def reaction3_stoich_S4(compartment: float) -> float:
    return 1.0 * compartment


def reaction4(k4: float, S4_conc: float, S3_conc: float) -> float:
    return S3_conc * S4_conc * k4


def reaction4_stoich_S3(compartment: float) -> float:
    return -1.0 * compartment


def reaction4_stoich_S4(compartment: float) -> float:
    return -1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.1)
        .add_variable("S3", initial_value=0.0)
        .add_variable("S4", initial_value=0.0)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("k3", value=0.1)
        .add_parameter("k4", value=0.1)
        .add_parameter("compartment", value=1.0)
        .add_parameter("S2", value=0.2)
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
            args=["S1_conc", "k1"],
            stoichiometry={"S1": Derived(fn=reaction1_stoich_S1, args=["compartment"])},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["compartment", "S2_conc", "k2"],
            stoichiometry={"S1": Derived(fn=reaction2_stoich_S1, args=["compartment"])},
        )
        .add_reaction(
            "reaction3",
            fn=reaction3,
            args=["S2_conc", "k3"],
            stoichiometry={
                "S3": Derived(fn=reaction3_stoich_S3, args=["compartment"]),
                "S4": Derived(fn=reaction3_stoich_S4, args=["compartment"]),
            },
        )
        .add_reaction(
            "reaction4",
            fn=reaction4,
            args=["k4", "S4_conc", "S3_conc"],
            stoichiometry={
                "S3": Derived(fn=reaction4_stoich_S3, args=["compartment"]),
                "S4": Derived(fn=reaction4_stoich_S4, args=["compartment"]),
            },
        )
    )
