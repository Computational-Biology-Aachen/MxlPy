from mxlpy import Derived, Model


def S4(S3: float, p1: float) -> float:
    return S3 / (p1 + 1)


def S5(p1: float, S4: float) -> float:
    return S4 * p1


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def S3_conc(S3: float, compartment: float) -> float:
    return S3 / compartment


def S4_conc(compartment: float, S4: float) -> float:
    return S4 / compartment


def S5_conc(S5: float, compartment: float) -> float:
    return S5 / compartment


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S3(compartment: float) -> float:
    return 1.0 * compartment


def reaction2(S5_conc: float, k2: float) -> float:
    return S5_conc * k2


def reaction2_stoich_S3(compartment: float) -> float:
    return -1.0 * compartment


def reaction2_stoich_S2(compartment: float) -> float:
    return 1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.0)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("k1", value=0.1)
        .add_parameter("k2", value=0.15)
        .add_parameter("p1", value=2.5)
        .add_parameter("compartment", value=1.0)
        .add_derived(
            "S4",
            fn=S4,
            args=["S3", "p1"],
        )
        .add_derived(
            "S5",
            fn=S5,
            args=["p1", "S4"],
        )
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
        .add_derived(
            "S5_conc",
            fn=S5_conc,
            args=["S5", "compartment"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["compartment"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S5_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["compartment"]),
                "S2": Derived(fn=reaction2_stoich_S2, args=["compartment"]),
            },
        )
    )
