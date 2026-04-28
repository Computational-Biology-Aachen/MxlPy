from mxlpy import Derived, Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def dS4() -> float:
    return 0.001


def reaction1(S2_conc: float, S1_conc: float, k1: float) -> float:
    return S1_conc * S2_conc * k1


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S3(C: float) -> float:
    return 1.0 * C


def reaction2(S3_conc: float, k2: float) -> float:
    return S3_conc * k2


def reaction2_stoich_S3(C: float) -> float:
    return -2.0 * C


def reaction2_stoich_S1(C: float) -> float:
    return 1.0 * C


def reaction2_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S4", initial_value=0.001)
        .add_variable("S1", initial_value=0.0015)
        .add_variable("S2", initial_value=0.002)
        .add_variable("S3", initial_value=0.0015)
        .add_parameter("k1", value=750.0)
        .add_parameter("k2", value=0.0025)
        .add_parameter("C", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_derived(
            "S2_conc",
            fn=S2_conc,
            args=["C", "S2"],
        )
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "C"],
        )
        .add_derived(
            "S4_conc",
            fn=S4_conc,
            args=["C", "S4"],
        )
        .add_reaction(
            "dS4",
            fn=dS4,
            args=[],
            stoichiometry={"S4": "C"},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S2_conc", "S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["C"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["C"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["C"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["C"]),
                "S1": Derived(fn=reaction2_stoich_S1, args=["C"]),
                "S2": Derived(fn=reaction2_stoich_S2, args=["C"]),
            },
        )
    )
