from mxlpy import Derived, InitialAssignment, Model


def init_generatedId_0(p1: float) -> float:
    return 4 * p1


def S4(S1: float, k3: float) -> float:
    return S1 * k3


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def dS2(S3: float, S1: float, k1: float, k2: float) -> float:
    return -S1 * k1 + S3 * k2


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S3(C: float) -> float:
    return 1.0 * C


def reaction1_stoich_S1(C: float, generatedId_0: float) -> float:
    return -1.0 * C * generatedId_0


def reaction2(S3_conc: float, k2: float) -> float:
    return S3_conc * k2


def reaction2_stoich_S3(C: float) -> float:
    return -1.0 * C


def reaction2_stoich_S1(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "generatedId_0",
            initial_value=InitialAssignment(fn=init_generatedId_0, args=["p1"]),
        )
        .add_variable("S2", initial_value=1.5e-5)
        .add_variable("S1", initial_value=1.0e-5)
        .add_variable("S3", initial_value=1.0e-5)
        .add_parameter("k1", value=0.015)
        .add_parameter("k2", value=0.5)
        .add_parameter("k3", value=1.5)
        .add_parameter("p1", value=0.25)
        .add_parameter("C", value=1.0)
        .add_derived(
            "S4",
            fn=S4,
            args=["S1", "k3"],
        )
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
            "dS2",
            fn=dS2,
            args=["S3", "S1", "k1", "k2"],
            stoichiometry={"S2": "C"},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S3": Derived(fn=reaction1_stoich_S3, args=["C"]),
                "S1": Derived(fn=reaction1_stoich_S1, args=["C", "generatedId_0"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["C"]),
                "S1": Derived(fn=reaction2_stoich_S1, args=["C"]),
            },
        )
    )
