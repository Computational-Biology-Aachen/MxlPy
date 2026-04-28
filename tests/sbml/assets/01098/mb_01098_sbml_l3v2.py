from mxlpy import Derived, InitialAssignment, Model


def init_generatedId_0(p1: float) -> float:
    return 4 * p1


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def reaction1(
    S2_conc: float, S1_conc: float, kf: float, kr: float, S3_conc: float
) -> float:
    return S1_conc * kf - S2_conc * S3_conc * kr


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def reaction1_stoich_S3(C: float, generatedId_0: float) -> float:
    return 1.0 * C * generatedId_0


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "generatedId_0",
            initial_value=InitialAssignment(fn=init_generatedId_0, args=["p1"]),
        )
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.5)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("kf", value=2.5)
        .add_parameter("kr", value=0.2)
        .add_parameter("p1", value=0.5)
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
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S2_conc", "S1_conc", "kf", "kr", "S3_conc"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["C"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["C"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["C", "generatedId_0"]),
            },
        )
    )
