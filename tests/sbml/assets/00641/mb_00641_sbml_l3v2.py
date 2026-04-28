from mxlpy import Derived, InitialAssignment, Model


def init_S1(C: float, p1: float) -> float:
    return 2.0 * C * p1


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def dS3(S4: float, k2: float) -> float:
    return S4 * k2


def dS4(S4: float, k2: float) -> float:
    return -S4 * k2


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S3", initial_value=1.5)
        .add_variable("S4", initial_value=4.0)
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["C", "p1"]),
        )
        .add_variable("S2", initial_value=2.0)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("p1", value=0.75)
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
            "dS3",
            fn=dS3,
            args=["S4", "k2"],
            stoichiometry={"S3": "C"},
        )
        .add_reaction(
            "dS4",
            fn=dS4,
            args=["S4", "k2"],
            stoichiometry={"S4": "C"},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["C"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["C"]),
            },
        )
    )
