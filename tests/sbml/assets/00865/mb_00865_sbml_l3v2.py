from mxlpy import Derived, Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def reaction1(time: float, S2_conc: float, S1_conc: float, k1: float) -> float:
    return S1_conc * S2_conc * k1 * time


def reaction1_stoich_S2(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S3(C: float) -> float:
    return 1.0 * C


def reaction2(time: float, S3_conc: float, k2: float) -> float:
    return S3_conc * k2 * time


def reaction2_stoich_S3(C: float) -> float:
    return -1.0 * C


def reaction2_stoich_S2(C: float) -> float:
    return 1.0 * C


def reaction3(k3: float, time: float, S3_conc: float) -> float:
    return S3_conc * k3 * time


def reaction3_stoich_S3(C: float) -> float:
    return -1.0 * C


def reaction3_stoich_S4(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S2", initial_value=2.0)
        .add_variable("S3", initial_value=0.0)
        .add_variable("S4", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("k2", value=0.9)
        .add_parameter("k3", value=0.7)
        .add_parameter("C", value=1.0)
        .add_parameter("S1", value=2.0)
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
            "reaction1",
            fn=reaction1,
            args=["time", "S2_conc", "S1_conc", "k1"],
            stoichiometry={
                "S2": Derived(fn=reaction1_stoich_S2, args=["C"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["C"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["time", "S3_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["C"]),
                "S2": Derived(fn=reaction2_stoich_S2, args=["C"]),
            },
        )
        .add_reaction(
            "reaction3",
            fn=reaction3,
            args=["k3", "time", "S3_conc"],
            stoichiometry={
                "S3": Derived(fn=reaction3_stoich_S3, args=["C"]),
                "S4": Derived(fn=reaction3_stoich_S4, args=["C"]),
            },
        )
    )
