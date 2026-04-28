from mxlpy import Derived, Model


def S3(k: float, S2: float) -> float:
    return S2 * k


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def reaction1(S1_conc: float, reaction1_k: float) -> float:
    return S1_conc * reaction1_k


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k", value=0.75)
        .add_parameter("C", value=1.0)
        .add_parameter("reaction1_k", value=5.0)
        .add_derived(
            "S3",
            fn=S3,
            args=["k", "S2"],
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
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "reaction1_k"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["C"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["C"]),
            },
        )
    )
