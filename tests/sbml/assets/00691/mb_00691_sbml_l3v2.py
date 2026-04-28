from mxlpy import Derived, Model


def S3(k1: float, S2: float) -> float:
    return S2 * k1


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def reaction1(S1_conc: float, k2: float) -> float:
    return S1_conc * k2


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S2", initial_value=0.5)
        .add_parameter("k1", value=1.5)
        .add_parameter("k2", value=5.0)
        .add_parameter("C", value=1.0)
        .add_parameter("S1", value=1.0)
        .add_derived(
            "S3",
            fn=S3,
            args=["k1", "S2"],
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
            args=["S1_conc", "k2"],
            stoichiometry={"S2": Derived(fn=reaction1_stoich_S2, args=["C"])},
        )
    )
