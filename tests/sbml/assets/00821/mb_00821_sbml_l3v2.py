from mxlpy import Derived, Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def reaction1(S2_conc: float, S1_conc: float, kf: float, kr: float, C: float) -> float:
    return S1_conc * kf - S2_conc * kr / C


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_parameter("kf", value=0.9)
        .add_parameter("kr", value=0.075)
        .add_parameter("C", value=1.0)
        .add_parameter("S2", value=0.0)
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
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S2_conc", "S1_conc", "kf", "kr", "C"],
            stoichiometry={"S1": Derived(fn=reaction1_stoich_S1, args=["C"])},
        )
    )
