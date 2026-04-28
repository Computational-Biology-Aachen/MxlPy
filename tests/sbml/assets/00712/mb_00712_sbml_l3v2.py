from mxlpy import Derived, Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def dk1(k3: float, k2: float) -> float:
    return k2 + k3


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=1.0)
        .add_variable("S1", initial_value=0.15)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k2", value=0.2)
        .add_parameter("k3", value=0.3)
        .add_parameter("C", value=2.5)
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
            "dk1",
            fn=dk1,
            args=["k3", "k2"],
            stoichiometry={"k1": 1.0},
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
