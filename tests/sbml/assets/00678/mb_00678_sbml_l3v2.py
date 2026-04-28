from mxlpy import Derived, InitialAssignment, Model


def init_S1(C: float, k1: float) -> float:
    return 0.133333333333333 * C * k1


def k1() -> float:
    return 0.75


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(C: float) -> float:
    return -1.0 * C


def reaction1_stoich_S2(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["C", "k1"]),
        )
        .add_variable("S2", initial_value=0.15)
        .add_parameter("C", value=6.6)
        .add_derived(
            "k1",
            fn=k1,
            args=[],
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
