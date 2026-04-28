from mxlpy import Derived, InitialAssignment, Model


def init_k0(S1_stoich: float) -> float:
    return S1_stoich + 2


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def J0() -> float:
    return 0.01


def J0_stoich_S2(C: float) -> float:
    return 1.0 * C


def J0_stoich_S1(S1_stoich: float, C: float) -> float:
    return -1.0 * C * S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_stoich", initial_value=2.0)
        .add_variable("S1", initial_value=2.0)
        .add_variable("S2", initial_value=3.0)
        .add_parameter(
            "k0",
            value=InitialAssignment(fn=init_k0, args=["S1_stoich"]),
        )
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
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={
                "S2": Derived(fn=J0_stoich_S2, args=["C"]),
                "S1": Derived(fn=J0_stoich_S1, args=["S1_stoich", "C"]),
            },
        )
    )
