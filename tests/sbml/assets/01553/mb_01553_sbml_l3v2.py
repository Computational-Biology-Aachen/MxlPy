from mxlpy import Derived, Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def J0() -> float:
    return 0.1


def J0_stoich_S1(C: float) -> float:
    return -1.0 * C


def J0_stoich_S2(S2_stoch: float, C: float) -> float:
    return 1.0 * C * S2_stoch


def create_model() -> Model:
    return (
        Model()
        .add_variable("S2_stoch", initial_value=2.0)
        .add_variable("S1", initial_value=2.0)
        .add_variable("S2", initial_value=3.0)
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
                "S1": Derived(fn=J0_stoich_S1, args=["C"]),
                "S2": Derived(fn=J0_stoich_S2, args=["S2_stoch", "C"]),
            },
        )
    )
