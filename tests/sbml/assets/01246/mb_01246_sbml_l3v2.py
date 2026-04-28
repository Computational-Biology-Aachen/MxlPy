from mxlpy import Derived, Model


def S1_conc(S1: float, c: float) -> float:
    return S1 / c


def J0() -> float:
    return 2


def J1() -> float:
    return 1


def J1_stoich_S1(c: float) -> float:
    return 1.0 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=2.0)
        .add_parameter("c", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["S1", "c"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=[],
            stoichiometry={"S1": Derived(fn=J1_stoich_S1, args=["c"])},
        )
    )
