from mxlpy import Model


def S1_conc(S1: float, c: float) -> float:
    return S1 / c


def J0() -> float:
    return 2


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
    )
