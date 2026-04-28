from mxlpy import Model


def x(S1_conc: float) -> float:
    return S1_conc


def S1(S1_conc: float, C1: float) -> float:
    return C1 * S1_conc


def dS1_conc() -> float:
    return 0.4


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_conc", initial_value=0.0)
        .add_parameter("C1", value=0.5)
        .add_derived(
            "x",
            fn=x,
            args=["S1_conc"],
        )
        .add_derived(
            "S1",
            fn=S1,
            args=["S1_conc", "C1"],
        )
        .add_reaction(
            "dS1_conc",
            fn=dS1_conc,
            args=[],
            stoichiometry={"S1_conc": 1.0},
        )
    )
