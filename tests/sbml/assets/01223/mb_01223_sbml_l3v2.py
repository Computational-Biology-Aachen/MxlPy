from mxlpy import Model


def c(time: float) -> float:
    return time + 2


def S1_conc(S1: float, c: float) -> float:
    return S1 / c


def dS1() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.2)
        .add_derived(
            "c",
            fn=c,
            args=["time"],
        )
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["S1", "c"],
        )
        .add_reaction(
            "dS1",
            fn=dS1,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
