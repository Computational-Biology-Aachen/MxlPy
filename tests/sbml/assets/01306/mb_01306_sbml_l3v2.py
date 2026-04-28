from mxlpy import Model


def p1(J0: float) -> float:
    return J0 + 1


def J0(time: float) -> float:
    return time


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "p1",
            fn=p1,
            args=["J0"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["time"],
            stoichiometry={},
        )
    )
