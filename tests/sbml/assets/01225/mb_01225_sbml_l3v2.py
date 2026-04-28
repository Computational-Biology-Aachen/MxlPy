from mxlpy import Model


def p1(J0: float) -> float:
    return J0


def J0() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0)
        .add_parameter("c", value=1.0)
        .add_derived(
            "p1",
            fn=p1,
            args=["J0"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
