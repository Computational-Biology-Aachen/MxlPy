from mxlpy import Model


def dp1(J0: float) -> float:
    return J0


def J0() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("p1", initial_value=0.0)
        .add_variable("S1", initial_value=0.0)
        .add_parameter("c", value=1.0)
        .add_reaction(
            "dp1",
            fn=dp1,
            args=["J0"],
            stoichiometry={"p1": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": 1.0},
        )
    )
