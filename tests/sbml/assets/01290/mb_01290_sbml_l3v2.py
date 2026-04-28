from mxlpy import Model


def dp1() -> float:
    return True


def dp2() -> float:
    return False


def create_model() -> Model:
    return (
        Model()
        .add_variable("p1", initial_value=1.0)
        .add_variable("p2", initial_value=2.0)
        .add_reaction(
            "dp1",
            fn=dp1,
            args=[],
            stoichiometry={"p1": 1.0},
        )
        .add_reaction(
            "dp2",
            fn=dp2,
            args=[],
            stoichiometry={"p2": 1.0},
        )
    )
