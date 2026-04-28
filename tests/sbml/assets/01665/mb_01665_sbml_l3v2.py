from mxlpy import Model


def dp1() -> float:
    return 6.02214179


def create_model() -> Model:
    return (
        Model()
        .add_variable("p1", initial_value=1.0)
        .add_reaction(
            "dp1",
            fn=dp1,
            args=[],
            stoichiometry={"p1": 1.0},
        )
    )
