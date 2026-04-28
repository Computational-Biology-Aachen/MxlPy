from mxlpy import Model


def J0() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("A", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"A": 2.0},
        )
    )
