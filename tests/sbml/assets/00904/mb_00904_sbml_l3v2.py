from mxlpy import Model


def dc(c: float) -> float:
    return 0.25 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable("c", initial_value=3.2)
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
