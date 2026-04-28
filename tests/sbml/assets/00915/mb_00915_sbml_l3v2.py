from mxlpy import Model


def dc(c: float) -> float:
    return 0.15 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable("c", initial_value=0.5)
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
