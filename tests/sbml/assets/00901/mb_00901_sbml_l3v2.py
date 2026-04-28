from mxlpy import Model


def dc(k1: float, c: float) -> float:
    return -c * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("c", initial_value=5.0)
        .add_parameter("k1", value=1.0)
        .add_reaction(
            "dc",
            fn=dc,
            args=["k1", "c"],
            stoichiometry={"c": 1.0},
        )
    )
