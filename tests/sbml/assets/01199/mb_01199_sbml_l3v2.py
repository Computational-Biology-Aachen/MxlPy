from mxlpy import Model


def z(x: float) -> float:
    return (1) if (x >= 0.49) else (0)


def dx() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("x", initial_value=0.0)
        .add_derived(
            "z",
            fn=z,
            args=["x"],
        )
        .add_reaction(
            "dx",
            fn=dx,
            args=[],
            stoichiometry={"x": 1.0},
        )
    )
