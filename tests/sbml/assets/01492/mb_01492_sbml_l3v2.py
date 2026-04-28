from mxlpy import Model


def z(y: float, x: float) -> float:
    return ((2) if (y > 1.49) else (1)) if (x <= 0.49) else (0)


def dy() -> float:
    return -2


def dx() -> float:
    return 1


def create_model() -> Model:
    return (
        Model()
        .add_variable("y", initial_value=2.0)
        .add_variable("x", initial_value=0.0)
        .add_derived(
            "z",
            fn=z,
            args=["y", "x"],
        )
        .add_reaction(
            "dy",
            fn=dy,
            args=[],
            stoichiometry={"y": 1.0},
        )
        .add_reaction(
            "dx",
            fn=dx,
            args=[],
            stoichiometry={"x": 1.0},
        )
    )
