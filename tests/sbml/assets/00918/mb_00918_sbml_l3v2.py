from mxlpy import InitialAssignment, Model


def init_c() -> float:
    return 1.65


def dc(c: float) -> float:
    return 1.25 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "c",
            initial_value=InitialAssignment(fn=init_c, args=[]),
        )
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
