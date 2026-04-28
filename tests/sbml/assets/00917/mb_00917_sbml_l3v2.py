from mxlpy import InitialAssignment, Model


def init_c() -> float:
    return 0.733333333333333


def dc(c: float) -> float:
    return 0.5 * c


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
