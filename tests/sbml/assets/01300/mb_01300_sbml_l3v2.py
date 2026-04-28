from mxlpy import InitialAssignment, Model


def init_p1(J0: float) -> float:
    return J0


def J0() -> float:
    return 3


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "p1",
            initial_value=InitialAssignment(fn=init_p1, args=["J0"]),
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={},
        )
    )
