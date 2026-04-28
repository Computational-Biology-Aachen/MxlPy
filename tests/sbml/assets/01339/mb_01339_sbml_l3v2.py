from mxlpy import Model


def dk1() -> float:
    return 1


def J0(k1: float) -> float:
    return k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=-5.0)
        .add_variable("S1", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "dk1",
            fn=dk1,
            args=[],
            stoichiometry={"k1": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"S1": 1.0},
        )
    )
