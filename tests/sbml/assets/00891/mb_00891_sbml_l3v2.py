from mxlpy import Model


def dk1(k3: float, time: float, k1: float) -> float:
    return -k1 * k3 * time


def dk2(k3: float, time: float, k1: float) -> float:
    return k1 * k3 * time


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=0.00015)
        .add_variable("k2", initial_value=0.0)
        .add_parameter("k3", value=1.0)
        .add_reaction(
            "dk1",
            fn=dk1,
            args=["k3", "time", "k1"],
            stoichiometry={"k1": 1.0},
        )
        .add_reaction(
            "dk2",
            fn=dk2,
            args=["k3", "time", "k1"],
            stoichiometry={"k2": 1.0},
        )
    )
