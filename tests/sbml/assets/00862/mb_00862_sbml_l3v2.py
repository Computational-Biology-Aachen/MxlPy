from mxlpy import Model


def reaction1(time: float, S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1 * time


def reaction2(S3: float, time: float, k2: float) -> float:
    return S3 * k2 * time


def reaction3(S3: float, time: float, k3: float) -> float:
    return S3 * k3 * time


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=2.0)
        .add_variable("S2", initial_value=2.0)
        .add_variable("S3", initial_value=0.0)
        .add_variable("S4", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("k2", value=0.9)
        .add_parameter("k3", value=0.7)
        .add_parameter("C", value=0.002)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["time", "S1", "k1", "S2"],
            stoichiometry={"S1": -1.0, "S2": -1.0, "S3": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "time", "k2"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S2": 1.0},
        )
        .add_reaction(
            "reaction3",
            fn=reaction3,
            args=["S3", "time", "k3"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S4": 1.0},
        )
    )
