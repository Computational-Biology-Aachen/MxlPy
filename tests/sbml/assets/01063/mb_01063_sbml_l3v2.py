from mxlpy import Model


def reaction1(S3: float, kf: float, kr: float, S1: float, S2: float) -> float:
    return -S1 * kf + S2 * S3 * kr


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.5)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("kf", value=2.5)
        .add_parameter("kr", value=0.2)
        .add_parameter("C", value=0.95)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S3", "kf", "kr", "S1", "S2"],
            stoichiometry={"S2": -1.0, "S3": -1.0, "S1": 1.0},
        )
    )
