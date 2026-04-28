from mxlpy import Model


def reaction1(S3: float, kr: float, kf: float, S1: float, S2: float) -> float:
    return -S1 * S2 * kf + S3 * kr


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.5)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("kf", value=1.1)
        .add_parameter("kr", value=0.09)
        .add_parameter("C", value=2.3)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S3", "kr", "kf", "S1", "S2"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S2": 1.0},
        )
    )
