from mxlpy import Model


def dS1(S3: float, k1: float, S1: float, k3: float, S2: float, k2: float) -> float:
    return -S1 * S2 * k1 + S3 * k2 + S3 * k3


def dS2(S3: float, S1: float, k1: float, S2: float, k2: float) -> float:
    return -S1 * S2 * k1 + S3 * k2


def dS3(S3: float, k1: float, S1: float, k3: float, S2: float, k2: float) -> float:
    return S1 * S2 * k1 - S3 * k2 - S3 * k3


def dS4(S3: float, k3: float) -> float:
    return S3 * k3


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.02)
        .add_variable("S2", initial_value=0.02)
        .add_variable("S3", initial_value=0.0)
        .add_variable("S4", initial_value=0.0)
        .add_parameter("k1", value=100.0)
        .add_parameter("k2", value=0.9)
        .add_parameter("k3", value=0.7)
        .add_reaction(
            "dS1",
            fn=dS1,
            args=["S3", "k1", "S1", "k3", "S2", "k2"],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S3", "S1", "k1", "S2", "k2"],
            stoichiometry={"S2": 1.0},
        )
        .add_reaction(
            "dS3",
            fn=dS3,
            args=["S3", "k1", "S1", "k3", "S2", "k2"],
            stoichiometry={"S3": 1.0},
        )
        .add_reaction(
            "dS4",
            fn=dS4,
            args=["S3", "k3"],
            stoichiometry={"S4": 1.0},
        )
    )
