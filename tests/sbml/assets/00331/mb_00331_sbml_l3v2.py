from mxlpy import Model


def dS3(k1: float, k2: float) -> float:
    return k1 * k2


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("S3", initial_value=0.1)
        .add_variable("S1", initial_value=0.15)
        .add_variable("S2", initial_value=0.1)
        .add_parameter("k1", value=1.75)
        .add_parameter("k2", value=0.015)
        .add_parameter("compartment", value=1.0)
        .add_reaction(
            "dS3",
            fn=dS3,
            args=["k1", "k2"],
            stoichiometry={"S3": 1.0},
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
