from mxlpy import Model


def reaction1(C: float, S1: float, k1: float, S2: float) -> float:
    return C * S1 * S2 * k1


def reaction2(S3: float, C: float, k2: float) -> float:
    return C * S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("C", value=1.0)
        .add_parameter("S1", value=0.0001)
        .add_parameter("S2", value=0.0002)
        .add_parameter("S3", value=0.0001)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["C", "S1", "k1", "S2"],
            stoichiometry={},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "C", "k2"],
            stoichiometry={},
        )
    )
