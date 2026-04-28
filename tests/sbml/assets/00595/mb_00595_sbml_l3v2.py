from mxlpy import Model


def reaction1(S1: float, reaction1_k: float) -> float:
    return S1 * reaction1_k


def reaction2(reaction2_k: float, S2: float) -> float:
    return S2 * reaction2_k


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.003)
        .add_variable("S2", initial_value=0.0)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_parameter("reaction1_k", value=1.0)
        .add_parameter("reaction2_k", value=2.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "reaction1_k"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["reaction2_k", "S2"],
            stoichiometry={"S2": -1.0, "S3": 1.0},
        )
    )
