from mxlpy import InitialAssignment, Model


def init_S1(p1: float) -> float:
    return 2 * p1


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["p1"]),
        )
        .add_variable("S2", initial_value=0.0002)
        .add_variable("S3", initial_value=0.0001)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("p1", value=1.25e-5)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1", "S2"],
            stoichiometry={"S1": -1.0, "S2": -1.0, "S3": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "k2"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S2": 1.0},
        )
    )
