from mxlpy import Model


def dk1(k3: float, p1: float, p2: float, k2: float) -> float:
    return p1 * (k2 * p2 + k3)


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=1.7)
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=2.0)
        .add_variable("S3", initial_value=1.0)
        .add_parameter("k2", value=0.3)
        .add_parameter("k3", value=-0.1)
        .add_parameter("p1", value=1.0)
        .add_parameter("p2", value=1.0)
        .add_parameter("C", value=1.25)
        .add_reaction(
            "dk1",
            fn=dk1,
            args=["k3", "p1", "p2", "k2"],
            stoichiometry={"k1": 1.0},
        )
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
