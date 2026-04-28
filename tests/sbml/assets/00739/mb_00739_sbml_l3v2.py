from mxlpy import Model


def S4(S3: float, p1: float) -> float:
    return S3 / (p1 + 1)


def S5(p1: float, S4: float) -> float:
    return S4 * p1


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def reaction2(S5: float, k2: float) -> float:
    return S5 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.0)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("k1", value=0.1)
        .add_parameter("k2", value=0.15)
        .add_parameter("p1", value=2.5)
        .add_parameter("C", value=2.5)
        .add_derived(
            "S4",
            fn=S4,
            args=["S3", "p1"],
        )
        .add_derived(
            "S5",
            fn=S5,
            args=["p1", "S4"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S3": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S5", "k2"],
            stoichiometry={"S3": -1.0, "S2": 1.0},
        )
    )
