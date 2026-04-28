from mxlpy import Model


def S4(p1: float, S2: float) -> float:
    return S2 * p1


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.01)
        .add_variable("S2", initial_value=0.02)
        .add_variable("S3", initial_value=0.015)
        .add_parameter("k1", value=75.0)
        .add_parameter("k2", value=0.025)
        .add_parameter("p1", value=0.75)
        .add_parameter("C", value=0.86)
        .add_derived(
            "S4",
            fn=S4,
            args=["p1", "S2"],
        )
        .add_derived(
            "S4_conc",
            fn=S4_conc,
            args=["C", "S4"],
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
