from mxlpy import Model


def S4(S1: float, k3: float) -> float:
    return S1 * k3


def S4_conc(C: float, S4: float) -> float:
    return S4 / C


def dS2(S3: float, S1: float, k1: float, k2: float) -> float:
    return -S1 * k1 + S3 * k2


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S2", initial_value=1.5e-5)
        .add_variable("S1", initial_value=1.0e-5)
        .add_variable("S3", initial_value=1.0e-5)
        .add_parameter("k1", value=0.015)
        .add_parameter("k2", value=0.5)
        .add_parameter("k3", value=1.5)
        .add_parameter("C", value=0.76)
        .add_derived(
            "S4",
            fn=S4,
            args=["S1", "k3"],
        )
        .add_derived(
            "S4_conc",
            fn=S4_conc,
            args=["C", "S4"],
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S3", "S1", "k1", "k2"],
            stoichiometry={"S2": 1.0},
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
            args=["S3", "k2"],
            stoichiometry={"S3": -1.0, "S1": 1.0},
        )
    )
