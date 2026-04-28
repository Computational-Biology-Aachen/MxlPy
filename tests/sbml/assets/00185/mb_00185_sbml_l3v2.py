from mxlpy import InitialAssignment, Model


def init_S1(k1: float) -> float:
    return 1.33333333333333 * k1


def init_S2(k2: float) -> float:
    return 3.0e-17 * k2


def S3(k1: float, S2: float) -> float:
    return S2 * k1


def dS1(S1: float, k2: float) -> float:
    return -S1 * k2


def dS2(S1: float, k2: float) -> float:
    return S1 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1",
            initial_value=InitialAssignment(fn=init_S1, args=["k1"]),
        )
        .add_variable(
            "S2",
            initial_value=InitialAssignment(fn=init_S2, args=["k2"]),
        )
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=50.0)
        .add_derived(
            "S3",
            fn=S3,
            args=["k1", "S2"],
        )
        .add_reaction(
            "dS1",
            fn=dS1,
            args=["S1", "k2"],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "dS2",
            fn=dS2,
            args=["S1", "k2"],
            stoichiometry={"S2": 1.0},
        )
    )
