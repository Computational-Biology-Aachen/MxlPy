from mxlpy import InitialAssignment, Model


def init_C(p1: float) -> float:
    return 2 * p1


def init_S1_amount(C: float) -> float:
    return 0.01 * C


def init_S2_amount(C: float) -> float:
    return 0.02 * C


def init_S3_amount(C: float) -> float:
    return 0.01 * C


def S1(S1_amount: float, C: float) -> float:
    return S1_amount / C


def S2(C: float, S2_amount: float) -> float:
    return S2_amount / C


def S3(S3_amount: float, C: float) -> float:
    return S3_amount / C


def reaction1(C: float, S1: float, k1: float, S2: float) -> float:
    return C * S1 * S2 * k1


def reaction2(S3: float, C: float, k2: float) -> float:
    return C * S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "C",
            initial_value=InitialAssignment(fn=init_C, args=["p1"]),
        )
        .add_variable(
            "S1_amount",
            initial_value=InitialAssignment(fn=init_S1_amount, args=["C"]),
        )
        .add_variable(
            "S2_amount",
            initial_value=InitialAssignment(fn=init_S2_amount, args=["C"]),
        )
        .add_variable(
            "S3_amount",
            initial_value=InitialAssignment(fn=init_S3_amount, args=["C"]),
        )
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("p1", value=0.25)
        .add_derived(
            "S1",
            fn=S1,
            args=["S1_amount", "C"],
        )
        .add_derived(
            "S2",
            fn=S2,
            args=["C", "S2_amount"],
        )
        .add_derived(
            "S3",
            fn=S3,
            args=["S3_amount", "C"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["C", "S1", "k1", "S2"],
            stoichiometry={"S1_amount": -1.0, "S2_amount": -1.0, "S3_amount": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "C", "k2"],
            stoichiometry={"S1_amount": 1.0, "S2_amount": 1.0, "S3_amount": -1.0},
        )
    )
