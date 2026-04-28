from mxlpy import InitialAssignment, Model


def init_C(k2: float) -> float:
    return (1 / 9) * k2


def init_S1_amount(C: float) -> float:
    return 1.0 * C


def init_S2_amount(C: float) -> float:
    return 1.5 * C


def S1(S1_amount: float, C: float) -> float:
    return S1_amount / C


def S2(C: float, S2_amount: float) -> float:
    return S2_amount / C


def reaction1(C: float, S1: float, k1: float) -> float:
    return C * S1 * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "C",
            initial_value=InitialAssignment(fn=init_C, args=["k2"]),
        )
        .add_variable(
            "S1_amount",
            initial_value=InitialAssignment(fn=init_S1_amount, args=["C"]),
        )
        .add_variable(
            "S2_amount",
            initial_value=InitialAssignment(fn=init_S2_amount, args=["C"]),
        )
        .add_parameter("k1", value=0.5)
        .add_parameter("k2", value=50.0)
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
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["C", "S1", "k1"],
            stoichiometry={"S1_amount": -1.0, "S2_amount": 1.0},
        )
    )
