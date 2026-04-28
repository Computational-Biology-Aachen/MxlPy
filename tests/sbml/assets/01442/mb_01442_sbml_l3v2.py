from mxlpy import Derived, InitialAssignment, Model


def init_A_sr1() -> float:
    return 2


def init_B_sr1() -> float:
    return 2


def init_A_sr2() -> float:
    return 1


def init_B_sr2() -> float:
    return 1


def J0() -> float:
    return -1


def J0_stoich_A(A_sr1: float, A_sr2: float) -> float:
    return -1.0 * A_sr1 + 1.0 * A_sr2


def J0_stoich_B(B_sr1: float, B_sr2: float) -> float:
    return -1.0 * B_sr1 + 1.0 * B_sr2


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "A_sr1",
            initial_value=InitialAssignment(fn=init_A_sr1, args=[]),
        )
        .add_variable(
            "B_sr1",
            initial_value=InitialAssignment(fn=init_B_sr1, args=[]),
        )
        .add_variable(
            "A_sr2",
            initial_value=InitialAssignment(fn=init_A_sr2, args=[]),
        )
        .add_variable(
            "B_sr2",
            initial_value=InitialAssignment(fn=init_B_sr2, args=[]),
        )
        .add_variable("A", initial_value=1.0)
        .add_variable("B", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={
                "A": Derived(fn=J0_stoich_A, args=["A_sr1", "A_sr2"]),
                "B": Derived(fn=J0_stoich_B, args=["B_sr1", "B_sr2"]),
            },
        )
    )
