from mxlpy import Derived, InitialAssignment, Model


def init_A1_sr() -> float:
    return 2


def init_A1_sr2() -> float:
    return 1


def J0() -> float:
    return -1


def J0_stoich_A(A1_sr2: float, A1_sr: float) -> float:
    return -1.0 * A1_sr - 1.0 * A1_sr2


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "A1_sr",
            initial_value=InitialAssignment(fn=init_A1_sr, args=[]),
        )
        .add_variable(
            "A1_sr2",
            initial_value=InitialAssignment(fn=init_A1_sr2, args=[]),
        )
        .add_variable("A", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"A": Derived(fn=J0_stoich_A, args=["A1_sr2", "A1_sr"])},
        )
    )
