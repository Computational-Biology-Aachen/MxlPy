from mxlpy import Derived, InitialAssignment, Model


def init_S1_degrade() -> float:
    return 3


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def dS1_create() -> float:
    return 1


def J0() -> float:
    return 1


def J0_stoich_S1(C: float, S1_degrade: float) -> float:
    return -1.0 * C * S1_degrade


def J1() -> float:
    return 1


def J1_stoich_S1(C: float, S1_create: float) -> float:
    return 1.0 * C * S1_create


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1_degrade",
            initial_value=InitialAssignment(fn=init_S1_degrade, args=[]),
        )
        .add_variable("S1_create", initial_value=1.0)
        .add_variable("S1", initial_value=2.0)
        .add_parameter("C", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_reaction(
            "dS1_create",
            fn=dS1_create,
            args=[],
            stoichiometry={"S1_create": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["C", "S1_degrade"])},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=[],
            stoichiometry={"S1": Derived(fn=J1_stoich_S1, args=["C", "S1_create"])},
        )
    )
