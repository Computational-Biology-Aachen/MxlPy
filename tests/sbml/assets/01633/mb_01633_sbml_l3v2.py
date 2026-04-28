from mxlpy import Derived, InitialAssignment, Model


def init_S1_degrade() -> float:
    return 3


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def S2_conc(C: float, S2: float) -> float:
    return S2 / C


def dS2_create() -> float:
    return 1


def J0() -> float:
    return 1


def J0_stoich_S1(C: float, S1_degrade: float) -> float:
    return -1.0 * C * S1_degrade


def J0_stoich_S2(C: float, S2_create: float) -> float:
    return 1.0 * C * S2_create


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S1_degrade",
            initial_value=InitialAssignment(fn=init_S1_degrade, args=[]),
        )
        .add_variable("S2_create", initial_value=1.0)
        .add_variable("S1", initial_value=30.0)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_derived(
            "S2_conc",
            fn=S2_conc,
            args=["C", "S2"],
        )
        .add_reaction(
            "dS2_create",
            fn=dS2_create,
            args=[],
            stoichiometry={"S2_create": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={
                "S1": Derived(fn=J0_stoich_S1, args=["C", "S1_degrade"]),
                "S2": Derived(fn=J0_stoich_S2, args=["C", "S2_create"]),
            },
        )
    )
