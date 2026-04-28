from mxlpy import Derived, InitialAssignment, Model


def init_Q(Xref: float) -> float:
    return Xref


def Z(Xref: float) -> float:
    return Xref


def dY(Xref: float) -> float:
    return Xref


def J0(k1: float) -> float:
    return k1


def J0_stoich_X(Xref: float) -> float:
    return 1.0 * Xref


def create_model() -> Model:
    return (
        Model()
        .add_variable("Y", initial_value=0.0)
        .add_variable("Xref", initial_value=1.0)
        .add_variable("X", initial_value=0.0)
        .add_parameter(
            "Q",
            value=InitialAssignment(fn=init_Q, args=["Xref"]),
        )
        .add_parameter("k1", value=1.0)
        .add_parameter("default_compartment", value=1.0)
        .add_derived(
            "Z",
            fn=Z,
            args=["Xref"],
        )
        .add_reaction(
            "dY",
            fn=dY,
            args=["Xref"],
            stoichiometry={"Y": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"X": Derived(fn=J0_stoich_X, args=["Xref"])},
        )
    )
