from mxlpy import Derived, Model


def dXref() -> float:
    return 0.01


def J0(k1: float) -> float:
    return k1


def J0_stoich_X(Xref: float) -> float:
    return 1.0 * Xref


def create_model() -> Model:
    return (
        Model()
        .add_variable("k1", initial_value=1.0)
        .add_variable("Xref", initial_value=1.0)
        .add_variable("X", initial_value=0.0)
        .add_parameter("default_compartment", value=1.0)
        .add_reaction(
            "dXref",
            fn=dXref,
            args=[],
            stoichiometry={"Xref": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"X": Derived(fn=J0_stoich_X, args=["Xref"])},
        )
    )
