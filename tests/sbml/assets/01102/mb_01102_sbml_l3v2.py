from mxlpy import Derived, InitialAssignment, Model


def init_Xref() -> float:
    return 3


def X_conc(X: float, default_compartment: float) -> float:
    return X / default_compartment


def J0(k1: float) -> float:
    return k1


def J0_stoich_X(default_compartment: float, Xref: float) -> float:
    return 1.0 * Xref * default_compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "Xref",
            initial_value=InitialAssignment(fn=init_Xref, args=[]),
        )
        .add_variable("X", initial_value=0.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("default_compartment", value=1.0)
        .add_derived(
            "X_conc",
            fn=X_conc,
            args=["X", "default_compartment"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={
                "X": Derived(fn=J0_stoich_X, args=["default_compartment", "Xref"])
            },
        )
    )
