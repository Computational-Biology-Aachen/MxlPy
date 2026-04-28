from mxlpy import Derived, Model


def Xref(X: float) -> float:
    return X


def J0(k1: float) -> float:
    return k1


def J0_stoich_X(Xref: float) -> float:
    return 1.0 * Xref


def J1(Y: float, k1: float) -> float:
    return Y * k1


def create_model() -> Model:
    return (
        Model()
        .add_variable("X", initial_value=1.0)
        .add_variable("Y", initial_value=1.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("default_compartment", value=1.0)
        .add_derived(
            "Xref",
            fn=Xref,
            args=["X"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"X": Derived(fn=J0_stoich_X, args=["Xref"])},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=["Y", "k1"],
            stoichiometry={"Y": 1.0},
        )
    )
