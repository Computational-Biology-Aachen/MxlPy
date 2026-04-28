from mxlpy import Derived, Model


def Xref(p1: float) -> float:
    return p1


def dp1() -> float:
    return 1


def J0(k1: float) -> float:
    return k1


def J0_stoich_X(Xref: float) -> float:
    return 1.0 * Xref


def create_model() -> Model:
    return (
        Model()
        .add_variable("p1", initial_value=1.0)
        .add_variable("X", initial_value=1.0)
        .add_parameter("k1", value=1.0)
        .add_parameter("default_compartment", value=1.0)
        .add_derived(
            "Xref",
            fn=Xref,
            args=["p1"],
        )
        .add_reaction(
            "dp1",
            fn=dp1,
            args=[],
            stoichiometry={"p1": 1.0},
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=["k1"],
            stoichiometry={"X": Derived(fn=J0_stoich_X, args=["Xref"])},
        )
    )
