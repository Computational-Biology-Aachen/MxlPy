from mxlpy import Derived, Model


def J0() -> float:
    return -1


def J0_stoich_A() -> float:
    return 0


def create_model() -> Model:
    return (
        Model()
        .add_variable("A", initial_value=1.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"A": Derived(fn=J0_stoich_A, args=[])},
        )
    )
