from mxlpy import KineticModelBuilder


def dc(c: float) -> float:
    return 0.5 * c


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("c", initial_value=0.015)
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
