from mxlpy import KineticModelBuilder


def reaction1(S1: float, k1: float) -> float:
    return S1 * k1


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("S1", initial_value=0.00015)
        .add_variable("S2", initial_value=0.0001)
        .add_parameter("k1", value=1.0)
        .add_parameter("compartment", value=1.0)
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
