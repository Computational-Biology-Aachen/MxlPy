from mxlpy import Model


def S4(k3: float, S2: float) -> float:
    return S2 * k3


def reaction1(S1: float, k1: float, S2: float) -> float:
    return S1 * S2 * k1


def reaction2(S3: float, k2: float) -> float:
    return S3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0e-5)
        .add_variable("S2", initial_value=1.5e-5)
        .add_variable("S3", initial_value=1.0e-5)
        .add_parameter("k1", value=150000.0)
        .add_parameter("k2", value=50.0)
        .add_parameter("k3", value=1.5)
        .add_parameter("compartment", value=10.0)
        .add_derived(
            "S4",
            fn=S4,
            args=["k3", "S2"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k1", "S2"],
            stoichiometry={"S1": -1.0, "S2": -1.0, "S3": 1.0},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3", "k2"],
            stoichiometry={"S3": -1.0, "S1": 1.0, "S2": 1.0},
        )
    )
