from mxlpy import Model


def S3(k1: float, S2: float) -> float:
    return S2 * k1


def S3_conc(S3: float, C: float) -> float:
    return S3 / C


def reaction1(S1: float, k2: float) -> float:
    return S1 * k2


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S2", initial_value=0.00015)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=5.0)
        .add_parameter("C", value=0.08)
        .add_derived(
            "S3",
            fn=S3,
            args=["k1", "S2"],
        )
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "C"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1", "k2"],
            stoichiometry={"S1": -1.0, "S2": 1.0},
        )
    )
