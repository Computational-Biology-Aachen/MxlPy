from mxlpy import Model


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=2.0)
        .add_parameter("C", value=2.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
    )
