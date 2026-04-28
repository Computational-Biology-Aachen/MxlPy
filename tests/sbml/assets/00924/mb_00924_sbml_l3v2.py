from mxlpy import Model


def k1(k2: float) -> float:
    return 3 * k2


def create_model() -> Model:
    return (
        Model()
        .add_parameter("k2", value=0.3)
        .add_derived(
            "k1",
            fn=k1,
            args=["k2"],
        )
    )
