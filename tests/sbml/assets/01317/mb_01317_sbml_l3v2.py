from mxlpy import Model


def p2(time: float) -> float:
    return time + 1


def create_model() -> Model:
    return (
        Model()
        .add_parameter("p1", value=5.0)
        .add_derived(
            "p2",
            fn=p2,
            args=["time"],
        )
    )
