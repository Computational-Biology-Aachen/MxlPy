from mxlpy import Model


def p1(time: float) -> float:
    return time % 2


def create_model() -> Model:
    return Model().add_derived(
        "p1",
        fn=p1,
        args=["time"],
    )
