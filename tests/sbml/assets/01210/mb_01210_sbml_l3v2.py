from mxlpy import Model


def x() -> float:
    return 3


def create_model() -> Model:
    return Model().add_derived(
        "x",
        fn=x,
        args=[],
    )
