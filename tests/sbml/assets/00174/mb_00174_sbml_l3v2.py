from mxlpy import Model


def S1() -> float:
    return 7


def create_model() -> Model:
    return Model().add_derived(
        "S1",
        fn=S1,
        args=[],
    )
