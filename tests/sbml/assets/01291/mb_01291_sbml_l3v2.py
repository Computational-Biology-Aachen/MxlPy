from mxlpy import Model


def p1() -> float:
    return True


def p2() -> float:
    return False


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "p1",
            fn=p1,
            args=[],
        )
        .add_derived(
            "p2",
            fn=p2,
            args=[],
        )
    )
