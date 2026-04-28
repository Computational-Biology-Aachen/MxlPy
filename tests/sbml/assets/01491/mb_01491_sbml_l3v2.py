from mxlpy import Model


def a() -> float:
    return 2


def b() -> float:
    return 4


def c() -> float:
    return 1


def d() -> float:
    return 1


def e() -> float:
    return 0


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "a",
            fn=a,
            args=[],
        )
        .add_derived(
            "b",
            fn=b,
            args=[],
        )
        .add_derived(
            "c",
            fn=c,
            args=[],
        )
        .add_derived(
            "d",
            fn=d,
            args=[],
        )
        .add_derived(
            "e",
            fn=e,
            args=[],
        )
    )
