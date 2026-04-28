from mxlpy import Model


def a() -> float:
    return 3


def b() -> float:
    return 4


def c() -> float:
    return 5


def d() -> float:
    return 6


def e() -> float:
    return 7


def f() -> float:
    return 8


def g() -> float:
    return 9


def h() -> float:
    return 10


def i() -> float:
    return 11


def j() -> float:
    return 12


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
        .add_derived(
            "f",
            fn=f,
            args=[],
        )
        .add_derived(
            "g",
            fn=g,
            args=[],
        )
        .add_derived(
            "h",
            fn=h,
            args=[],
        )
        .add_derived(
            "i",
            fn=i,
            args=[],
        )
        .add_derived(
            "j",
            fn=j,
            args=[],
        )
    )
