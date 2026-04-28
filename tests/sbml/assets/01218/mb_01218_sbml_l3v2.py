from mxlpy import InitialAssignment, Model


def init_a() -> float:
    return 2


def init_c(b: float) -> float:
    return b + 1


def init_e(d: float) -> float:
    return d + 1


def b(a: float) -> float:
    return a + 1


def d(c: float) -> float:
    return c + 1


def f(e: float) -> float:
    return e + 1


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "a",
            value=InitialAssignment(fn=init_a, args=[]),
        )
        .add_parameter(
            "c",
            value=InitialAssignment(fn=init_c, args=["b"]),
        )
        .add_parameter(
            "e",
            value=InitialAssignment(fn=init_e, args=["d"]),
        )
        .add_derived(
            "b",
            fn=b,
            args=["a"],
        )
        .add_derived(
            "d",
            fn=d,
            args=["c"],
        )
        .add_derived(
            "f",
            fn=f,
            args=["e"],
        )
    )
