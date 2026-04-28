from mxlpy import InitialAssignment, Model


def init_a(b: float) -> float:
    return b + 1


def init_c(d: float) -> float:
    return d + 1


def init_e(f: float) -> float:
    return f + 1


def b(c: float) -> float:
    return c + 1


def d(e: float) -> float:
    return e + 1


def f() -> float:
    return 2


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "a",
            value=InitialAssignment(fn=init_a, args=["b"]),
        )
        .add_parameter(
            "c",
            value=InitialAssignment(fn=init_c, args=["d"]),
        )
        .add_parameter(
            "e",
            value=InitialAssignment(fn=init_e, args=["f"]),
        )
        .add_derived(
            "b",
            fn=b,
            args=["c"],
        )
        .add_derived(
            "d",
            fn=d,
            args=["e"],
        )
        .add_derived(
            "f",
            fn=f,
            args=[],
        )
    )
