from mxlpy import KineticModelBuilder


def p1(time: float) -> float:
    return max(-5, -time)


def p2(time: float) -> float:
    return time


def p3(time: float) -> float:
    return max(5, time)


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_derived(
            "p1",
            fn=p1,
            args=["time"],
        )
        .add_derived(
            "p2",
            fn=p2,
            args=["time"],
        )
        .add_derived(
            "p3",
            fn=p3,
            args=["time"],
        )
    )
