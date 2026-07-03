from mxlpy import KineticModelBuilder


def p1(time: float) -> float:
    return time % 2


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_derived(
        "p1",
        fn=p1,
        args=["time"],
    )
