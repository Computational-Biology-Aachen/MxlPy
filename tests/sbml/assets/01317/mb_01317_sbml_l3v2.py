from mxlpy import KineticModelBuilder


def p2(time: float) -> float:
    return time + 1


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter("p1", value=5.0)
        .add_derived(
            "p2",
            fn=p2,
            args=["time"],
        )
    )
