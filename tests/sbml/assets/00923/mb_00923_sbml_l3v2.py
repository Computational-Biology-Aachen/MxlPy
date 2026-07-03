from mxlpy import KineticModelBuilder


def k1(k2: float) -> float:
    return 4 * k2


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter("k2", value=0.3)
        .add_derived(
            "k1",
            fn=k1,
            args=["k2"],
        )
    )
