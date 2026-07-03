from mxlpy import KineticModelBuilder


def x() -> float:
    return 3


def create_model() -> KineticModelBuilder:
    return KineticModelBuilder().add_derived(
        "x",
        fn=x,
        args=[],
    )
