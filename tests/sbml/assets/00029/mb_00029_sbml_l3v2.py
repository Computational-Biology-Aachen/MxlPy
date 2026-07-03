from mxlpy import KineticModelBuilder


def S1() -> float:
    return 7


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def create_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameter("compartment", value=1.0)
        .add_derived(
            "S1",
            fn=S1,
            args=[],
        )
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["compartment", "S1"],
        )
    )
