from mxlpy import Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def dS1() -> float:
    return 7


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0)
        .add_parameter("compartment", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["compartment", "S1"],
        )
        .add_reaction(
            "dS1",
            fn=dS1,
            args=[],
            stoichiometry={"S1": "compartment"},
        )
    )
