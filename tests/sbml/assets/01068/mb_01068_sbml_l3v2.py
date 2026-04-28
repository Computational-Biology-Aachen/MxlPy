from mxlpy import Derived, InitialAssignment, Model


def init_generatedId_0(p1: float) -> float:
    return 2 * p1


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S2(compartment: float, generatedId_0: float) -> float:
    return 1.0 * compartment * generatedId_0


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "generatedId_0",
            initial_value=InitialAssignment(fn=init_generatedId_0, args=["p1"]),
        )
        .add_variable("S1", initial_value=1.5)
        .add_variable("S2", initial_value=0.0)
        .add_parameter("k1", value=1.5)
        .add_parameter("p1", value=1.0)
        .add_parameter("compartment", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["compartment", "S1"],
        )
        .add_derived(
            "S2_conc",
            fn=S2_conc,
            args=["compartment", "S2"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S2": Derived(
                    fn=reaction1_stoich_S2, args=["compartment", "generatedId_0"]
                ),
            },
        )
    )
