from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def S3_conc(S3: float, compartment: float) -> float:
    return S3 / compartment


def reaction1(S1_conc: float, reaction1_k: float) -> float:
    return S1_conc * reaction1_k


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S2(compartment: float) -> float:
    return 1.0 * compartment


def reaction2(S2_conc: float, reaction2_k: float) -> float:
    return S2_conc * reaction2_k


def reaction2_stoich_S2(compartment: float) -> float:
    return -1.0 * compartment


def reaction2_stoich_S3(compartment: float) -> float:
    return 1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0003)
        .add_variable("S2", initial_value=0.0)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("compartment", value=1.0)
        .add_parameter("reaction1_k", value=1.0)
        .add_parameter("reaction2_k", value=2.0)
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
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "compartment"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "reaction1_k"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S2": Derived(fn=reaction1_stoich_S2, args=["compartment"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S2_conc", "reaction2_k"],
            stoichiometry={
                "S2": Derived(fn=reaction2_stoich_S2, args=["compartment"]),
                "S3": Derived(fn=reaction2_stoich_S3, args=["compartment"]),
            },
        )
    )
