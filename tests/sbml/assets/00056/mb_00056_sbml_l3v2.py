from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S3_conc(S3: float, compartment1: float) -> float:
    return S3 / compartment1


def reaction1(S1_conc: float, k1: float) -> float:
    return S1_conc * k1


def reaction1_stoich_S1(compartment: float) -> float:
    return -1.0 * compartment


def reaction1_stoich_S3(compartment1: float) -> float:
    return 1.0 * compartment1


def reaction2(S1_conc: float, S3_conc: float, k2: float) -> float:
    return k2 * (-S1_conc + S3_conc)


def reaction2_stoich_S3(compartment1: float) -> float:
    return -1.0 * compartment1


def reaction2_stoich_S1(compartment: float) -> float:
    return 1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_variable("S3", initial_value=0.0)
        .add_parameter("k1", value=0.75)
        .add_parameter("k2", value=0.25)
        .add_parameter("compartment", value=1.0)
        .add_parameter("compartment1", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["compartment", "S1"],
        )
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "compartment1"],
        )
        .add_reaction(
            "reaction1",
            fn=reaction1,
            args=["S1_conc", "k1"],
            stoichiometry={
                "S1": Derived(fn=reaction1_stoich_S1, args=["compartment"]),
                "S3": Derived(fn=reaction1_stoich_S3, args=["compartment1"]),
            },
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S1_conc", "S3_conc", "k2"],
            stoichiometry={
                "S3": Derived(fn=reaction2_stoich_S3, args=["compartment1"]),
                "S1": Derived(fn=reaction2_stoich_S1, args=["compartment"]),
            },
        )
    )
