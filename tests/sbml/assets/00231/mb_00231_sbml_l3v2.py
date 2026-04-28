from mxlpy import Derived, Model


def S1_conc(compartment: float, S1: float) -> float:
    return S1 / compartment


def S2_conc(compartment: float, S2: float) -> float:
    return S2 / compartment


def S3_conc(S3: float, compartment: float) -> float:
    return S3 / compartment


def reaction1(S2_conc: float, S1_conc: float, k1: float) -> float:
    return S1_conc * S2_conc * k1


def reaction1_stoich_S3(compartment: float) -> float:
    return 1.0 * compartment


def reaction2(S3_conc: float, k2: float) -> float:
    return S3_conc * k2


def reaction2_stoich_S3(compartment: float) -> float:
    return -1.0 * compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S3", initial_value=1.5)
        .add_parameter("k1", value=7.5)
        .add_parameter("k2", value=0.3)
        .add_parameter("compartment", value=1.0)
        .add_parameter("S1", value=1.0)
        .add_parameter("S2", value=2.0)
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
            args=["S2_conc", "S1_conc", "k1"],
            stoichiometry={"S3": Derived(fn=reaction1_stoich_S3, args=["compartment"])},
        )
        .add_reaction(
            "reaction2",
            fn=reaction2,
            args=["S3_conc", "k2"],
            stoichiometry={"S3": Derived(fn=reaction2_stoich_S3, args=["compartment"])},
        )
    )
