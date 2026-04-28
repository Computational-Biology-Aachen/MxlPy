from mxlpy import Derived, Model


def S1_conc(comp: float, S1: float) -> float:
    return S1 / comp


def S3_conc(S3: float, comp: float) -> float:
    return S3 / comp


def dcomp() -> float:
    return 1


def __J0(comp: float, S3_conc: float) -> float:
    return (1 / 10) * S3_conc / comp


def __J0_stoich_S1(comp: float) -> float:
    return -1.0 * comp


def create_model() -> Model:
    return (
        Model()
        .add_variable("comp", initial_value=5.0)
        .add_variable("S1", initial_value=1.0)
        .add_parameter("S3", value=4.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["comp", "S1"],
        )
        .add_derived(
            "S3_conc",
            fn=S3_conc,
            args=["S3", "comp"],
        )
        .add_reaction(
            "dcomp",
            fn=dcomp,
            args=[],
            stoichiometry={"comp": 1.0},
        )
        .add_reaction(
            "__J0",
            fn=__J0,
            args=["comp", "S3_conc"],
            stoichiometry={"S1": Derived(fn=__J0_stoich_S1, args=["comp"])},
        )
    )
