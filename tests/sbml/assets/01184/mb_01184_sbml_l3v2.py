from mxlpy import Derived, InitialAssignment, Model


def init_k1(y: float) -> float:
    return y


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def __J0(k1: float) -> float:
    return k1


def __J0_stoich_S1(C: float) -> float:
    return 1.0 * C


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=1.0)
        .add_parameter(
            "k1",
            value=InitialAssignment(fn=init_k1, args=["y"]),
        )
        .add_parameter("y", value=3.0)
        .add_parameter("C", value=1.0)
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_reaction(
            "__J0",
            fn=__J0,
            args=["k1"],
            stoichiometry={"S1": Derived(fn=__J0_stoich_S1, args=["C"])},
        )
    )
