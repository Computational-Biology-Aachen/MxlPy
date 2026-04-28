from mxlpy import Model


def __J0(k1: float) -> float:
    return k1


def __J1(S1: float, S2: float, k2: float) -> float:
    return S1 * k2 / S2


def __J2(k3: float, default_compartment: float, S2: float) -> float:
    return S2 * k3 / default_compartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1", initial_value=0.0)
        .add_variable("S2", initial_value=0.00100000000000000)
        .add_parameter("k1", value=1.00000000000000)
        .add_parameter("k2", value=3.00000000000000)
        .add_parameter("k3", value=1.40000000000000)
        .add_parameter("S1conv", value=3.00000000000000)
        .add_parameter("modelconv", value=4.00000000000000)
        .add_parameter("default_compartment", value=1.00000000000000)
        .add_reaction(
            "__J0",
            fn=__J0,
            args=["k1"],
            stoichiometry={"S1": 1.00000000000000},
        )
        .add_reaction(
            "__J1",
            fn=__J1,
            args=["S1", "S2", "k2"],
            stoichiometry={"S1": -2.00000000000000, "S2": 3.00000000000000},
        )
        .add_reaction(
            "__J2",
            fn=__J2,
            args=["k3", "default_compartment", "S2"],
            stoichiometry={"S2": -1.00000000000000},
        )
    )
