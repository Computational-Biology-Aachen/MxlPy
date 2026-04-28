from mxlpy import InitialAssignment, Model


def init_s_amount(c: float) -> float:
    return 2.0 * c


def s(s_amount: float, c: float) -> float:
    return s_amount / c


def dc(c: float) -> float:
    return 0.5 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable("c", initial_value=1.0)
        .add_variable(
            "s_amount",
            initial_value=InitialAssignment(fn=init_s_amount, args=["c"]),
        )
        .add_derived(
            "s",
            fn=s,
            args=["s_amount", "c"],
        )
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
