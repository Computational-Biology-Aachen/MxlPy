from mxlpy import Model


def s_conc(s: float, c: float) -> float:
    return s / c


def dc(c: float) -> float:
    return 0.5 * c


def create_model() -> Model:
    return (
        Model()
        .add_variable("c", initial_value=1.0)
        .add_variable("s", initial_value=2.0)
        .add_derived(
            "s_conc",
            fn=s_conc,
            args=["s", "c"],
        )
        .add_reaction(
            "dc",
            fn=dc,
            args=["c"],
            stoichiometry={"c": 1.0},
        )
    )
