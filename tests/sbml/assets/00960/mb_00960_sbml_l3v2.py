from mxlpy import InitialAssignment, Model


def init_P1() -> float:
    return 6.02214179e23


def init_P2() -> float:
    return 6.02214179e23


def init_P3() -> float:
    return 6.02214179e23


def init_P4() -> float:
    return 6.02214179e23


def init_P5() -> float:
    return 4.17899999992689e18


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "P1",
            value=InitialAssignment(fn=init_P1, args=[]),
        )
        .add_parameter(
            "P2",
            value=InitialAssignment(fn=init_P2, args=[]),
        )
        .add_parameter(
            "P3",
            value=InitialAssignment(fn=init_P3, args=[]),
        )
        .add_parameter(
            "P4",
            value=InitialAssignment(fn=init_P4, args=[]),
        )
        .add_parameter(
            "P5",
            value=InitialAssignment(fn=init_P5, args=[]),
        )
    )
