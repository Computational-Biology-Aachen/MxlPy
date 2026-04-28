from mxlpy import Model


def P1() -> float:
    return 6.02214179e23


def P2() -> float:
    return 6.02214179e23


def P3() -> float:
    return 6.02214179e23


def P4() -> float:
    return 6.02214179e23


def P5() -> float:
    return 4.17899999992689e18


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "P1",
            fn=P1,
            args=[],
        )
        .add_derived(
            "P2",
            fn=P2,
            args=[],
        )
        .add_derived(
            "P3",
            fn=P3,
            args=[],
        )
        .add_derived(
            "P4",
            fn=P4,
            args=[],
        )
        .add_derived(
            "P5",
            fn=P5,
            args=[],
        )
    )
