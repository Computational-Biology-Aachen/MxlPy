import math

from mxlpy import Model


def P1() -> float:
    return math.pi


def P3() -> float:
    return 2


def P4() -> float:
    return 2


def P5() -> float:
    return 3


def P6() -> float:
    return 3


def P7(P1: float) -> float:
    return (2) if (math.pi == P1) else (3)


def P8(P1: float) -> float:
    return (2) if (math.pi != P1) else (3)


def P9() -> float:
    return 24


def P10() -> float:
    return 3


def P11() -> float:
    return 2


def P12() -> float:
    return 2


def P13() -> float:
    return 2


def P14() -> float:
    return 1.13949392732455


def P15() -> float:
    return -1.0229863836713


def P16() -> float:
    return 4.93315487558689


def P17() -> float:
    return 0.304520293447143


def P18() -> float:
    return 2.82831545788997


def P19() -> float:
    return 1.12099946637054


def P20() -> float:
    return 1.14109666064347


def P21() -> float:
    return -1.47112767430373


def P22() -> float:
    return math.asinh(99)


def P23() -> float:
    return 0.802881971289134


def P24() -> float:
    return -0.867300527694053


def P25() -> float:
    return 1.51330668842682


def P26() -> float:
    return 5.29834236561059


def P27() -> float:
    return 0.122561229016493


def P28() -> float:
    return math.e


def P29() -> float:
    return math.exp(math.e)


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "P1",
            fn=P1,
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
        .add_derived(
            "P6",
            fn=P6,
            args=[],
        )
        .add_derived(
            "P7",
            fn=P7,
            args=["P1"],
        )
        .add_derived(
            "P8",
            fn=P8,
            args=["P1"],
        )
        .add_derived(
            "P9",
            fn=P9,
            args=[],
        )
        .add_derived(
            "P10",
            fn=P10,
            args=[],
        )
        .add_derived(
            "P11",
            fn=P11,
            args=[],
        )
        .add_derived(
            "P12",
            fn=P12,
            args=[],
        )
        .add_derived(
            "P13",
            fn=P13,
            args=[],
        )
        .add_derived(
            "P14",
            fn=P14,
            args=[],
        )
        .add_derived(
            "P15",
            fn=P15,
            args=[],
        )
        .add_derived(
            "P16",
            fn=P16,
            args=[],
        )
        .add_derived(
            "P17",
            fn=P17,
            args=[],
        )
        .add_derived(
            "P18",
            fn=P18,
            args=[],
        )
        .add_derived(
            "P19",
            fn=P19,
            args=[],
        )
        .add_derived(
            "P20",
            fn=P20,
            args=[],
        )
        .add_derived(
            "P21",
            fn=P21,
            args=[],
        )
        .add_derived(
            "P22",
            fn=P22,
            args=[],
        )
        .add_derived(
            "P23",
            fn=P23,
            args=[],
        )
        .add_derived(
            "P24",
            fn=P24,
            args=[],
        )
        .add_derived(
            "P25",
            fn=P25,
            args=[],
        )
        .add_derived(
            "P26",
            fn=P26,
            args=[],
        )
        .add_derived(
            "P27",
            fn=P27,
            args=[],
        )
        .add_derived(
            "P28",
            fn=P28,
            args=[],
        )
        .add_derived(
            "P29",
            fn=P29,
            args=[],
        )
    )
