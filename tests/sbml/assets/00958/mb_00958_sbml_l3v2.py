import math

from mxlpy import InitialAssignment, Model


def init_P1() -> float:
    return math.pi


def init_P3() -> float:
    return 2


def init_P4() -> float:
    return 2


def init_P5() -> float:
    return 3


def init_P6() -> float:
    return 3


def init_P7(P1: float) -> float:
    return (2) if (math.pi == P1) else (3)


def init_P8(P1: float) -> float:
    return (2) if (math.pi != P1) else (3)


def init_P9() -> float:
    return 24


def init_P10() -> float:
    return 3


def init_P11() -> float:
    return 2


def init_P12() -> float:
    return 2


def init_P13() -> float:
    return 2


def init_P14() -> float:
    return 1.13949392732455


def init_P15() -> float:
    return -1.0229863836713


def init_P16() -> float:
    return 4.93315487558689


def init_P17() -> float:
    return 0.304520293447143


def init_P18() -> float:
    return 2.82831545788997


def init_P19() -> float:
    return 1.12099946637054


def init_P20() -> float:
    return 1.14109666064347


def init_P21() -> float:
    return -1.47112767430373


def init_P22() -> float:
    return math.asinh(99)


def init_P23() -> float:
    return 0.802881971289134


def init_P24() -> float:
    return -0.867300527694053


def init_P25() -> float:
    return 1.51330668842682


def init_P26() -> float:
    return 5.29834236561059


def init_P27() -> float:
    return 0.122561229016493


def init_P28() -> float:
    return math.e


def init_P29() -> float:
    return math.exp(math.e)


def create_model() -> Model:
    return (
        Model()
        .add_parameter(
            "P1",
            value=InitialAssignment(fn=init_P1, args=[]),
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
        .add_parameter(
            "P6",
            value=InitialAssignment(fn=init_P6, args=[]),
        )
        .add_parameter(
            "P7",
            value=InitialAssignment(fn=init_P7, args=["P1"]),
        )
        .add_parameter(
            "P8",
            value=InitialAssignment(fn=init_P8, args=["P1"]),
        )
        .add_parameter(
            "P9",
            value=InitialAssignment(fn=init_P9, args=[]),
        )
        .add_parameter(
            "P10",
            value=InitialAssignment(fn=init_P10, args=[]),
        )
        .add_parameter(
            "P11",
            value=InitialAssignment(fn=init_P11, args=[]),
        )
        .add_parameter(
            "P12",
            value=InitialAssignment(fn=init_P12, args=[]),
        )
        .add_parameter(
            "P13",
            value=InitialAssignment(fn=init_P13, args=[]),
        )
        .add_parameter(
            "P14",
            value=InitialAssignment(fn=init_P14, args=[]),
        )
        .add_parameter(
            "P15",
            value=InitialAssignment(fn=init_P15, args=[]),
        )
        .add_parameter(
            "P16",
            value=InitialAssignment(fn=init_P16, args=[]),
        )
        .add_parameter(
            "P17",
            value=InitialAssignment(fn=init_P17, args=[]),
        )
        .add_parameter(
            "P18",
            value=InitialAssignment(fn=init_P18, args=[]),
        )
        .add_parameter(
            "P19",
            value=InitialAssignment(fn=init_P19, args=[]),
        )
        .add_parameter(
            "P20",
            value=InitialAssignment(fn=init_P20, args=[]),
        )
        .add_parameter(
            "P21",
            value=InitialAssignment(fn=init_P21, args=[]),
        )
        .add_parameter(
            "P22",
            value=InitialAssignment(fn=init_P22, args=[]),
        )
        .add_parameter(
            "P23",
            value=InitialAssignment(fn=init_P23, args=[]),
        )
        .add_parameter(
            "P24",
            value=InitialAssignment(fn=init_P24, args=[]),
        )
        .add_parameter(
            "P25",
            value=InitialAssignment(fn=init_P25, args=[]),
        )
        .add_parameter(
            "P26",
            value=InitialAssignment(fn=init_P26, args=[]),
        )
        .add_parameter(
            "P27",
            value=InitialAssignment(fn=init_P27, args=[]),
        )
        .add_parameter(
            "P28",
            value=InitialAssignment(fn=init_P28, args=[]),
        )
        .add_parameter(
            "P29",
            value=InitialAssignment(fn=init_P29, args=[]),
        )
    )
