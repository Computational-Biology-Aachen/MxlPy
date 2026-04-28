import math

from mxlpy import InitialAssignment, Model


def init_P1() -> float:
    return 3.7


def init_P2() -> float:
    return 1


def init_P3() -> float:
    return 1


def init_P4() -> float:
    return math.pi


def init_P5() -> float:
    return 1.0471975511966


def init_P6() -> float:
    return (1 / 2) * math.pi


def init_P7() -> float:
    return -0.523598775598299


def init_P8() -> float:
    return 1.22777238637419


def init_P9() -> float:
    return -1.43067687253053


def init_P10() -> float:
    return 1


def init_P11() -> float:
    return 4


def init_P12() -> float:
    return -4


def init_P13() -> float:
    return -0.947721602131112


def init_P14() -> float:
    return 0.975897449330606


def init_P15() -> float:
    return 1


def init_P16() -> float:
    return math.e


def init_P17() -> float:
    return 2.15976625378492


def init_P18() -> float:
    return -5


def init_P19() -> float:
    return 9


def init_P20() -> float:
    return -1.6094379124341


def init_P21() -> float:
    return 0


def init_P22() -> float:
    return -1.6094379124341 / math.log(10)


def init_P23() -> float:
    return 0


def init_P24() -> float:
    return 1


def init_P25() -> float:
    return 4


def init_P26() -> float:
    return 5.5622817277544


def init_P27() -> float:
    return 16


def init_P28() -> float:
    return 9.61


def init_P29() -> float:
    return 2


def init_P30() -> float:
    return 2.72029410174709


def init_P31() -> float:
    return 0.863209366648874


def init_P32() -> float:
    return 0


def init_P33() -> float:
    return 0.373876664830236


def init_P34() -> float:
    return 0


def init_P35() -> float:
    return 2.01433821447683


def init_P36() -> float:
    return -math.tan(6)


def init_P37() -> float:
    return 3


def init_P38() -> float:
    return -1


def init_P39() -> float:
    return 5 / 2


def init_P40() -> float:
    return 24


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
        .add_parameter(
            "P6",
            value=InitialAssignment(fn=init_P6, args=[]),
        )
        .add_parameter(
            "P7",
            value=InitialAssignment(fn=init_P7, args=[]),
        )
        .add_parameter(
            "P8",
            value=InitialAssignment(fn=init_P8, args=[]),
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
        .add_parameter(
            "P30",
            value=InitialAssignment(fn=init_P30, args=[]),
        )
        .add_parameter(
            "P31",
            value=InitialAssignment(fn=init_P31, args=[]),
        )
        .add_parameter(
            "P32",
            value=InitialAssignment(fn=init_P32, args=[]),
        )
        .add_parameter(
            "P33",
            value=InitialAssignment(fn=init_P33, args=[]),
        )
        .add_parameter(
            "P34",
            value=InitialAssignment(fn=init_P34, args=[]),
        )
        .add_parameter(
            "P35",
            value=InitialAssignment(fn=init_P35, args=[]),
        )
        .add_parameter(
            "P36",
            value=InitialAssignment(fn=init_P36, args=[]),
        )
        .add_parameter(
            "P37",
            value=InitialAssignment(fn=init_P37, args=[]),
        )
        .add_parameter(
            "P38",
            value=InitialAssignment(fn=init_P38, args=[]),
        )
        .add_parameter(
            "P39",
            value=InitialAssignment(fn=init_P39, args=[]),
        )
        .add_parameter(
            "P40",
            value=InitialAssignment(fn=init_P40, args=[]),
        )
    )
