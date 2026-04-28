import math

from mxlpy import Model


def P2() -> float:
    return 1


def P3() -> float:
    return 1


def P4() -> float:
    return math.pi


def P5() -> float:
    return 1.0471975511966


def P6() -> float:
    return (1 / 2) * math.pi


def P7() -> float:
    return -0.523598775598299


def P8() -> float:
    return 1.22777238637419


def P9() -> float:
    return -1.43067687253053


def P10() -> float:
    return 1


def P11() -> float:
    return 4


def P12() -> float:
    return -4


def P13() -> float:
    return -0.947721602131112


def P14() -> float:
    return 0.975897449330606


def P15() -> float:
    return 1


def P16() -> float:
    return math.e


def P17() -> float:
    return 2.15976625378492


def P18() -> float:
    return -5


def P19() -> float:
    return 9


def P20() -> float:
    return -1.6094379124341


def P21() -> float:
    return 0


def P22() -> float:
    return -1.6094379124341 / math.log(10)


def P23() -> float:
    return 0


def P24() -> float:
    return 1


def P25() -> float:
    return 4


def P26() -> float:
    return 5.5622817277544


def P27() -> float:
    return 16


def P28() -> float:
    return 9.61


def P29() -> float:
    return 2


def P30() -> float:
    return 2.72029410174709


def P31() -> float:
    return 0.863209366648874


def P32() -> float:
    return 0


def P33() -> float:
    return 0.373876664830236


def P34() -> float:
    return 0


def P35() -> float:
    return 2.01433821447683


def P36() -> float:
    return -math.tan(6)


def P37() -> float:
    return -0.379948962255225


def create_model() -> Model:
    return (
        Model()
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
        .add_derived(
            "P6",
            fn=P6,
            args=[],
        )
        .add_derived(
            "P7",
            fn=P7,
            args=[],
        )
        .add_derived(
            "P8",
            fn=P8,
            args=[],
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
        .add_derived(
            "P30",
            fn=P30,
            args=[],
        )
        .add_derived(
            "P31",
            fn=P31,
            args=[],
        )
        .add_derived(
            "P32",
            fn=P32,
            args=[],
        )
        .add_derived(
            "P33",
            fn=P33,
            args=[],
        )
        .add_derived(
            "P34",
            fn=P34,
            args=[],
        )
        .add_derived(
            "P35",
            fn=P35,
            args=[],
        )
        .add_derived(
            "P36",
            fn=P36,
            args=[],
        )
        .add_derived(
            "P37",
            fn=P37,
            args=[],
        )
    )
