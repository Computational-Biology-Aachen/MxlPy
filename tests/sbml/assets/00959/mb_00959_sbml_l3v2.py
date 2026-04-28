import math

from mxlpy import Model


def P1(time: float) -> float:
    return 1 / math.cos(time)


def P2(time: float) -> float:
    return 1 / math.cos(time)


def P3(time: float) -> float:
    return 1 / math.sin(time + 0.001)


def P4(time: float) -> float:
    return -(1 / math.sin(time + 0.001))


def P5(time: float) -> float:
    return 1 / math.tan(time + 0.001)


def P6(time: float) -> float:
    return -(1 / math.tan(time + 0.001))


def P7(time: float) -> float:
    return math.sinh(time)


def P8(time: float) -> float:
    return -math.sinh(time)


def P9(time: float) -> float:
    return math.cosh(time)


def P10(time: float) -> float:
    return math.cosh(time)


def P11(time: float) -> float:
    return math.acos(1 / (time + 1))


def P12(time: float) -> float:
    return math.acos(1 / (-time - 1))


def P13(time: float) -> float:
    return math.asin(1 / (time + 1))


def P14(time: float) -> float:
    return -(math.asin(1 / (time + 1)))


def P15(time: float) -> float:
    return math.atan(1 / time)


def P16(time: float) -> float:
    return -(math.atan(1 / time))


def P17(time: float) -> float:
    return math.asinh(time)


def P18(time: float) -> float:
    return -math.asinh(time)


def P19(time: float) -> float:
    return math.acosh(time + 1)


def P20(time: float) -> float:
    return (math.atanh(time)) if (time < 1) else (10)


def P21(time: float) -> float:
    return (-math.atanh(time)) if (time < 1) else (-10)


def P22(time: float) -> float:
    return (
        (0)
        if (time >= 1) or (time <= 0)
        else (math.log(math.sqrt(-1 + 1 / time) * math.sqrt(1 + 1 / time) + 1 / time))
    )


def P23(time: float) -> float:
    return math.log(math.sqrt(1 + (time + 1) ** (-2)) + 1 / (time + 1))


def P24(time: float) -> float:
    return -(math.log(math.sqrt(1 + (time + 1) ** (-2)) + 1 / (time + 1)))


def P25(time: float) -> float:
    return -1 / 2 * math.log(1 - 1 / (time + 1.001)) + (1 / 2) * math.log(
        1 + 1 / (time + 1.001)
    )


def P26(time: float) -> float:
    return -(
        -1 / 2 * math.log(1 - 1 / (time + 1.001))
        + (1 / 2) * math.log(1 + 1 / (time + 1.001))
    )


def create_model() -> Model:
    return (
        Model()
        .add_derived(
            "P1",
            fn=P1,
            args=["time"],
        )
        .add_derived(
            "P2",
            fn=P2,
            args=["time"],
        )
        .add_derived(
            "P3",
            fn=P3,
            args=["time"],
        )
        .add_derived(
            "P4",
            fn=P4,
            args=["time"],
        )
        .add_derived(
            "P5",
            fn=P5,
            args=["time"],
        )
        .add_derived(
            "P6",
            fn=P6,
            args=["time"],
        )
        .add_derived(
            "P7",
            fn=P7,
            args=["time"],
        )
        .add_derived(
            "P8",
            fn=P8,
            args=["time"],
        )
        .add_derived(
            "P9",
            fn=P9,
            args=["time"],
        )
        .add_derived(
            "P10",
            fn=P10,
            args=["time"],
        )
        .add_derived(
            "P11",
            fn=P11,
            args=["time"],
        )
        .add_derived(
            "P12",
            fn=P12,
            args=["time"],
        )
        .add_derived(
            "P13",
            fn=P13,
            args=["time"],
        )
        .add_derived(
            "P14",
            fn=P14,
            args=["time"],
        )
        .add_derived(
            "P15",
            fn=P15,
            args=["time"],
        )
        .add_derived(
            "P16",
            fn=P16,
            args=["time"],
        )
        .add_derived(
            "P17",
            fn=P17,
            args=["time"],
        )
        .add_derived(
            "P18",
            fn=P18,
            args=["time"],
        )
        .add_derived(
            "P19",
            fn=P19,
            args=["time"],
        )
        .add_derived(
            "P20",
            fn=P20,
            args=["time"],
        )
        .add_derived(
            "P21",
            fn=P21,
            args=["time"],
        )
        .add_derived(
            "P22",
            fn=P22,
            args=["time"],
        )
        .add_derived(
            "P23",
            fn=P23,
            args=["time"],
        )
        .add_derived(
            "P24",
            fn=P24,
            args=["time"],
        )
        .add_derived(
            "P25",
            fn=P25,
            args=["time"],
        )
        .add_derived(
            "P26",
            fn=P26,
            args=["time"],
        )
    )
