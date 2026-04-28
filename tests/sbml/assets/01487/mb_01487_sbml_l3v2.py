import math

from mxlpy import Model


def P1(time: float) -> float:
    return time


def P2(time: float) -> float:
    return abs(time)


def P3(time: float) -> float:
    return abs(time)


def P4(time: float) -> float:
    return math.acos(time)


def P5(time: float) -> float:
    return math.acos(-time)


def P6(time: float) -> float:
    return math.asin(time)


def P7(time: float) -> float:
    return -math.asin(time)


def P8(time: float) -> float:
    return math.atan(time)


def P9(time: float) -> float:
    return -math.atan(time)


def P10(time: float) -> float:
    return math.ceil(time)


def P11(time: float) -> float:
    return math.ceil(-time)


def P13(time: float) -> float:
    return math.cos(time)


def P14(time: float) -> float:
    return math.cos(time)


def P15(time: float) -> float:
    return math.exp(time)


def P16(time: float) -> float:
    return math.exp(-time)


def P18(time: float) -> float:
    return math.floor(time)


def P19(time: float) -> float:
    return math.floor(-time)


def P20(time: float) -> float:
    return math.log(time + 1)


def P22(time: float) -> float:
    return math.log(time + 1) / math.log(10)


def P29(time: float) -> float:
    return math.sqrt(time)


def P31(time: float) -> float:
    return math.sin(time)


def P32(time: float) -> float:
    return -math.sin(time)


def P34(time: float) -> float:
    return math.tan(time)


def P35(time: float) -> float:
    return -math.tan(time)


def P36(time: float) -> float:
    return math.tanh(time)


def P37(time: float) -> float:
    return -math.tanh(time)


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
            "P22",
            fn=P22,
            args=["time"],
        )
        .add_derived(
            "P29",
            fn=P29,
            args=["time"],
        )
        .add_derived(
            "P31",
            fn=P31,
            args=["time"],
        )
        .add_derived(
            "P32",
            fn=P32,
            args=["time"],
        )
        .add_derived(
            "P34",
            fn=P34,
            args=["time"],
        )
        .add_derived(
            "P35",
            fn=P35,
            args=["time"],
        )
        .add_derived(
            "P36",
            fn=P36,
            args=["time"],
        )
        .add_derived(
            "P37",
            fn=P37,
            args=["time"],
        )
    )
