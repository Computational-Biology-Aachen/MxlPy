import math

from mxlpy import Derived, Model


def P1(S1_stoich: float) -> float:
    return math.exp(S1_stoich)


def P2(S1_stoich: float) -> float:
    return abs(S1_stoich)


def P3(S1_stoich: float) -> float:
    return abs(S1_stoich)


def P4(S1_stoich: float) -> float:
    return math.acos(1 - S1_stoich)


def P5(S1_stoich: float) -> float:
    return math.acos((1 / 4) * S1_stoich)


def P6(S1_stoich: float) -> float:
    return math.asin(S1_stoich - 1)


def P7(S1_stoich: float) -> float:
    return -math.asin((1 / 4) * S1_stoich)


def P8(S1_stoich: float) -> float:
    return math.atan(S1_stoich + 0.8)


def P9(S1_stoich: float) -> float:
    return -math.atan(3 * S1_stoich + 1.09)


def P10(S1_stoich: float) -> float:
    return math.ceil((1 / 4) * S1_stoich)


def P11(S1_stoich: float) -> float:
    return math.ceil(4 * S1_stoich - 0.45)


def P12(S1_stoich: float) -> float:
    return math.ceil(-2 * S1_stoich - 0.6)


def P13(S1_stoich: float) -> float:
    return math.cos(4 * S1_stoich + 1.1)


def P14(S1_stoich: float) -> float:
    return math.cos((1 / 10) * S1_stoich + 0.02)


def P15() -> float:
    return 1


def P16(S1_stoich: float) -> float:
    return math.exp((1 / 2) * S1_stoich)


def P17(S1_stoich: float) -> float:
    return 0.718923733431926 * math.exp((1 / 2) * S1_stoich)


def P18(S1_stoich: float) -> float:
    return math.floor(-2 * S1_stoich - 0.6)


def P19(S1_stoich: float) -> float:
    return math.floor(4 * S1_stoich + 1.1)


def P20(S1_stoich: float) -> float:
    return math.log((1 / 10) * S1_stoich)


def P21(S1_stoich: float) -> float:
    return math.log((1 / 2) * S1_stoich)


def P22(S1_stoich: float) -> float:
    return math.log((1 / 10) * S1_stoich) / math.log(10)


def P23(S1_stoich: float) -> float:
    return math.log((1 / 2) * S1_stoich) / math.log(10)


def P24() -> float:
    return 1


def P25(S1_stoich: float) -> float:
    return S1_stoich**S1_stoich


def P26() -> float:
    return 5.5622817277544


def P27(S1_stoich: float) -> float:
    return (S1_stoich**2) ** S1_stoich


def P28(S1_stoich: float) -> float:
    return 3.1**S1_stoich


def P29(S1_stoich: float) -> float:
    return math.sqrt(2) * math.sqrt(S1_stoich)


def P30(S1_stoich: float) -> float:
    return math.sqrt((1 / 5) * S1_stoich + 7)


def P31(S1_stoich: float) -> float:
    return math.sin(S1_stoich + 0.1)


def P32() -> float:
    return 0


def P33(S1_stoich: float) -> float:
    return -math.sin(2 * S1_stoich + 1.9)


def P34() -> float:
    return 0


def P35(S1_stoich: float) -> float:
    return math.tan((1 / 2) * S1_stoich + 0.11)


def P36(S1_stoich: float) -> float:
    return -math.tan(3 * S1_stoich)


def P37(S1_stoich: float) -> float:
    return 1 / math.cos((1 / 4) * S1_stoich)


def P38(S1_stoich: float) -> float:
    return 1 / math.sin(2.25 * S1_stoich)


def P39(S1_stoich: float) -> float:
    return 1 / math.tan((1 / 10) * S1_stoich)


def P40(S1_stoich: float) -> float:
    return math.sinh((1 / 10) * S1_stoich + 0.1)


def P41(S1_stoich: float) -> float:
    return math.cosh(S1_stoich - 0.3)


def P42(S1_stoich: float) -> float:
    return math.acos(1 / (S1_stoich + 0.3))


def P43(S1_stoich: float) -> float:
    return math.asin(1 / (S1_stoich - 0.9))


def P44(S1_stoich: float) -> float:
    return math.atan(1 / (S1_stoich - 2.1))


def P45(S1_stoich: float) -> float:
    return math.asinh(50 * S1_stoich - 1)


def P46(S1_stoich: float) -> float:
    return math.acosh((1 / 2) * S1_stoich + 0.34)


def P47(S1_stoich: float) -> float:
    return math.atanh(S1_stoich - 2.7)


def P48(S1_stoich: float) -> float:
    return math.log(
        math.sqrt(-1 + 4.76190476190476 / S1_stoich)
        * math.sqrt(1 + 4.76190476190476 / S1_stoich)
        + 4.76190476190476 / S1_stoich
    )


def P49(S1_stoich: float) -> float:
    return math.log(math.sqrt(1 + 40000.0 / S1_stoich**2) + 200.0 / S1_stoich)


def S1_conc(C: float, S1: float) -> float:
    return S1 / C


def J0() -> float:
    return 1


def J0_stoich_S1(S1_stoich: float, C: float) -> float:
    return 1.0 * C * S1_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable("S1_stoich", initial_value=2.0)
        .add_variable("S1", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_derived(
            "P1",
            fn=P1,
            args=["S1_stoich"],
        )
        .add_derived(
            "P2",
            fn=P2,
            args=["S1_stoich"],
        )
        .add_derived(
            "P3",
            fn=P3,
            args=["S1_stoich"],
        )
        .add_derived(
            "P4",
            fn=P4,
            args=["S1_stoich"],
        )
        .add_derived(
            "P5",
            fn=P5,
            args=["S1_stoich"],
        )
        .add_derived(
            "P6",
            fn=P6,
            args=["S1_stoich"],
        )
        .add_derived(
            "P7",
            fn=P7,
            args=["S1_stoich"],
        )
        .add_derived(
            "P8",
            fn=P8,
            args=["S1_stoich"],
        )
        .add_derived(
            "P9",
            fn=P9,
            args=["S1_stoich"],
        )
        .add_derived(
            "P10",
            fn=P10,
            args=["S1_stoich"],
        )
        .add_derived(
            "P11",
            fn=P11,
            args=["S1_stoich"],
        )
        .add_derived(
            "P12",
            fn=P12,
            args=["S1_stoich"],
        )
        .add_derived(
            "P13",
            fn=P13,
            args=["S1_stoich"],
        )
        .add_derived(
            "P14",
            fn=P14,
            args=["S1_stoich"],
        )
        .add_derived(
            "P15",
            fn=P15,
            args=[],
        )
        .add_derived(
            "P16",
            fn=P16,
            args=["S1_stoich"],
        )
        .add_derived(
            "P17",
            fn=P17,
            args=["S1_stoich"],
        )
        .add_derived(
            "P18",
            fn=P18,
            args=["S1_stoich"],
        )
        .add_derived(
            "P19",
            fn=P19,
            args=["S1_stoich"],
        )
        .add_derived(
            "P20",
            fn=P20,
            args=["S1_stoich"],
        )
        .add_derived(
            "P21",
            fn=P21,
            args=["S1_stoich"],
        )
        .add_derived(
            "P22",
            fn=P22,
            args=["S1_stoich"],
        )
        .add_derived(
            "P23",
            fn=P23,
            args=["S1_stoich"],
        )
        .add_derived(
            "P24",
            fn=P24,
            args=[],
        )
        .add_derived(
            "P25",
            fn=P25,
            args=["S1_stoich"],
        )
        .add_derived(
            "P26",
            fn=P26,
            args=[],
        )
        .add_derived(
            "P27",
            fn=P27,
            args=["S1_stoich"],
        )
        .add_derived(
            "P28",
            fn=P28,
            args=["S1_stoich"],
        )
        .add_derived(
            "P29",
            fn=P29,
            args=["S1_stoich"],
        )
        .add_derived(
            "P30",
            fn=P30,
            args=["S1_stoich"],
        )
        .add_derived(
            "P31",
            fn=P31,
            args=["S1_stoich"],
        )
        .add_derived(
            "P32",
            fn=P32,
            args=[],
        )
        .add_derived(
            "P33",
            fn=P33,
            args=["S1_stoich"],
        )
        .add_derived(
            "P34",
            fn=P34,
            args=[],
        )
        .add_derived(
            "P35",
            fn=P35,
            args=["S1_stoich"],
        )
        .add_derived(
            "P36",
            fn=P36,
            args=["S1_stoich"],
        )
        .add_derived(
            "P37",
            fn=P37,
            args=["S1_stoich"],
        )
        .add_derived(
            "P38",
            fn=P38,
            args=["S1_stoich"],
        )
        .add_derived(
            "P39",
            fn=P39,
            args=["S1_stoich"],
        )
        .add_derived(
            "P40",
            fn=P40,
            args=["S1_stoich"],
        )
        .add_derived(
            "P41",
            fn=P41,
            args=["S1_stoich"],
        )
        .add_derived(
            "P42",
            fn=P42,
            args=["S1_stoich"],
        )
        .add_derived(
            "P43",
            fn=P43,
            args=["S1_stoich"],
        )
        .add_derived(
            "P44",
            fn=P44,
            args=["S1_stoich"],
        )
        .add_derived(
            "P45",
            fn=P45,
            args=["S1_stoich"],
        )
        .add_derived(
            "P46",
            fn=P46,
            args=["S1_stoich"],
        )
        .add_derived(
            "P47",
            fn=P47,
            args=["S1_stoich"],
        )
        .add_derived(
            "P48",
            fn=P48,
            args=["S1_stoich"],
        )
        .add_derived(
            "P49",
            fn=P49,
            args=["S1_stoich"],
        )
        .add_derived(
            "S1_conc",
            fn=S1_conc,
            args=["C", "S1"],
        )
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S1": Derived(fn=J0_stoich_S1, args=["S1_stoich", "C"])},
        )
    )
