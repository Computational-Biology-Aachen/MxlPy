import math

from mxlpy import Model


def dP2() -> float:
    return 1


def dP3() -> float:
    return 1


def dP4() -> float:
    return math.pi


def dP5() -> float:
    return 1.0471975511966


def dP6() -> float:
    return (1 / 2) * math.pi


def dP7() -> float:
    return -0.523598775598299


def dP8() -> float:
    return 1.22777238637419


def dP9() -> float:
    return -1.43067687253053


def dP10() -> float:
    return 1


def dP11() -> float:
    return 4


def dP12() -> float:
    return -4


def dP13() -> float:
    return -0.947721602131112


def dP14() -> float:
    return 0.975897449330606


def dP15() -> float:
    return 1


def dP16() -> float:
    return math.e


def dP17() -> float:
    return 2.15976625378492


def dP18() -> float:
    return -5


def dP19() -> float:
    return 9


def dP20() -> float:
    return -1.6094379124341


def dP21() -> float:
    return 0


def dP22() -> float:
    return -1.6094379124341 / math.log(10)


def dP23() -> float:
    return 0


def dP24() -> float:
    return 1


def dP25() -> float:
    return 4


def dP26() -> float:
    return 5.5622817277544


def dP27() -> float:
    return 16


def dP28() -> float:
    return 9.61


def dP29() -> float:
    return 2


def dP30() -> float:
    return 2.72029410174709


def dP31() -> float:
    return 0.863209366648874


def dP32() -> float:
    return 0


def dP33() -> float:
    return 0.373876664830236


def dP34() -> float:
    return 0


def dP35() -> float:
    return 2.01433821447683


def dP36() -> float:
    return -math.tan(6)


def dP37() -> float:
    return 0


def dP38() -> float:
    return 0.804062391404892


def dP39() -> float:
    return -math.tanh(6)


def create_model() -> Model:
    return (
        Model()
        .add_variable("P2", initial_value=0.0)
        .add_variable("P3", initial_value=0.0)
        .add_variable("P4", initial_value=0.0)
        .add_variable("P5", initial_value=0.0)
        .add_variable("P6", initial_value=0.0)
        .add_variable("P7", initial_value=0.0)
        .add_variable("P8", initial_value=0.0)
        .add_variable("P9", initial_value=0.0)
        .add_variable("P10", initial_value=0.0)
        .add_variable("P11", initial_value=0.0)
        .add_variable("P12", initial_value=0.0)
        .add_variable("P13", initial_value=0.0)
        .add_variable("P14", initial_value=0.0)
        .add_variable("P15", initial_value=0.0)
        .add_variable("P16", initial_value=0.0)
        .add_variable("P17", initial_value=0.0)
        .add_variable("P18", initial_value=0.0)
        .add_variable("P19", initial_value=0.0)
        .add_variable("P20", initial_value=0.0)
        .add_variable("P21", initial_value=0.0)
        .add_variable("P22", initial_value=0.0)
        .add_variable("P23", initial_value=0.0)
        .add_variable("P24", initial_value=0.0)
        .add_variable("P25", initial_value=0.0)
        .add_variable("P26", initial_value=0.0)
        .add_variable("P27", initial_value=0.0)
        .add_variable("P28", initial_value=0.0)
        .add_variable("P29", initial_value=0.0)
        .add_variable("P30", initial_value=0.0)
        .add_variable("P31", initial_value=0.0)
        .add_variable("P32", initial_value=0.0)
        .add_variable("P33", initial_value=0.0)
        .add_variable("P34", initial_value=0.0)
        .add_variable("P35", initial_value=0.0)
        .add_variable("P36", initial_value=0.0)
        .add_variable("P37", initial_value=0.0)
        .add_variable("P38", initial_value=0.0)
        .add_variable("P39", initial_value=0.0)
        .add_reaction(
            "dP2",
            fn=dP2,
            args=[],
            stoichiometry={"P2": 1.0},
        )
        .add_reaction(
            "dP3",
            fn=dP3,
            args=[],
            stoichiometry={"P3": 1.0},
        )
        .add_reaction(
            "dP4",
            fn=dP4,
            args=[],
            stoichiometry={"P4": 1.0},
        )
        .add_reaction(
            "dP5",
            fn=dP5,
            args=[],
            stoichiometry={"P5": 1.0},
        )
        .add_reaction(
            "dP6",
            fn=dP6,
            args=[],
            stoichiometry={"P6": 1.0},
        )
        .add_reaction(
            "dP7",
            fn=dP7,
            args=[],
            stoichiometry={"P7": 1.0},
        )
        .add_reaction(
            "dP8",
            fn=dP8,
            args=[],
            stoichiometry={"P8": 1.0},
        )
        .add_reaction(
            "dP9",
            fn=dP9,
            args=[],
            stoichiometry={"P9": 1.0},
        )
        .add_reaction(
            "dP10",
            fn=dP10,
            args=[],
            stoichiometry={"P10": 1.0},
        )
        .add_reaction(
            "dP11",
            fn=dP11,
            args=[],
            stoichiometry={"P11": 1.0},
        )
        .add_reaction(
            "dP12",
            fn=dP12,
            args=[],
            stoichiometry={"P12": 1.0},
        )
        .add_reaction(
            "dP13",
            fn=dP13,
            args=[],
            stoichiometry={"P13": 1.0},
        )
        .add_reaction(
            "dP14",
            fn=dP14,
            args=[],
            stoichiometry={"P14": 1.0},
        )
        .add_reaction(
            "dP15",
            fn=dP15,
            args=[],
            stoichiometry={"P15": 1.0},
        )
        .add_reaction(
            "dP16",
            fn=dP16,
            args=[],
            stoichiometry={"P16": 1.0},
        )
        .add_reaction(
            "dP17",
            fn=dP17,
            args=[],
            stoichiometry={"P17": 1.0},
        )
        .add_reaction(
            "dP18",
            fn=dP18,
            args=[],
            stoichiometry={"P18": 1.0},
        )
        .add_reaction(
            "dP19",
            fn=dP19,
            args=[],
            stoichiometry={"P19": 1.0},
        )
        .add_reaction(
            "dP20",
            fn=dP20,
            args=[],
            stoichiometry={"P20": 1.0},
        )
        .add_reaction(
            "dP21",
            fn=dP21,
            args=[],
            stoichiometry={"P21": 1.0},
        )
        .add_reaction(
            "dP22",
            fn=dP22,
            args=[],
            stoichiometry={"P22": 1.0},
        )
        .add_reaction(
            "dP23",
            fn=dP23,
            args=[],
            stoichiometry={"P23": 1.0},
        )
        .add_reaction(
            "dP24",
            fn=dP24,
            args=[],
            stoichiometry={"P24": 1.0},
        )
        .add_reaction(
            "dP25",
            fn=dP25,
            args=[],
            stoichiometry={"P25": 1.0},
        )
        .add_reaction(
            "dP26",
            fn=dP26,
            args=[],
            stoichiometry={"P26": 1.0},
        )
        .add_reaction(
            "dP27",
            fn=dP27,
            args=[],
            stoichiometry={"P27": 1.0},
        )
        .add_reaction(
            "dP28",
            fn=dP28,
            args=[],
            stoichiometry={"P28": 1.0},
        )
        .add_reaction(
            "dP29",
            fn=dP29,
            args=[],
            stoichiometry={"P29": 1.0},
        )
        .add_reaction(
            "dP30",
            fn=dP30,
            args=[],
            stoichiometry={"P30": 1.0},
        )
        .add_reaction(
            "dP31",
            fn=dP31,
            args=[],
            stoichiometry={"P31": 1.0},
        )
        .add_reaction(
            "dP32",
            fn=dP32,
            args=[],
            stoichiometry={"P32": 1.0},
        )
        .add_reaction(
            "dP33",
            fn=dP33,
            args=[],
            stoichiometry={"P33": 1.0},
        )
        .add_reaction(
            "dP34",
            fn=dP34,
            args=[],
            stoichiometry={"P34": 1.0},
        )
        .add_reaction(
            "dP35",
            fn=dP35,
            args=[],
            stoichiometry={"P35": 1.0},
        )
        .add_reaction(
            "dP36",
            fn=dP36,
            args=[],
            stoichiometry={"P36": 1.0},
        )
        .add_reaction(
            "dP37",
            fn=dP37,
            args=[],
            stoichiometry={"P37": 1.0},
        )
        .add_reaction(
            "dP38",
            fn=dP38,
            args=[],
            stoichiometry={"P38": 1.0},
        )
        .add_reaction(
            "dP39",
            fn=dP39,
            args=[],
            stoichiometry={"P39": 1.0},
        )
    )
