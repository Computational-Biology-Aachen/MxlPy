import math

from mxlpy import Model


def J0() -> float:
    return math.e


def J1() -> float:
    return math.exp(math.e)


def J2() -> float:
    return 1


def J3() -> float:
    return 1


def J4() -> float:
    return math.pi


def J5() -> float:
    return 1.0471975511966


def J6() -> float:
    return (1 / 2) * math.pi


def J7() -> float:
    return -0.523598775598299


def J8() -> float:
    return 1.22777238637419


def J9() -> float:
    return -1.43067687253053


def J10() -> float:
    return 1


def J11() -> float:
    return 4


def J12() -> float:
    return -4


def J13() -> float:
    return -0.947721602131112


def J14() -> float:
    return 0.975897449330606


def J15() -> float:
    return 1


def J16() -> float:
    return math.e


def J17() -> float:
    return 2.15976625378492


def J18() -> float:
    return -5


def J19() -> float:
    return 9


def J20() -> float:
    return -1.6094379124341


def J21() -> float:
    return 0


def J22() -> float:
    return -1.6094379124341 / math.log(10)


def J23() -> float:
    return 0


def J24() -> float:
    return 1


def J25() -> float:
    return 4


def J26() -> float:
    return 5.5622817277544


def J27() -> float:
    return 16


def J28() -> float:
    return 9.61


def J29() -> float:
    return 2


def J30() -> float:
    return 2.72029410174709


def J31() -> float:
    return 0.863209366648874


def J32() -> float:
    return 0


def J33() -> float:
    return 0.373876664830236


def J34() -> float:
    return 0


def J35() -> float:
    return 2.01433821447683


def J36() -> float:
    return -math.tan(6)


def J37() -> float:
    return 1.13949392732455


def J38() -> float:
    return -1.0229863836713


def J39() -> float:
    return 4.93315487558689


def J40() -> float:
    return 0.304520293447143


def J41() -> float:
    return 2.82831545788997


def J42() -> float:
    return 1.12099946637054


def J43() -> float:
    return 1.14109666064347


def J44() -> float:
    return -1.47112767430373


def J45() -> float:
    return math.asinh(99)


def J46() -> float:
    return 0.802881971289134


def J47() -> float:
    return -0.867300527694053


def J48() -> float:
    return 1.51330668842682


def J49() -> float:
    return 5.29834236561059


def J50() -> float:
    return 1


def J51() -> float:
    return 0


def create_model() -> Model:
    return (
        Model()
        .add_variable("S0", initial_value=0.0)
        .add_variable("S1", initial_value=0.0)
        .add_variable("S2", initial_value=0.0)
        .add_variable("S3", initial_value=0.0)
        .add_variable("S4", initial_value=0.0)
        .add_variable("S5", initial_value=0.0)
        .add_variable("S6", initial_value=0.0)
        .add_variable("S7", initial_value=0.0)
        .add_variable("S8", initial_value=0.0)
        .add_variable("S9", initial_value=0.0)
        .add_variable("S10", initial_value=0.0)
        .add_variable("S11", initial_value=0.0)
        .add_variable("S12", initial_value=0.0)
        .add_variable("S13", initial_value=0.0)
        .add_variable("S14", initial_value=0.0)
        .add_variable("S15", initial_value=0.0)
        .add_variable("S16", initial_value=0.0)
        .add_variable("S17", initial_value=0.0)
        .add_variable("S18", initial_value=0.0)
        .add_variable("S19", initial_value=0.0)
        .add_variable("S20", initial_value=0.0)
        .add_variable("S21", initial_value=0.0)
        .add_variable("S22", initial_value=0.0)
        .add_variable("S23", initial_value=0.0)
        .add_variable("S24", initial_value=0.0)
        .add_variable("S25", initial_value=0.0)
        .add_variable("S26", initial_value=0.0)
        .add_variable("S27", initial_value=0.0)
        .add_variable("S28", initial_value=0.0)
        .add_variable("S29", initial_value=0.0)
        .add_variable("S30", initial_value=0.0)
        .add_variable("S31", initial_value=0.0)
        .add_variable("S32", initial_value=0.0)
        .add_variable("S33", initial_value=0.0)
        .add_variable("S34", initial_value=0.0)
        .add_variable("S35", initial_value=0.0)
        .add_variable("S36", initial_value=0.0)
        .add_variable("S37", initial_value=0.0)
        .add_variable("S38", initial_value=0.0)
        .add_variable("S39", initial_value=0.0)
        .add_variable("S40", initial_value=0.0)
        .add_variable("S41", initial_value=0.0)
        .add_variable("S42", initial_value=0.0)
        .add_variable("S43", initial_value=0.0)
        .add_variable("S44", initial_value=0.0)
        .add_variable("S45", initial_value=0.0)
        .add_variable("S46", initial_value=0.0)
        .add_variable("S47", initial_value=0.0)
        .add_variable("S48", initial_value=0.0)
        .add_variable("S49", initial_value=0.0)
        .add_variable("S50", initial_value=0.0)
        .add_variable("S51", initial_value=0.0)
        .add_parameter("C", value=1.0)
        .add_reaction(
            "J0",
            fn=J0,
            args=[],
            stoichiometry={"S0": 1.0},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=[],
            stoichiometry={"S1": 1.0},
        )
        .add_reaction(
            "J2",
            fn=J2,
            args=[],
            stoichiometry={"S2": 1.0},
        )
        .add_reaction(
            "J3",
            fn=J3,
            args=[],
            stoichiometry={"S3": 1.0},
        )
        .add_reaction(
            "J4",
            fn=J4,
            args=[],
            stoichiometry={"S4": 1.0},
        )
        .add_reaction(
            "J5",
            fn=J5,
            args=[],
            stoichiometry={"S5": 1.0},
        )
        .add_reaction(
            "J6",
            fn=J6,
            args=[],
            stoichiometry={"S6": 1.0},
        )
        .add_reaction(
            "J7",
            fn=J7,
            args=[],
            stoichiometry={"S7": 1.0},
        )
        .add_reaction(
            "J8",
            fn=J8,
            args=[],
            stoichiometry={"S8": 1.0},
        )
        .add_reaction(
            "J9",
            fn=J9,
            args=[],
            stoichiometry={"S9": 1.0},
        )
        .add_reaction(
            "J10",
            fn=J10,
            args=[],
            stoichiometry={"S10": 1.0},
        )
        .add_reaction(
            "J11",
            fn=J11,
            args=[],
            stoichiometry={"S11": 1.0},
        )
        .add_reaction(
            "J12",
            fn=J12,
            args=[],
            stoichiometry={"S12": 1.0},
        )
        .add_reaction(
            "J13",
            fn=J13,
            args=[],
            stoichiometry={"S13": 1.0},
        )
        .add_reaction(
            "J14",
            fn=J14,
            args=[],
            stoichiometry={"S14": 1.0},
        )
        .add_reaction(
            "J15",
            fn=J15,
            args=[],
            stoichiometry={"S15": 1.0},
        )
        .add_reaction(
            "J16",
            fn=J16,
            args=[],
            stoichiometry={"S16": 1.0},
        )
        .add_reaction(
            "J17",
            fn=J17,
            args=[],
            stoichiometry={"S17": 1.0},
        )
        .add_reaction(
            "J18",
            fn=J18,
            args=[],
            stoichiometry={"S18": 1.0},
        )
        .add_reaction(
            "J19",
            fn=J19,
            args=[],
            stoichiometry={"S19": 1.0},
        )
        .add_reaction(
            "J20",
            fn=J20,
            args=[],
            stoichiometry={"S20": 1.0},
        )
        .add_reaction(
            "J21",
            fn=J21,
            args=[],
            stoichiometry={"S21": 1.0},
        )
        .add_reaction(
            "J22",
            fn=J22,
            args=[],
            stoichiometry={"S22": 1.0},
        )
        .add_reaction(
            "J23",
            fn=J23,
            args=[],
            stoichiometry={"S23": 1.0},
        )
        .add_reaction(
            "J24",
            fn=J24,
            args=[],
            stoichiometry={"S24": 1.0},
        )
        .add_reaction(
            "J25",
            fn=J25,
            args=[],
            stoichiometry={"S25": 1.0},
        )
        .add_reaction(
            "J26",
            fn=J26,
            args=[],
            stoichiometry={"S26": 1.0},
        )
        .add_reaction(
            "J27",
            fn=J27,
            args=[],
            stoichiometry={"S27": 1.0},
        )
        .add_reaction(
            "J28",
            fn=J28,
            args=[],
            stoichiometry={"S28": 1.0},
        )
        .add_reaction(
            "J29",
            fn=J29,
            args=[],
            stoichiometry={"S29": 1.0},
        )
        .add_reaction(
            "J30",
            fn=J30,
            args=[],
            stoichiometry={"S30": 1.0},
        )
        .add_reaction(
            "J31",
            fn=J31,
            args=[],
            stoichiometry={"S31": 1.0},
        )
        .add_reaction(
            "J32",
            fn=J32,
            args=[],
            stoichiometry={"S32": 1.0},
        )
        .add_reaction(
            "J33",
            fn=J33,
            args=[],
            stoichiometry={"S33": 1.0},
        )
        .add_reaction(
            "J34",
            fn=J34,
            args=[],
            stoichiometry={"S34": 1.0},
        )
        .add_reaction(
            "J35",
            fn=J35,
            args=[],
            stoichiometry={"S35": 1.0},
        )
        .add_reaction(
            "J36",
            fn=J36,
            args=[],
            stoichiometry={"S36": 1.0},
        )
        .add_reaction(
            "J37",
            fn=J37,
            args=[],
            stoichiometry={"S37": 1.0},
        )
        .add_reaction(
            "J38",
            fn=J38,
            args=[],
            stoichiometry={"S38": 1.0},
        )
        .add_reaction(
            "J39",
            fn=J39,
            args=[],
            stoichiometry={"S39": 1.0},
        )
        .add_reaction(
            "J40",
            fn=J40,
            args=[],
            stoichiometry={"S40": 1.0},
        )
        .add_reaction(
            "J41",
            fn=J41,
            args=[],
            stoichiometry={"S41": 1.0},
        )
        .add_reaction(
            "J42",
            fn=J42,
            args=[],
            stoichiometry={"S42": 1.0},
        )
        .add_reaction(
            "J43",
            fn=J43,
            args=[],
            stoichiometry={"S43": 1.0},
        )
        .add_reaction(
            "J44",
            fn=J44,
            args=[],
            stoichiometry={"S44": 1.0},
        )
        .add_reaction(
            "J45",
            fn=J45,
            args=[],
            stoichiometry={"S45": 1.0},
        )
        .add_reaction(
            "J46",
            fn=J46,
            args=[],
            stoichiometry={"S46": 1.0},
        )
        .add_reaction(
            "J47",
            fn=J47,
            args=[],
            stoichiometry={"S47": 1.0},
        )
        .add_reaction(
            "J48",
            fn=J48,
            args=[],
            stoichiometry={"S48": 1.0},
        )
        .add_reaction(
            "J49",
            fn=J49,
            args=[],
            stoichiometry={"S49": 1.0},
        )
        .add_reaction(
            "J50",
            fn=J50,
            args=[],
            stoichiometry={"S50": 1.0},
        )
        .add_reaction(
            "J51",
            fn=J51,
            args=[],
            stoichiometry={"S51": 1.0},
        )
    )
