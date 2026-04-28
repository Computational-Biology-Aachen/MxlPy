import math

from mxlpy import Derived, InitialAssignment, Model


def init_S0_stoich() -> float:
    return math.e


def init_S1_stoich() -> float:
    return math.exp(math.e)


def init_S2_stoich() -> float:
    return 1


def init_S3_stoich() -> float:
    return 1


def init_S4_stoich() -> float:
    return math.pi


def init_S5_stoich() -> float:
    return 1.0471975511966


def init_S6_stoich() -> float:
    return (1 / 2) * math.pi


def init_S7_stoich() -> float:
    return -0.523598775598299


def init_S8_stoich() -> float:
    return 1.22777238637419


def init_S9_stoich() -> float:
    return -1.43067687253053


def init_S10_stoich() -> float:
    return 1


def init_S11_stoich() -> float:
    return 4


def init_S12_stoich() -> float:
    return -4


def init_S13_stoich() -> float:
    return -0.947721602131112


def init_S14_stoich() -> float:
    return 0.975897449330606


def init_S15_stoich() -> float:
    return 1


def init_S16_stoich() -> float:
    return math.e


def init_S17_stoich() -> float:
    return 2.15976625378492


def init_S18_stoich() -> float:
    return -5


def init_S19_stoich() -> float:
    return 9


def init_S20_stoich() -> float:
    return -1.6094379124341


def init_S21_stoich() -> float:
    return 0


def init_S22_stoich() -> float:
    return -1.6094379124341 / math.log(10)


def init_S23_stoich() -> float:
    return 0


def init_S24_stoich() -> float:
    return 1


def init_S25_stoich() -> float:
    return 4


def init_S26_stoich() -> float:
    return 5.5622817277544


def init_S27_stoich() -> float:
    return 16


def init_S28_stoich() -> float:
    return 9.61


def init_S29_stoich() -> float:
    return 2


def init_S30_stoich() -> float:
    return 2.72029410174709


def init_S31_stoich() -> float:
    return 0.863209366648874


def init_S32_stoich() -> float:
    return 0


def init_S33_stoich() -> float:
    return 0.373876664830236


def init_S34_stoich() -> float:
    return 0


def init_S35_stoich() -> float:
    return 2.01433821447683


def init_S36_stoich() -> float:
    return -math.tan(6)


def init_S37_stoich() -> float:
    return 1.13949392732455


def init_S38_stoich() -> float:
    return -1.0229863836713


def init_S39_stoich() -> float:
    return 4.93315487558689


def init_S40_stoich() -> float:
    return 0.304520293447143


def init_S41_stoich() -> float:
    return 2.82831545788997


def init_S42_stoich() -> float:
    return 1.12099946637054


def init_S43_stoich() -> float:
    return 1.14109666064347


def init_S44_stoich() -> float:
    return -1.47112767430373


def init_S45_stoich() -> float:
    return math.asinh(99)


def init_S46_stoich() -> float:
    return 0.802881971289134


def init_S47_stoich() -> float:
    return -0.867300527694053


def init_S48_stoich() -> float:
    return 1.51330668842682


def init_S49_stoich() -> float:
    return 5.29834236561059


def init_S50_stoich() -> float:
    return 1


def init_S51_stoich() -> float:
    return 0


def J0() -> float:
    return 1


def J0_stoich_S0(S0_stoich: float) -> float:
    return 1.0 * S0_stoich


def J1() -> float:
    return 1


def J1_stoich_S1(S1_stoich: float) -> float:
    return 1.0 * S1_stoich


def J2() -> float:
    return 1


def J2_stoich_S2(S2_stoich: float) -> float:
    return 1.0 * S2_stoich


def J3() -> float:
    return 1


def J3_stoich_S3(S3_stoich: float) -> float:
    return 1.0 * S3_stoich


def J4() -> float:
    return 1


def J4_stoich_S4(S4_stoich: float) -> float:
    return 1.0 * S4_stoich


def J5() -> float:
    return 1


def J5_stoich_S5(S5_stoich: float) -> float:
    return 1.0 * S5_stoich


def J6() -> float:
    return 1


def J6_stoich_S6(S6_stoich: float) -> float:
    return 1.0 * S6_stoich


def J7() -> float:
    return 1


def J7_stoich_S7(S7_stoich: float) -> float:
    return 1.0 * S7_stoich


def J8() -> float:
    return 1


def J8_stoich_S8(S8_stoich: float) -> float:
    return 1.0 * S8_stoich


def J9() -> float:
    return 1


def J9_stoich_S9(S9_stoich: float) -> float:
    return 1.0 * S9_stoich


def J10() -> float:
    return 1


def J10_stoich_S10(S10_stoich: float) -> float:
    return 1.0 * S10_stoich


def J11() -> float:
    return 1


def J11_stoich_S11(S11_stoich: float) -> float:
    return 1.0 * S11_stoich


def J12() -> float:
    return 1


def J12_stoich_S12(S12_stoich: float) -> float:
    return 1.0 * S12_stoich


def J13() -> float:
    return 1


def J13_stoich_S13(S13_stoich: float) -> float:
    return 1.0 * S13_stoich


def J14() -> float:
    return 1


def J14_stoich_S14(S14_stoich: float) -> float:
    return 1.0 * S14_stoich


def J15() -> float:
    return 1


def J15_stoich_S15(S15_stoich: float) -> float:
    return 1.0 * S15_stoich


def J16() -> float:
    return 1


def J16_stoich_S16(S16_stoich: float) -> float:
    return 1.0 * S16_stoich


def J17() -> float:
    return 1


def J17_stoich_S17(S17_stoich: float) -> float:
    return 1.0 * S17_stoich


def J18() -> float:
    return 1


def J18_stoich_S18(S18_stoich: float) -> float:
    return 1.0 * S18_stoich


def J19() -> float:
    return 1


def J19_stoich_S19(S19_stoich: float) -> float:
    return 1.0 * S19_stoich


def J20() -> float:
    return 1


def J20_stoich_S20(S20_stoich: float) -> float:
    return 1.0 * S20_stoich


def J21() -> float:
    return 1


def J21_stoich_S21(S21_stoich: float) -> float:
    return 1.0 * S21_stoich


def J22() -> float:
    return 1


def J22_stoich_S22(S22_stoich: float) -> float:
    return 1.0 * S22_stoich


def J23() -> float:
    return 1


def J23_stoich_S23(S23_stoich: float) -> float:
    return 1.0 * S23_stoich


def J24() -> float:
    return 1


def J24_stoich_S24(S24_stoich: float) -> float:
    return 1.0 * S24_stoich


def J25() -> float:
    return 1


def J25_stoich_S25(S25_stoich: float) -> float:
    return 1.0 * S25_stoich


def J26() -> float:
    return 1


def J26_stoich_S26(S26_stoich: float) -> float:
    return 1.0 * S26_stoich


def J27() -> float:
    return 1


def J27_stoich_S27(S27_stoich: float) -> float:
    return 1.0 * S27_stoich


def J28() -> float:
    return 1


def J28_stoich_S28(S28_stoich: float) -> float:
    return 1.0 * S28_stoich


def J29() -> float:
    return 1


def J29_stoich_S29(S29_stoich: float) -> float:
    return 1.0 * S29_stoich


def J30() -> float:
    return 1


def J30_stoich_S30(S30_stoich: float) -> float:
    return 1.0 * S30_stoich


def J31() -> float:
    return 1


def J31_stoich_S31(S31_stoich: float) -> float:
    return 1.0 * S31_stoich


def J32() -> float:
    return 1


def J32_stoich_S32(S32_stoich: float) -> float:
    return 1.0 * S32_stoich


def J33() -> float:
    return 1


def J33_stoich_S33(S33_stoich: float) -> float:
    return 1.0 * S33_stoich


def J34() -> float:
    return 1


def J34_stoich_S34(S34_stoich: float) -> float:
    return 1.0 * S34_stoich


def J35() -> float:
    return 1


def J35_stoich_S35(S35_stoich: float) -> float:
    return 1.0 * S35_stoich


def J36() -> float:
    return 1


def J36_stoich_S36(S36_stoich: float) -> float:
    return 1.0 * S36_stoich


def J37() -> float:
    return 1


def J37_stoich_S37(S37_stoich: float) -> float:
    return 1.0 * S37_stoich


def J38() -> float:
    return 1


def J38_stoich_S38(S38_stoich: float) -> float:
    return 1.0 * S38_stoich


def J39() -> float:
    return 1


def J39_stoich_S39(S39_stoich: float) -> float:
    return 1.0 * S39_stoich


def J40() -> float:
    return 1


def J40_stoich_S40(S40_stoich: float) -> float:
    return 1.0 * S40_stoich


def J41() -> float:
    return 1


def J41_stoich_S41(S41_stoich: float) -> float:
    return 1.0 * S41_stoich


def J42() -> float:
    return 1


def J42_stoich_S42(S42_stoich: float) -> float:
    return 1.0 * S42_stoich


def J43() -> float:
    return 1


def J43_stoich_S43(S43_stoich: float) -> float:
    return 1.0 * S43_stoich


def J44() -> float:
    return 1


def J44_stoich_S44(S44_stoich: float) -> float:
    return 1.0 * S44_stoich


def J45() -> float:
    return 1


def J45_stoich_S45(S45_stoich: float) -> float:
    return 1.0 * S45_stoich


def J46() -> float:
    return 1


def J46_stoich_S46(S46_stoich: float) -> float:
    return 1.0 * S46_stoich


def J47() -> float:
    return 1


def J47_stoich_S47(S47_stoich: float) -> float:
    return 1.0 * S47_stoich


def J48() -> float:
    return 1


def J48_stoich_S48(S48_stoich: float) -> float:
    return 1.0 * S48_stoich


def J49() -> float:
    return 1


def J49_stoich_S49(S49_stoich: float) -> float:
    return 1.0 * S49_stoich


def J50() -> float:
    return 1


def J50_stoich_S50(S50_stoich: float) -> float:
    return 1.0 * S50_stoich


def J51() -> float:
    return 1


def J51_stoich_S51(S51_stoich: float) -> float:
    return 1.0 * S51_stoich


def create_model() -> Model:
    return (
        Model()
        .add_variable(
            "S0_stoich",
            initial_value=InitialAssignment(fn=init_S0_stoich, args=[]),
        )
        .add_variable(
            "S1_stoich",
            initial_value=InitialAssignment(fn=init_S1_stoich, args=[]),
        )
        .add_variable(
            "S2_stoich",
            initial_value=InitialAssignment(fn=init_S2_stoich, args=[]),
        )
        .add_variable(
            "S3_stoich",
            initial_value=InitialAssignment(fn=init_S3_stoich, args=[]),
        )
        .add_variable(
            "S4_stoich",
            initial_value=InitialAssignment(fn=init_S4_stoich, args=[]),
        )
        .add_variable(
            "S5_stoich",
            initial_value=InitialAssignment(fn=init_S5_stoich, args=[]),
        )
        .add_variable(
            "S6_stoich",
            initial_value=InitialAssignment(fn=init_S6_stoich, args=[]),
        )
        .add_variable(
            "S7_stoich",
            initial_value=InitialAssignment(fn=init_S7_stoich, args=[]),
        )
        .add_variable(
            "S8_stoich",
            initial_value=InitialAssignment(fn=init_S8_stoich, args=[]),
        )
        .add_variable(
            "S9_stoich",
            initial_value=InitialAssignment(fn=init_S9_stoich, args=[]),
        )
        .add_variable(
            "S10_stoich",
            initial_value=InitialAssignment(fn=init_S10_stoich, args=[]),
        )
        .add_variable(
            "S11_stoich",
            initial_value=InitialAssignment(fn=init_S11_stoich, args=[]),
        )
        .add_variable(
            "S12_stoich",
            initial_value=InitialAssignment(fn=init_S12_stoich, args=[]),
        )
        .add_variable(
            "S13_stoich",
            initial_value=InitialAssignment(fn=init_S13_stoich, args=[]),
        )
        .add_variable(
            "S14_stoich",
            initial_value=InitialAssignment(fn=init_S14_stoich, args=[]),
        )
        .add_variable(
            "S15_stoich",
            initial_value=InitialAssignment(fn=init_S15_stoich, args=[]),
        )
        .add_variable(
            "S16_stoich",
            initial_value=InitialAssignment(fn=init_S16_stoich, args=[]),
        )
        .add_variable(
            "S17_stoich",
            initial_value=InitialAssignment(fn=init_S17_stoich, args=[]),
        )
        .add_variable(
            "S18_stoich",
            initial_value=InitialAssignment(fn=init_S18_stoich, args=[]),
        )
        .add_variable(
            "S19_stoich",
            initial_value=InitialAssignment(fn=init_S19_stoich, args=[]),
        )
        .add_variable(
            "S20_stoich",
            initial_value=InitialAssignment(fn=init_S20_stoich, args=[]),
        )
        .add_variable(
            "S21_stoich",
            initial_value=InitialAssignment(fn=init_S21_stoich, args=[]),
        )
        .add_variable(
            "S22_stoich",
            initial_value=InitialAssignment(fn=init_S22_stoich, args=[]),
        )
        .add_variable(
            "S23_stoich",
            initial_value=InitialAssignment(fn=init_S23_stoich, args=[]),
        )
        .add_variable(
            "S24_stoich",
            initial_value=InitialAssignment(fn=init_S24_stoich, args=[]),
        )
        .add_variable(
            "S25_stoich",
            initial_value=InitialAssignment(fn=init_S25_stoich, args=[]),
        )
        .add_variable(
            "S26_stoich",
            initial_value=InitialAssignment(fn=init_S26_stoich, args=[]),
        )
        .add_variable(
            "S27_stoich",
            initial_value=InitialAssignment(fn=init_S27_stoich, args=[]),
        )
        .add_variable(
            "S28_stoich",
            initial_value=InitialAssignment(fn=init_S28_stoich, args=[]),
        )
        .add_variable(
            "S29_stoich",
            initial_value=InitialAssignment(fn=init_S29_stoich, args=[]),
        )
        .add_variable(
            "S30_stoich",
            initial_value=InitialAssignment(fn=init_S30_stoich, args=[]),
        )
        .add_variable(
            "S31_stoich",
            initial_value=InitialAssignment(fn=init_S31_stoich, args=[]),
        )
        .add_variable(
            "S32_stoich",
            initial_value=InitialAssignment(fn=init_S32_stoich, args=[]),
        )
        .add_variable(
            "S33_stoich",
            initial_value=InitialAssignment(fn=init_S33_stoich, args=[]),
        )
        .add_variable(
            "S34_stoich",
            initial_value=InitialAssignment(fn=init_S34_stoich, args=[]),
        )
        .add_variable(
            "S35_stoich",
            initial_value=InitialAssignment(fn=init_S35_stoich, args=[]),
        )
        .add_variable(
            "S36_stoich",
            initial_value=InitialAssignment(fn=init_S36_stoich, args=[]),
        )
        .add_variable(
            "S37_stoich",
            initial_value=InitialAssignment(fn=init_S37_stoich, args=[]),
        )
        .add_variable(
            "S38_stoich",
            initial_value=InitialAssignment(fn=init_S38_stoich, args=[]),
        )
        .add_variable(
            "S39_stoich",
            initial_value=InitialAssignment(fn=init_S39_stoich, args=[]),
        )
        .add_variable(
            "S40_stoich",
            initial_value=InitialAssignment(fn=init_S40_stoich, args=[]),
        )
        .add_variable(
            "S41_stoich",
            initial_value=InitialAssignment(fn=init_S41_stoich, args=[]),
        )
        .add_variable(
            "S42_stoich",
            initial_value=InitialAssignment(fn=init_S42_stoich, args=[]),
        )
        .add_variable(
            "S43_stoich",
            initial_value=InitialAssignment(fn=init_S43_stoich, args=[]),
        )
        .add_variable(
            "S44_stoich",
            initial_value=InitialAssignment(fn=init_S44_stoich, args=[]),
        )
        .add_variable(
            "S45_stoich",
            initial_value=InitialAssignment(fn=init_S45_stoich, args=[]),
        )
        .add_variable(
            "S46_stoich",
            initial_value=InitialAssignment(fn=init_S46_stoich, args=[]),
        )
        .add_variable(
            "S47_stoich",
            initial_value=InitialAssignment(fn=init_S47_stoich, args=[]),
        )
        .add_variable(
            "S48_stoich",
            initial_value=InitialAssignment(fn=init_S48_stoich, args=[]),
        )
        .add_variable(
            "S49_stoich",
            initial_value=InitialAssignment(fn=init_S49_stoich, args=[]),
        )
        .add_variable(
            "S50_stoich",
            initial_value=InitialAssignment(fn=init_S50_stoich, args=[]),
        )
        .add_variable(
            "S51_stoich",
            initial_value=InitialAssignment(fn=init_S51_stoich, args=[]),
        )
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
            stoichiometry={"S0": Derived(fn=J0_stoich_S0, args=["S0_stoich"])},
        )
        .add_reaction(
            "J1",
            fn=J1,
            args=[],
            stoichiometry={"S1": Derived(fn=J1_stoich_S1, args=["S1_stoich"])},
        )
        .add_reaction(
            "J2",
            fn=J2,
            args=[],
            stoichiometry={"S2": Derived(fn=J2_stoich_S2, args=["S2_stoich"])},
        )
        .add_reaction(
            "J3",
            fn=J3,
            args=[],
            stoichiometry={"S3": Derived(fn=J3_stoich_S3, args=["S3_stoich"])},
        )
        .add_reaction(
            "J4",
            fn=J4,
            args=[],
            stoichiometry={"S4": Derived(fn=J4_stoich_S4, args=["S4_stoich"])},
        )
        .add_reaction(
            "J5",
            fn=J5,
            args=[],
            stoichiometry={"S5": Derived(fn=J5_stoich_S5, args=["S5_stoich"])},
        )
        .add_reaction(
            "J6",
            fn=J6,
            args=[],
            stoichiometry={"S6": Derived(fn=J6_stoich_S6, args=["S6_stoich"])},
        )
        .add_reaction(
            "J7",
            fn=J7,
            args=[],
            stoichiometry={"S7": Derived(fn=J7_stoich_S7, args=["S7_stoich"])},
        )
        .add_reaction(
            "J8",
            fn=J8,
            args=[],
            stoichiometry={"S8": Derived(fn=J8_stoich_S8, args=["S8_stoich"])},
        )
        .add_reaction(
            "J9",
            fn=J9,
            args=[],
            stoichiometry={"S9": Derived(fn=J9_stoich_S9, args=["S9_stoich"])},
        )
        .add_reaction(
            "J10",
            fn=J10,
            args=[],
            stoichiometry={"S10": Derived(fn=J10_stoich_S10, args=["S10_stoich"])},
        )
        .add_reaction(
            "J11",
            fn=J11,
            args=[],
            stoichiometry={"S11": Derived(fn=J11_stoich_S11, args=["S11_stoich"])},
        )
        .add_reaction(
            "J12",
            fn=J12,
            args=[],
            stoichiometry={"S12": Derived(fn=J12_stoich_S12, args=["S12_stoich"])},
        )
        .add_reaction(
            "J13",
            fn=J13,
            args=[],
            stoichiometry={"S13": Derived(fn=J13_stoich_S13, args=["S13_stoich"])},
        )
        .add_reaction(
            "J14",
            fn=J14,
            args=[],
            stoichiometry={"S14": Derived(fn=J14_stoich_S14, args=["S14_stoich"])},
        )
        .add_reaction(
            "J15",
            fn=J15,
            args=[],
            stoichiometry={"S15": Derived(fn=J15_stoich_S15, args=["S15_stoich"])},
        )
        .add_reaction(
            "J16",
            fn=J16,
            args=[],
            stoichiometry={"S16": Derived(fn=J16_stoich_S16, args=["S16_stoich"])},
        )
        .add_reaction(
            "J17",
            fn=J17,
            args=[],
            stoichiometry={"S17": Derived(fn=J17_stoich_S17, args=["S17_stoich"])},
        )
        .add_reaction(
            "J18",
            fn=J18,
            args=[],
            stoichiometry={"S18": Derived(fn=J18_stoich_S18, args=["S18_stoich"])},
        )
        .add_reaction(
            "J19",
            fn=J19,
            args=[],
            stoichiometry={"S19": Derived(fn=J19_stoich_S19, args=["S19_stoich"])},
        )
        .add_reaction(
            "J20",
            fn=J20,
            args=[],
            stoichiometry={"S20": Derived(fn=J20_stoich_S20, args=["S20_stoich"])},
        )
        .add_reaction(
            "J21",
            fn=J21,
            args=[],
            stoichiometry={"S21": Derived(fn=J21_stoich_S21, args=["S21_stoich"])},
        )
        .add_reaction(
            "J22",
            fn=J22,
            args=[],
            stoichiometry={"S22": Derived(fn=J22_stoich_S22, args=["S22_stoich"])},
        )
        .add_reaction(
            "J23",
            fn=J23,
            args=[],
            stoichiometry={"S23": Derived(fn=J23_stoich_S23, args=["S23_stoich"])},
        )
        .add_reaction(
            "J24",
            fn=J24,
            args=[],
            stoichiometry={"S24": Derived(fn=J24_stoich_S24, args=["S24_stoich"])},
        )
        .add_reaction(
            "J25",
            fn=J25,
            args=[],
            stoichiometry={"S25": Derived(fn=J25_stoich_S25, args=["S25_stoich"])},
        )
        .add_reaction(
            "J26",
            fn=J26,
            args=[],
            stoichiometry={"S26": Derived(fn=J26_stoich_S26, args=["S26_stoich"])},
        )
        .add_reaction(
            "J27",
            fn=J27,
            args=[],
            stoichiometry={"S27": Derived(fn=J27_stoich_S27, args=["S27_stoich"])},
        )
        .add_reaction(
            "J28",
            fn=J28,
            args=[],
            stoichiometry={"S28": Derived(fn=J28_stoich_S28, args=["S28_stoich"])},
        )
        .add_reaction(
            "J29",
            fn=J29,
            args=[],
            stoichiometry={"S29": Derived(fn=J29_stoich_S29, args=["S29_stoich"])},
        )
        .add_reaction(
            "J30",
            fn=J30,
            args=[],
            stoichiometry={"S30": Derived(fn=J30_stoich_S30, args=["S30_stoich"])},
        )
        .add_reaction(
            "J31",
            fn=J31,
            args=[],
            stoichiometry={"S31": Derived(fn=J31_stoich_S31, args=["S31_stoich"])},
        )
        .add_reaction(
            "J32",
            fn=J32,
            args=[],
            stoichiometry={"S32": Derived(fn=J32_stoich_S32, args=["S32_stoich"])},
        )
        .add_reaction(
            "J33",
            fn=J33,
            args=[],
            stoichiometry={"S33": Derived(fn=J33_stoich_S33, args=["S33_stoich"])},
        )
        .add_reaction(
            "J34",
            fn=J34,
            args=[],
            stoichiometry={"S34": Derived(fn=J34_stoich_S34, args=["S34_stoich"])},
        )
        .add_reaction(
            "J35",
            fn=J35,
            args=[],
            stoichiometry={"S35": Derived(fn=J35_stoich_S35, args=["S35_stoich"])},
        )
        .add_reaction(
            "J36",
            fn=J36,
            args=[],
            stoichiometry={"S36": Derived(fn=J36_stoich_S36, args=["S36_stoich"])},
        )
        .add_reaction(
            "J37",
            fn=J37,
            args=[],
            stoichiometry={"S37": Derived(fn=J37_stoich_S37, args=["S37_stoich"])},
        )
        .add_reaction(
            "J38",
            fn=J38,
            args=[],
            stoichiometry={"S38": Derived(fn=J38_stoich_S38, args=["S38_stoich"])},
        )
        .add_reaction(
            "J39",
            fn=J39,
            args=[],
            stoichiometry={"S39": Derived(fn=J39_stoich_S39, args=["S39_stoich"])},
        )
        .add_reaction(
            "J40",
            fn=J40,
            args=[],
            stoichiometry={"S40": Derived(fn=J40_stoich_S40, args=["S40_stoich"])},
        )
        .add_reaction(
            "J41",
            fn=J41,
            args=[],
            stoichiometry={"S41": Derived(fn=J41_stoich_S41, args=["S41_stoich"])},
        )
        .add_reaction(
            "J42",
            fn=J42,
            args=[],
            stoichiometry={"S42": Derived(fn=J42_stoich_S42, args=["S42_stoich"])},
        )
        .add_reaction(
            "J43",
            fn=J43,
            args=[],
            stoichiometry={"S43": Derived(fn=J43_stoich_S43, args=["S43_stoich"])},
        )
        .add_reaction(
            "J44",
            fn=J44,
            args=[],
            stoichiometry={"S44": Derived(fn=J44_stoich_S44, args=["S44_stoich"])},
        )
        .add_reaction(
            "J45",
            fn=J45,
            args=[],
            stoichiometry={"S45": Derived(fn=J45_stoich_S45, args=["S45_stoich"])},
        )
        .add_reaction(
            "J46",
            fn=J46,
            args=[],
            stoichiometry={"S46": Derived(fn=J46_stoich_S46, args=["S46_stoich"])},
        )
        .add_reaction(
            "J47",
            fn=J47,
            args=[],
            stoichiometry={"S47": Derived(fn=J47_stoich_S47, args=["S47_stoich"])},
        )
        .add_reaction(
            "J48",
            fn=J48,
            args=[],
            stoichiometry={"S48": Derived(fn=J48_stoich_S48, args=["S48_stoich"])},
        )
        .add_reaction(
            "J49",
            fn=J49,
            args=[],
            stoichiometry={"S49": Derived(fn=J49_stoich_S49, args=["S49_stoich"])},
        )
        .add_reaction(
            "J50",
            fn=J50,
            args=[],
            stoichiometry={"S50": Derived(fn=J50_stoich_S50, args=["S50_stoich"])},
        )
        .add_reaction(
            "J51",
            fn=J51,
            args=[],
            stoichiometry={"S51": Derived(fn=J51_stoich_S51, args=["S51_stoich"])},
        )
    )
