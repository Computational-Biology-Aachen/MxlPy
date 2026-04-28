from mxlpy import Derived, Model


def as8(v9_Kd: float, v9_h: float, p5: float) -> float:
    return (p5 / v9_Kd) ** v9_h / ((p5 / v9_Kd) ** v9_h + 1)


def cod1(pro1_strength: float) -> float:
    return pro1_strength


def cod2(as1: float, rs1: float, pro2_strength: float) -> float:
    return as1 * pro2_strength * rs1


def cod3(pro3_strength: float, rs3: float, rs2: float) -> float:
    return pro3_strength * rs2 * rs3


def cod4(pro4_strength: float, rs6: float, rs7: float) -> float:
    return pro4_strength * rs6 * rs7


def cod5(as2: float, pro5_strength: float) -> float:
    return as2 * pro5_strength


def cod6(pro6_strength: float, as4: float, as3: float) -> float:
    return pro6_strength * (as3 + as4)


def cod7(pro7_strength: float, as7: float, as8: float) -> float:
    return pro7_strength * (as7 + as8)


def cod8(pro8_strength: float, rs4: float, as5: float) -> float:
    return as5 * pro8_strength * rs4


def cod9(pro9_strength: float, rs5: float, as6: float) -> float:
    return as6 * pro9_strength * rs5


def rs1(p9: float, v13_h: float, v13_Kd: float) -> float:
    return 1.0 / ((p9 / v13_Kd) ** v13_h + 1)


def rs2(v2_h: float, p2: float, v2_Kd: float) -> float:
    return 1.0 / ((p2 / v2_Kd) ** v2_h + 1)


def rs3(p3: float, v3_h: float, v3_Kd: float) -> float:
    return 1.0 / ((p3 / v3_Kd) ** v3_h + 1)


def rs4(p8: float, v11_Kd: float, v11_h: float) -> float:
    return 1.0 / ((p8 / v11_Kd) ** v11_h + 1)


def rs5(v12_Kd: float, p8: float, v12_h: float) -> float:
    return 1.0 / ((p8 / v12_Kd) ** v12_h + 1)


def rs6(v14_Kd: float, v14_h: float, p2: float) -> float:
    return 1.0 / ((p2 / v14_Kd) ** v14_h + 1)


def rs7(p3: float, v15_h: float, v15_Kd: float) -> float:
    return 1.0 / ((p3 / v15_Kd) ** v15_h + 1)


def as1(v1_h: float, v1_Kd: float, p1: float) -> float:
    return (p1 / v1_Kd) ** v1_h / ((p1 / v1_Kd) ** v1_h + 1)


def as2(v4_h: float, p4: float, v4_Kd: float) -> float:
    return (p4 / v4_Kd) ** v4_h / ((p4 / v4_Kd) ** v4_h + 1)


def as3(v5_h: float, v5_Kd: float, p5: float) -> float:
    return (p5 / v5_Kd) ** v5_h / ((p5 / v5_Kd) ** v5_h + 1)


def as4(v6_Kd: float, v6_h: float, p6: float) -> float:
    return (p6 / v6_Kd) ** v6_h / ((p6 / v6_Kd) ** v6_h + 1)


def as5(v7_h: float, p7: float, v7_Kd: float) -> float:
    return (p7 / v7_Kd) ** v7_h / ((p7 / v7_Kd) ** v7_h + 1)


def as6(v10_h: float, p7: float, v10_Kd: float) -> float:
    return (p7 / v10_Kd) ** v10_h / ((p7 / v10_Kd) ** v10_h + 1)


def as7(v8_h: float, p6: float, v8_Kd: float) -> float:
    return (p6 / v8_Kd) ** v8_h / ((p6 / v8_Kd) ** v8_h + 1)


def pp9_mrna_conc(DefaultCompartment: float, pp9_mrna: float) -> float:
    return pp9_mrna / DefaultCompartment


def p9_conc(p9: float, DefaultCompartment: float) -> float:
    return p9 / DefaultCompartment


def pp8_mrna_conc(DefaultCompartment: float, pp8_mrna: float) -> float:
    return pp8_mrna / DefaultCompartment


def p8_conc(p8: float, DefaultCompartment: float) -> float:
    return p8 / DefaultCompartment


def pp7_mrna_conc(DefaultCompartment: float, pp7_mrna: float) -> float:
    return pp7_mrna / DefaultCompartment


def p7_conc(DefaultCompartment: float, p7: float) -> float:
    return p7 / DefaultCompartment


def pp6_mrna_conc(pp6_mrna: float, DefaultCompartment: float) -> float:
    return pp6_mrna / DefaultCompartment


def p6_conc(DefaultCompartment: float, p6: float) -> float:
    return p6 / DefaultCompartment


def pp5_mrna_conc(pp5_mrna: float, DefaultCompartment: float) -> float:
    return pp5_mrna / DefaultCompartment


def p5_conc(DefaultCompartment: float, p5: float) -> float:
    return p5 / DefaultCompartment


def pp4_mrna_conc(DefaultCompartment: float, pp4_mrna: float) -> float:
    return pp4_mrna / DefaultCompartment


def p4_conc(DefaultCompartment: float, p4: float) -> float:
    return p4 / DefaultCompartment


def pp3_mrna_conc(pp3_mrna: float, DefaultCompartment: float) -> float:
    return pp3_mrna / DefaultCompartment


def p3_conc(p3: float, DefaultCompartment: float) -> float:
    return p3 / DefaultCompartment


def pp2_mrna_conc(pp2_mrna: float, DefaultCompartment: float) -> float:
    return pp2_mrna / DefaultCompartment


def p2_conc(DefaultCompartment: float, p2: float) -> float:
    return p2 / DefaultCompartment


def pp1_mrna_conc(DefaultCompartment: float, pp1_mrna: float) -> float:
    return pp1_mrna / DefaultCompartment


def p1_conc(DefaultCompartment: float, p1: float) -> float:
    return p1 / DefaultCompartment


def pp9_v1(cod9: float) -> float:
    return cod9


def pp9_v1_stoich_pp9_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp9_v2(pp9_mrna_conc: float, pp9_mrna_degradation_rate: float) -> float:
    return pp9_mrna_conc * pp9_mrna_degradation_rate


def pp9_v2_stoich_pp9_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp9_v3(pp9_mrna_conc: float, rbs9_strength: float) -> float:
    return pp9_mrna_conc * rbs9_strength


def pp9_v3_stoich_p9(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp9_v4(p9_degradation_rate: float, p9_conc: float) -> float:
    return p9_conc * p9_degradation_rate


def pp9_v4_stoich_p9(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp8_v1(cod8: float) -> float:
    return cod8


def pp8_v1_stoich_pp8_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp8_v2(pp8_mrna_conc: float, pp8_mrna_degradation_rate: float) -> float:
    return pp8_mrna_conc * pp8_mrna_degradation_rate


def pp8_v2_stoich_pp8_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp8_v3(rbs8_strength: float, pp8_mrna_conc: float) -> float:
    return pp8_mrna_conc * rbs8_strength


def pp8_v3_stoich_p8(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp8_v4(p8_degradation_rate: float, p8_conc: float) -> float:
    return p8_conc * p8_degradation_rate


def pp8_v4_stoich_p8(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp7_v1(cod7: float) -> float:
    return cod7


def pp7_v1_stoich_pp7_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp7_v2(pp7_mrna_conc: float, pp7_mrna_degradation_rate: float) -> float:
    return pp7_mrna_conc * pp7_mrna_degradation_rate


def pp7_v2_stoich_pp7_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp7_v3(pp7_mrna_conc: float, rbs7_strength: float) -> float:
    return pp7_mrna_conc * rbs7_strength


def pp7_v3_stoich_p7(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp7_v4(p7_conc: float, p7_degradation_rate: float) -> float:
    return p7_conc * p7_degradation_rate


def pp7_v4_stoich_p7(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp6_v1(cod6: float) -> float:
    return cod6


def pp6_v1_stoich_pp6_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp6_v2(pp6_mrna_conc: float, pp6_mrna_degradation_rate: float) -> float:
    return pp6_mrna_conc * pp6_mrna_degradation_rate


def pp6_v2_stoich_pp6_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp6_v3(rbs6_strength: float, pp6_mrna_conc: float) -> float:
    return pp6_mrna_conc * rbs6_strength


def pp6_v3_stoich_p6(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp6_v4(p6_conc: float, p6_degradation_rate: float) -> float:
    return p6_conc * p6_degradation_rate


def pp6_v4_stoich_p6(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp5_v1(cod5: float) -> float:
    return cod5


def pp5_v1_stoich_pp5_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp5_v2(pp5_mrna_degradation_rate: float, pp5_mrna_conc: float) -> float:
    return pp5_mrna_conc * pp5_mrna_degradation_rate


def pp5_v2_stoich_pp5_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp5_v3(rbs5_strength: float, pp5_mrna_conc: float) -> float:
    return pp5_mrna_conc * rbs5_strength


def pp5_v3_stoich_p5(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp5_v4(p5_conc: float, p5_degradation_rate: float) -> float:
    return p5_conc * p5_degradation_rate


def pp5_v4_stoich_p5(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp4_v1(cod4: float) -> float:
    return cod4


def pp4_v1_stoich_pp4_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp4_v2(pp4_mrna_conc: float, pp4_mrna_degradation_rate: float) -> float:
    return pp4_mrna_conc * pp4_mrna_degradation_rate


def pp4_v2_stoich_pp4_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp4_v3(pp4_mrna_conc: float, rbs4_strength: float) -> float:
    return pp4_mrna_conc * rbs4_strength


def pp4_v3_stoich_p4(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp4_v4(p4_degradation_rate: float, p4_conc: float) -> float:
    return p4_conc * p4_degradation_rate


def pp4_v4_stoich_p4(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp3_v1(cod3: float) -> float:
    return cod3


def pp3_v1_stoich_pp3_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp3_v2(pp3_mrna_degradation_rate: float, pp3_mrna_conc: float) -> float:
    return pp3_mrna_conc * pp3_mrna_degradation_rate


def pp3_v2_stoich_pp3_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp3_v3(rbs3_strength: float, pp3_mrna_conc: float) -> float:
    return pp3_mrna_conc * rbs3_strength


def pp3_v3_stoich_p3(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp3_v4(p3_degradation_rate: float, p3_conc: float) -> float:
    return p3_conc * p3_degradation_rate


def pp3_v4_stoich_p3(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp2_v1(cod2: float) -> float:
    return cod2


def pp2_v1_stoich_pp2_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp2_v2(pp2_mrna_degradation_rate: float, pp2_mrna_conc: float) -> float:
    return pp2_mrna_conc * pp2_mrna_degradation_rate


def pp2_v2_stoich_pp2_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp2_v3(rbs2_strength: float, pp2_mrna_conc: float) -> float:
    return pp2_mrna_conc * rbs2_strength


def pp2_v3_stoich_p2(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp2_v4(p2_conc: float, p2_degradation_rate: float) -> float:
    return p2_conc * p2_degradation_rate


def pp2_v4_stoich_p2(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp1_v1(cod1: float) -> float:
    return cod1


def pp1_v1_stoich_pp1_mrna(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp1_v2(pp1_mrna_conc: float, pp1_mrna_degradation_rate: float) -> float:
    return pp1_mrna_conc * pp1_mrna_degradation_rate


def pp1_v2_stoich_pp1_mrna(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def pp1_v3(rbs1_strength: float, pp1_mrna_conc: float) -> float:
    return pp1_mrna_conc * rbs1_strength


def pp1_v3_stoich_p1(DefaultCompartment: float) -> float:
    return 1.0 * DefaultCompartment


def pp1_v4(p1_conc: float, p1_degradation_rate: float) -> float:
    return p1_conc * p1_degradation_rate


def pp1_v4_stoich_p1(DefaultCompartment: float) -> float:
    return -1.0 * DefaultCompartment


def create_model() -> Model:
    return (
        Model()
        .add_variable("pp9_mrna", initial_value=0.0)
        .add_variable("p9", initial_value=0.0)
        .add_variable("pp8_mrna", initial_value=0.0)
        .add_variable("p8", initial_value=0.0)
        .add_variable("pp7_mrna", initial_value=0.0)
        .add_variable("p7", initial_value=0.0)
        .add_variable("pp6_mrna", initial_value=0.0)
        .add_variable("p6", initial_value=0.0)
        .add_variable("pp5_mrna", initial_value=0.0)
        .add_variable("p5", initial_value=0.0)
        .add_variable("pp4_mrna", initial_value=0.0)
        .add_variable("p4", initial_value=0.0)
        .add_variable("pp3_mrna", initial_value=0.0)
        .add_variable("p3", initial_value=0.0)
        .add_variable("pp2_mrna", initial_value=0.0)
        .add_variable("p2", initial_value=0.0)
        .add_variable("pp1_mrna", initial_value=0.0)
        .add_variable("p1", initial_value=5.0)
        .add_parameter("pp8_mrna_degradation_rate", value=1.0)
        .add_parameter("pp9_mrna_degradation_rate", value=1.0)
        .add_parameter("p1_degradation_rate", value=0.5)
        .add_parameter("p2_degradation_rate", value=0.5)
        .add_parameter("p3_degradation_rate", value=0.5)
        .add_parameter("p4_degradation_rate", value=0.5)
        .add_parameter("p5_degradation_rate", value=0.5)
        .add_parameter("p6_degradation_rate", value=0.5)
        .add_parameter("p7_degradation_rate", value=0.5)
        .add_parameter("p8_degradation_rate", value=0.5)
        .add_parameter("p9_degradation_rate", value=0.5)
        .add_parameter("v1_Kd", value=11.147)
        .add_parameter("v1_h", value=1.0)
        .add_parameter("v2_Kd", value=1.0)
        .add_parameter("v2_h", value=4.0)
        .add_parameter("v3_Kd", value=20.0)
        .add_parameter("v3_h", value=1.0)
        .add_parameter("v4_Kd", value=0.2)
        .add_parameter("v4_h", value=4.0)
        .add_parameter("v5_Kd", value=0.2)
        .add_parameter("v5_h", value=4.0)
        .add_parameter("v6_Kd", value=0.04)
        .add_parameter("v6_h", value=4.0)
        .add_parameter("v7_Kd", value=0.02)
        .add_parameter("v7_h", value=4.0)
        .add_parameter("v8_Kd", value=0.04)
        .add_parameter("v8_h", value=4.0)
        .add_parameter("v9_Kd", value=0.2)
        .add_parameter("v9_h", value=4.0)
        .add_parameter("pp1_mrna_degradation_rate", value=1.0)
        .add_parameter("pp2_mrna_degradation_rate", value=1.0)
        .add_parameter("pro1_strength", value=2.0)
        .add_parameter("pro2_strength", value=4.5077)
        .add_parameter("pro3_strength", value=5.0)
        .add_parameter("pro4_strength", value=5.0)
        .add_parameter("pro5_strength", value=5.0)
        .add_parameter("pro6_strength", value=1.31)
        .add_parameter("pro7_strength", value=1.31)
        .add_parameter("pro8_strength", value=5.0)
        .add_parameter("pro9_strength", value=5.0)
        .add_parameter("pp3_mrna_degradation_rate", value=1.0)
        .add_parameter("v10_Kd", value=0.02)
        .add_parameter("v10_h", value=4.0)
        .add_parameter("v11_Kd", value=0.1)
        .add_parameter("v11_h", value=2.0)
        .add_parameter("v12_Kd", value=0.1)
        .add_parameter("v12_h", value=2.0)
        .add_parameter("v13_Kd", value=0.01)
        .add_parameter("v13_h", value=2.0)
        .add_parameter("pp4_mrna_degradation_rate", value=1.0)
        .add_parameter("v14_Kd", value=1.0)
        .add_parameter("v14_h", value=4.0)
        .add_parameter("v15_Kd", value=20.0)
        .add_parameter("v15_h", value=1.0)
        .add_parameter("pp5_mrna_degradation_rate", value=1.0)
        .add_parameter("rbs1_strength", value=0.3668)
        .add_parameter("rbs2_strength", value=1.4102)
        .add_parameter("rbs3_strength", value=0.8)
        .add_parameter("rbs4_strength", value=2.21)
        .add_parameter("rbs5_strength", value=0.5)
        .add_parameter("rbs6_strength", value=2.0)
        .add_parameter("rbs7_strength", value=5.0)
        .add_parameter("rbs8_strength", value=3.6377)
        .add_parameter("rbs9_strength", value=8.0)
        .add_parameter("pp6_mrna_degradation_rate", value=1.0)
        .add_parameter("pp7_mrna_degradation_rate", value=1.0)
        .add_parameter("DefaultCompartment", value=1.0)
        .add_derived(
            "as8",
            fn=as8,
            args=["v9_Kd", "v9_h", "p5"],
        )
        .add_derived(
            "cod1",
            fn=cod1,
            args=["pro1_strength"],
        )
        .add_derived(
            "cod2",
            fn=cod2,
            args=["as1", "rs1", "pro2_strength"],
        )
        .add_derived(
            "cod3",
            fn=cod3,
            args=["pro3_strength", "rs3", "rs2"],
        )
        .add_derived(
            "cod4",
            fn=cod4,
            args=["pro4_strength", "rs6", "rs7"],
        )
        .add_derived(
            "cod5",
            fn=cod5,
            args=["as2", "pro5_strength"],
        )
        .add_derived(
            "cod6",
            fn=cod6,
            args=["pro6_strength", "as4", "as3"],
        )
        .add_derived(
            "cod7",
            fn=cod7,
            args=["pro7_strength", "as7", "as8"],
        )
        .add_derived(
            "cod8",
            fn=cod8,
            args=["pro8_strength", "rs4", "as5"],
        )
        .add_derived(
            "cod9",
            fn=cod9,
            args=["pro9_strength", "rs5", "as6"],
        )
        .add_derived(
            "rs1",
            fn=rs1,
            args=["p9", "v13_h", "v13_Kd"],
        )
        .add_derived(
            "rs2",
            fn=rs2,
            args=["v2_h", "p2", "v2_Kd"],
        )
        .add_derived(
            "rs3",
            fn=rs3,
            args=["p3", "v3_h", "v3_Kd"],
        )
        .add_derived(
            "rs4",
            fn=rs4,
            args=["p8", "v11_Kd", "v11_h"],
        )
        .add_derived(
            "rs5",
            fn=rs5,
            args=["v12_Kd", "p8", "v12_h"],
        )
        .add_derived(
            "rs6",
            fn=rs6,
            args=["v14_Kd", "v14_h", "p2"],
        )
        .add_derived(
            "rs7",
            fn=rs7,
            args=["p3", "v15_h", "v15_Kd"],
        )
        .add_derived(
            "as1",
            fn=as1,
            args=["v1_h", "v1_Kd", "p1"],
        )
        .add_derived(
            "as2",
            fn=as2,
            args=["v4_h", "p4", "v4_Kd"],
        )
        .add_derived(
            "as3",
            fn=as3,
            args=["v5_h", "v5_Kd", "p5"],
        )
        .add_derived(
            "as4",
            fn=as4,
            args=["v6_Kd", "v6_h", "p6"],
        )
        .add_derived(
            "as5",
            fn=as5,
            args=["v7_h", "p7", "v7_Kd"],
        )
        .add_derived(
            "as6",
            fn=as6,
            args=["v10_h", "p7", "v10_Kd"],
        )
        .add_derived(
            "as7",
            fn=as7,
            args=["v8_h", "p6", "v8_Kd"],
        )
        .add_derived(
            "pp9_mrna_conc",
            fn=pp9_mrna_conc,
            args=["DefaultCompartment", "pp9_mrna"],
        )
        .add_derived(
            "p9_conc",
            fn=p9_conc,
            args=["p9", "DefaultCompartment"],
        )
        .add_derived(
            "pp8_mrna_conc",
            fn=pp8_mrna_conc,
            args=["DefaultCompartment", "pp8_mrna"],
        )
        .add_derived(
            "p8_conc",
            fn=p8_conc,
            args=["p8", "DefaultCompartment"],
        )
        .add_derived(
            "pp7_mrna_conc",
            fn=pp7_mrna_conc,
            args=["DefaultCompartment", "pp7_mrna"],
        )
        .add_derived(
            "p7_conc",
            fn=p7_conc,
            args=["DefaultCompartment", "p7"],
        )
        .add_derived(
            "pp6_mrna_conc",
            fn=pp6_mrna_conc,
            args=["pp6_mrna", "DefaultCompartment"],
        )
        .add_derived(
            "p6_conc",
            fn=p6_conc,
            args=["DefaultCompartment", "p6"],
        )
        .add_derived(
            "pp5_mrna_conc",
            fn=pp5_mrna_conc,
            args=["pp5_mrna", "DefaultCompartment"],
        )
        .add_derived(
            "p5_conc",
            fn=p5_conc,
            args=["DefaultCompartment", "p5"],
        )
        .add_derived(
            "pp4_mrna_conc",
            fn=pp4_mrna_conc,
            args=["DefaultCompartment", "pp4_mrna"],
        )
        .add_derived(
            "p4_conc",
            fn=p4_conc,
            args=["DefaultCompartment", "p4"],
        )
        .add_derived(
            "pp3_mrna_conc",
            fn=pp3_mrna_conc,
            args=["pp3_mrna", "DefaultCompartment"],
        )
        .add_derived(
            "p3_conc",
            fn=p3_conc,
            args=["p3", "DefaultCompartment"],
        )
        .add_derived(
            "pp2_mrna_conc",
            fn=pp2_mrna_conc,
            args=["pp2_mrna", "DefaultCompartment"],
        )
        .add_derived(
            "p2_conc",
            fn=p2_conc,
            args=["DefaultCompartment", "p2"],
        )
        .add_derived(
            "pp1_mrna_conc",
            fn=pp1_mrna_conc,
            args=["DefaultCompartment", "pp1_mrna"],
        )
        .add_derived(
            "p1_conc",
            fn=p1_conc,
            args=["DefaultCompartment", "p1"],
        )
        .add_reaction(
            "pp9_v1",
            fn=pp9_v1,
            args=["cod9"],
            stoichiometry={
                "pp9_mrna": Derived(
                    fn=pp9_v1_stoich_pp9_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp9_v2",
            fn=pp9_v2,
            args=["pp9_mrna_conc", "pp9_mrna_degradation_rate"],
            stoichiometry={
                "pp9_mrna": Derived(
                    fn=pp9_v2_stoich_pp9_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp9_v3",
            fn=pp9_v3,
            args=["pp9_mrna_conc", "rbs9_strength"],
            stoichiometry={
                "p9": Derived(fn=pp9_v3_stoich_p9, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp9_v4",
            fn=pp9_v4,
            args=["p9_degradation_rate", "p9_conc"],
            stoichiometry={
                "p9": Derived(fn=pp9_v4_stoich_p9, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp8_v1",
            fn=pp8_v1,
            args=["cod8"],
            stoichiometry={
                "pp8_mrna": Derived(
                    fn=pp8_v1_stoich_pp8_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp8_v2",
            fn=pp8_v2,
            args=["pp8_mrna_conc", "pp8_mrna_degradation_rate"],
            stoichiometry={
                "pp8_mrna": Derived(
                    fn=pp8_v2_stoich_pp8_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp8_v3",
            fn=pp8_v3,
            args=["rbs8_strength", "pp8_mrna_conc"],
            stoichiometry={
                "p8": Derived(fn=pp8_v3_stoich_p8, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp8_v4",
            fn=pp8_v4,
            args=["p8_degradation_rate", "p8_conc"],
            stoichiometry={
                "p8": Derived(fn=pp8_v4_stoich_p8, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp7_v1",
            fn=pp7_v1,
            args=["cod7"],
            stoichiometry={
                "pp7_mrna": Derived(
                    fn=pp7_v1_stoich_pp7_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp7_v2",
            fn=pp7_v2,
            args=["pp7_mrna_conc", "pp7_mrna_degradation_rate"],
            stoichiometry={
                "pp7_mrna": Derived(
                    fn=pp7_v2_stoich_pp7_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp7_v3",
            fn=pp7_v3,
            args=["pp7_mrna_conc", "rbs7_strength"],
            stoichiometry={
                "p7": Derived(fn=pp7_v3_stoich_p7, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp7_v4",
            fn=pp7_v4,
            args=["p7_conc", "p7_degradation_rate"],
            stoichiometry={
                "p7": Derived(fn=pp7_v4_stoich_p7, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp6_v1",
            fn=pp6_v1,
            args=["cod6"],
            stoichiometry={
                "pp6_mrna": Derived(
                    fn=pp6_v1_stoich_pp6_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp6_v2",
            fn=pp6_v2,
            args=["pp6_mrna_conc", "pp6_mrna_degradation_rate"],
            stoichiometry={
                "pp6_mrna": Derived(
                    fn=pp6_v2_stoich_pp6_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp6_v3",
            fn=pp6_v3,
            args=["rbs6_strength", "pp6_mrna_conc"],
            stoichiometry={
                "p6": Derived(fn=pp6_v3_stoich_p6, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp6_v4",
            fn=pp6_v4,
            args=["p6_conc", "p6_degradation_rate"],
            stoichiometry={
                "p6": Derived(fn=pp6_v4_stoich_p6, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp5_v1",
            fn=pp5_v1,
            args=["cod5"],
            stoichiometry={
                "pp5_mrna": Derived(
                    fn=pp5_v1_stoich_pp5_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp5_v2",
            fn=pp5_v2,
            args=["pp5_mrna_degradation_rate", "pp5_mrna_conc"],
            stoichiometry={
                "pp5_mrna": Derived(
                    fn=pp5_v2_stoich_pp5_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp5_v3",
            fn=pp5_v3,
            args=["rbs5_strength", "pp5_mrna_conc"],
            stoichiometry={
                "p5": Derived(fn=pp5_v3_stoich_p5, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp5_v4",
            fn=pp5_v4,
            args=["p5_conc", "p5_degradation_rate"],
            stoichiometry={
                "p5": Derived(fn=pp5_v4_stoich_p5, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp4_v1",
            fn=pp4_v1,
            args=["cod4"],
            stoichiometry={
                "pp4_mrna": Derived(
                    fn=pp4_v1_stoich_pp4_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp4_v2",
            fn=pp4_v2,
            args=["pp4_mrna_conc", "pp4_mrna_degradation_rate"],
            stoichiometry={
                "pp4_mrna": Derived(
                    fn=pp4_v2_stoich_pp4_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp4_v3",
            fn=pp4_v3,
            args=["pp4_mrna_conc", "rbs4_strength"],
            stoichiometry={
                "p4": Derived(fn=pp4_v3_stoich_p4, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp4_v4",
            fn=pp4_v4,
            args=["p4_degradation_rate", "p4_conc"],
            stoichiometry={
                "p4": Derived(fn=pp4_v4_stoich_p4, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp3_v1",
            fn=pp3_v1,
            args=["cod3"],
            stoichiometry={
                "pp3_mrna": Derived(
                    fn=pp3_v1_stoich_pp3_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp3_v2",
            fn=pp3_v2,
            args=["pp3_mrna_degradation_rate", "pp3_mrna_conc"],
            stoichiometry={
                "pp3_mrna": Derived(
                    fn=pp3_v2_stoich_pp3_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp3_v3",
            fn=pp3_v3,
            args=["rbs3_strength", "pp3_mrna_conc"],
            stoichiometry={
                "p3": Derived(fn=pp3_v3_stoich_p3, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp3_v4",
            fn=pp3_v4,
            args=["p3_degradation_rate", "p3_conc"],
            stoichiometry={
                "p3": Derived(fn=pp3_v4_stoich_p3, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp2_v1",
            fn=pp2_v1,
            args=["cod2"],
            stoichiometry={
                "pp2_mrna": Derived(
                    fn=pp2_v1_stoich_pp2_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp2_v2",
            fn=pp2_v2,
            args=["pp2_mrna_degradation_rate", "pp2_mrna_conc"],
            stoichiometry={
                "pp2_mrna": Derived(
                    fn=pp2_v2_stoich_pp2_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp2_v3",
            fn=pp2_v3,
            args=["rbs2_strength", "pp2_mrna_conc"],
            stoichiometry={
                "p2": Derived(fn=pp2_v3_stoich_p2, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp2_v4",
            fn=pp2_v4,
            args=["p2_conc", "p2_degradation_rate"],
            stoichiometry={
                "p2": Derived(fn=pp2_v4_stoich_p2, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp1_v1",
            fn=pp1_v1,
            args=["cod1"],
            stoichiometry={
                "pp1_mrna": Derived(
                    fn=pp1_v1_stoich_pp1_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp1_v2",
            fn=pp1_v2,
            args=["pp1_mrna_conc", "pp1_mrna_degradation_rate"],
            stoichiometry={
                "pp1_mrna": Derived(
                    fn=pp1_v2_stoich_pp1_mrna, args=["DefaultCompartment"]
                )
            },
        )
        .add_reaction(
            "pp1_v3",
            fn=pp1_v3,
            args=["rbs1_strength", "pp1_mrna_conc"],
            stoichiometry={
                "p1": Derived(fn=pp1_v3_stoich_p1, args=["DefaultCompartment"])
            },
        )
        .add_reaction(
            "pp1_v4",
            fn=pp1_v4,
            args=["p1_conc", "p1_degradation_rate"],
            stoichiometry={
                "p1": Derived(fn=pp1_v4_stoich_p1, args=["DefaultCompartment"])
            },
        )
    )
