"""Calvin cycle model from Poolman et al. 2000."""

from __future__ import annotations

from mxlpy import KineticModelBuilder
from mxlpy.fns import moiety_1s

__all__ = [
    "free_orthophosphate",
    "get_model",
    "n_export",
    "rapid_equilibrium_1s_1p",
    "rapid_equilibrium_2s_1p",
    "rapid_equilibrium_2s_2p",
    "starch",
    "v1",
    "v13",
    "v16",
    "v3",
    "v6",
    "v9",
    "v_out",
]


def rapid_equilibrium_1s_1p(
    s1: float,
    p1: float,
    kre: float,
    q: float,
) -> float:
    """Rapid equilibrium reaction with 1 substrate and 1 product."""
    return kre * (s1 - p1 / q)


def rapid_equilibrium_2s_1p(
    s1: float,
    s2: float,
    p1: float,
    kre: float,
    q: float,
) -> float:
    """Rapid equilibrium reaction with 2 substrates and 1 product."""
    return kre * (s1 * s2 - p1 / q)


def rapid_equilibrium_2s_2p(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    kre: float,
    q: float,
) -> float:
    """Rapid equilibrium reaction with 2 substrates and 2 products."""
    return kre * (s1 * s2 - (p1 * p2) / q)


def v_out(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
) -> float:
    """Efflux reaction."""
    return (vmax_efflux * s1) / (n_total * k_efflux)


def v1(
    rubp: float,
    pga: float,
    fbp: float,
    sbp: float,
    p: float,
    vmax: float,
    km: float,
    k1_pga: float,
    ki_fbp: float,
    ki_sbp: float,
    ki_pi: float,
    ki_nadph: float,
    nadph: float,
) -> float:
    """Rubisco reaction."""
    return (vmax * rubp) / (
        rubp
        + km
        * (
            1
            + (pga / k1_pga)
            + (fbp / ki_fbp)
            + (sbp / ki_sbp)
            + (p / ki_pi)
            + (nadph / ki_nadph)
        )
    )


def v3(
    bpga: float,
    gap: float,
    phosphate_pool: float,
    proton_pool_stroma: float,
    nadph: float,
    nadp: float,
    kre: float,
    q3: float,
) -> float:
    """GAPDH reaction."""
    return kre * (
        (nadph * bpga * proton_pool_stroma) - (1 / q3) * (gap * nadp * phosphate_pool)
    )


def v6(
    fbp: float,
    f6p: float,
    pi: float,
    vmax: float,
    km: float,
    ki_f6p: float,
    ki_pi: float,
) -> float:
    """FBPase reaction."""
    return (vmax * fbp) / (fbp + km * (1 + (f6p / ki_f6p) + (pi / ki_pi)))


def v9(
    sbp: float,
    pi: float,
    vma: float,
    km: float,
    ki_pi: float,
) -> float:
    """SBPase reaction."""
    return (vma * sbp) / (sbp + km * (1 + (pi / ki_pi)))


def v13(
    ru5p: float,
    atp: float,
    pi: float,
    pga: float,
    rubp: float,
    adp: float,
    vmax: float,
    km_ru5p: float,
    km_adp: float,
    ki_pga: float,
    ki_rubp: float,
    ki_pi: float,
    ki_adp_ru5p: float,
    ki_adp_atp: float,
) -> float:
    """Phosphoribulokinase reaction."""
    return (vmax * ru5p * atp) / (
        (ru5p + km_ru5p * (1 + (pga / ki_pga) + (rubp / ki_rubp) + (pi / ki_pi)))
        * (atp * (1 + (adp / ki_adp_ru5p)) + km_adp * (1 + (adp / ki_adp_atp)))
    )


def v16(
    adp: float,
    pi: float,
    vmax: float,
    km_adp: float,
    km_pi: float,
) -> float:
    """ATP synthase reaction."""
    return (vmax * adp * pi) / ((adp + km_adp) * (pi + km_pi))


def starch(
    g1p: float,
    atp: float,
    adp: float,
    pi: float,
    pga: float,
    f6p: float,
    fbp: float,
    vmax: float,
    km_g1p: float,
    km_atp: float,
    ki_adp: float,
    ka_pga: float,
    ka_f6p: float,
    ka_fbp: float,
) -> float:
    """Starch synthesis reaction."""
    return (vmax * g1p * atp) / (
        (g1p + km_g1p)
        * (
            (1 + (adp / ki_adp)) * (atp + km_atp)
            + ((km_atp * pi) / (ka_pga * pga + ka_f6p * f6p + ka_fbp * fbp))
        )
    )


def free_orthophosphate(
    pga: float,
    bpga: float,
    gap: float,
    dhap: float,
    fbp: float,
    f6p: float,
    g6p: float,
    g1p: float,
    sbp: float,
    s7p: float,
    e4p: float,
    x4p: float,
    r5p: float,
    rubp: float,
    ru5p: float,
    atp: float,
    phosphate_total: float,
) -> float:
    """Free orthophosphate moiety."""
    return phosphate_total - (
        pga
        + 2 * bpga
        + gap
        + dhap
        + 2 * fbp
        + f6p
        + g6p
        + g1p
        + 2 * sbp
        + s7p
        + e4p
        + x4p
        + r5p
        + 2 * rubp
        + ru5p
        + atp
    )


def n_export(
    pi: float,
    pga: float,
    gap: float,
    dhap: float,
    kpxt: float,
    pi_ext: float,
    kpi: float,
    kpga: float,
    kgap: float,
    kdhap: float,
) -> float:
    """Export scaling."""
    return 1 + (1 + (kpxt / pi_ext)) * (
        (pi / kpi) + (pga / kpga) + (gap / kgap) + (dhap / kdhap)
    )


parameters = {
    "Vmax_1": 2.72,  # [mM/s], Pettersson 1988
    "Vmax_6": 1.6,  # [mM/s], Pettersson 1988
    "Vmax_9": 0.32,  # [mM/s], Pettersson 1988
    "Vmax_13": 8.0,  # [mM/s], Pettersson 1988
    "Vmax_16": 2.8,  # [mM/s], Pettersson 1988
    "Vmax_starch": 0.32,  # [mM/s], Pettersson 1988
    "Vmax_efflux": 2.0,  # [mM/s], Pettersson 1988
    "Km_1": 0.02,  # [mM], Pettersson 1988
    "Km_6": 0.03,  # [mM], Pettersson 1988
    "Km_9": 0.013,  # [mM], Pettersson 1988
    "Km_13_1": 0.05,  # [mM], Pettersson 1988
    "Km_13_2": 0.05,  # [mM], Pettersson 1988
    "Km_16_1": 0.014,  # [mM], Pettersson 1988
    "Km_16_2": 0.3,  # [mM], Pettersson 1988
    "Km_starch_1": 0.08,  # [mM], Pettersson 1988
    "Km_starch_2": 0.08,  # [mM], Pettersson 1988
    "K_pga": 0.25,  # [mM], Pettersson 1988
    "K_gap": 0.075,  # [mM], Pettersson 1988
    "K_dhap": 0.077,  # [mM], Pettersson 1988
    "K_pi": 0.63,  # [mM], Pettersson 1988
    "K_pxt": 0.74,  # [mM], Pettersson 1988
    "Ki_1_1": 0.04,  # [mM], Pettersson 1988
    "Ki_1_2": 0.04,  # [mM], Pettersson 1988
    "Ki_1_3": 0.075,  # [mM], Pettersson 1988
    "Ki_1_4": 0.9,  # [mM], Pettersson 1988
    "Ki_1_5": 0.07,  # [mM], Pettersson 1988
    "Ki_6_1": 0.7,  # [mM], Pettersson 1988
    "Ki_6_2": 12.0,  # [mM], Pettersson 1988
    "Ki_9": 12.0,  # [mM], Pettersson 1988
    "Ki_13_1": 2.0,  # [mM], Pettersson 1988
    "Ki_13_2": 0.7,  # [mM], Pettersson 1988
    "Ki_13_3": 4.0,  # [mM], Pettersson 1988
    "Ki_13_4": 2.5,  # [mM], Pettersson 1988
    "Ki_13_5": 0.4,  # [mM], Pettersson 1988
    "Ki_starch": 10.0,  # [mM], Pettersson 1988
    "Ka_starch_1": 0.1,  # [mM], Pettersson 1988
    "Ka_starch_2": 0.02,  # [mM], Pettersson 1988
    "Ka_starch_3": 0.02,  # [mM], Pettersson 1988
    "k_rapid_eq": 800000000.0,  # Rapid Equilibrium speed
    "q2": 0.00031,  # [], Pettersson 1988
    "q3": 16000000.0,  # [], Pettersson 1988
    "q4": 22.0,  # [], Pettersson 1988
    "q5": 7.1,  # [1/mM]], Pettersson 1988
    "q7": 0.084,  # [], Pettersson 1988
    "q8": 13.0,  # [1/mM]], Pettersson 1988
    "q10": 0.85,  # [], Pettersson 1988
    "q11": 0.4,  # [], Pettersson 1988
    "q12": 0.67,  # [], Pettersson 1988
    "q14": 2.3,  # [], Pettersson 1988
    "q15": 0.058,  # [], Pettersson 1988
    "CO2": 0.2,  # [mM], Pettersson 1988
    "Phosphate_total": 15.0,  # [mM], Pettersson 1988
    "AP_total": 0.5,  # [mM], Pettersson 1988
    "N_total": 0.5,  # [mM], Pettersson 1988
    "Phosphate_pool_ext": 0.5,  # [mM], Pettersson 1988
    "pH_medium": 7.6,  # [], Pettersson 1988
    "pH_stroma": 7.9,  # [], Pettersson 1988
    "proton_pool_stroma": 1.2589254117941661e-05,  # [mM], Pettersson 1988
    "NADPH_pool": 0.21,  # [mM], Pettersson 1988
    "NADP_pool": 0.29,  # [mM], Pettersson 1988
}

variables = {
    "PGA": 0.6437280277346407,
    "BPGA": 0.001360476366780556,
    "GAP": 0.011274125311289358,
    "DHAP": 0.24803073890728228,
    "FBP": 0.019853938009873073,
    "F6P": 1.0950701164493861,
    "G6P": 2.5186612678035734,
    "G1P": 0.14608235353185037,
    "SBP": 0.09193353265673603,
    "S7P": 0.23124426886012006,
    "E4P": 0.028511831060903877,
    "X5P": 0.036372985623662736,
    "R5P": 0.06092475016463224,
    "RUBP": 0.24993009253928708,
    "RU5P": 0.02436989993734177,
    "ATP": 0.43604115800259613,
}


def get_model() -> KineticModelBuilder:
    """Calvin cycle model from Poolman et al. 2000."""
    model = KineticModelBuilder()
    model.add_parameters(parameters)
    model.add_variables(variables)

    model.add_derived(
        name="ADP",
        fn=moiety_1s,
        args=["ATP", "AP_total"],
    )

    model.add_derived(
        name="Phosphate_pool",
        fn=free_orthophosphate,
        args=[
            "PGA",
            "BPGA",
            "GAP",
            "DHAP",
            "FBP",
            "F6P",
            "G6P",
            "G1P",
            "SBP",
            "S7P",
            "E4P",
            "X5P",
            "R5P",
            "RUBP",
            "RU5P",
            "ATP",
            "Phosphate_total",
        ],
    )

    model.add_derived(
        name="N_pool",
        fn=n_export,
        args=[
            "Phosphate_pool",
            "PGA",
            "GAP",
            "DHAP",
            "K_pxt",
            "Phosphate_pool_ext",
            "K_pi",
            "K_pga",
            "K_gap",
            "K_dhap",
        ],
    )

    model.add_reaction(
        name="v1",
        fn=v1,
        stoichiometry={"RUBP": -1, "PGA": 2},
        args=[
            "RUBP",
            "PGA",
            "FBP",
            "SBP",
            "Phosphate_pool",
            "Vmax_1",
            "Km_1",
            "Ki_1_1",
            "Ki_1_2",
            "Ki_1_3",
            "Ki_1_4",
            "Ki_1_5",
            "NADPH_pool",
        ],
    )
    model.add_reaction(
        name="v2",
        fn=rapid_equilibrium_2s_2p,
        stoichiometry={"PGA": -1, "ATP": -1, "BPGA": 1},
        args=[
            "PGA",
            "ATP",
            "BPGA",
            "ADP",
            "k_rapid_eq",
            "q2",
        ],
    )
    model.add_reaction(
        name="v3",
        fn=v3,
        stoichiometry={"BPGA": -1, "GAP": 1},
        args=[
            "BPGA",
            "GAP",
            "Phosphate_pool",
            "proton_pool_stroma",
            "NADPH_pool",
            "NADP_pool",
            "k_rapid_eq",
            "q3",
        ],
    )
    model.add_reaction(
        name="v4",
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={"GAP": -1, "DHAP": 1},
        args=[
            "GAP",
            "DHAP",
            "k_rapid_eq",
            "q4",
        ],
    )
    model.add_reaction(
        name="v5",
        fn=rapid_equilibrium_2s_1p,
        stoichiometry={"GAP": -1, "DHAP": -1, "FBP": 1},
        args=[
            "GAP",
            "DHAP",
            "FBP",
            "k_rapid_eq",
            "q5",
        ],
    )
    model.add_reaction(
        name="v6",
        fn=v6,
        stoichiometry={"FBP": -1, "F6P": 1},
        args=[
            "FBP",
            "F6P",
            "Phosphate_pool",
            "Vmax_6",
            "Km_6",
            "Ki_6_1",
            "Ki_6_2",
        ],
    )
    model.add_reaction(
        name="v7",
        fn=rapid_equilibrium_2s_2p,
        stoichiometry={"GAP": -1, "F6P": -1, "E4P": 1, "X5P": 1},
        args=[
            "GAP",
            "F6P",
            "E4P",
            "X5P",
            "k_rapid_eq",
            "q7",
        ],
    )
    model.add_reaction(
        name="v8",
        fn=rapid_equilibrium_2s_1p,
        stoichiometry={"DHAP": -1, "E4P": -1, "SBP": 1},
        args=[
            "DHAP",
            "E4P",
            "SBP",
            "k_rapid_eq",
            "q8",
        ],
    )
    model.add_reaction(
        name="v9",
        fn=v9,
        stoichiometry={"SBP": -1, "S7P": 1},
        args=[
            "SBP",
            "Phosphate_pool",
            "Vmax_9",
            "Km_9",
            "Ki_9",
        ],
    )
    model.add_reaction(
        name="v10",
        fn=rapid_equilibrium_2s_2p,
        stoichiometry={"GAP": -1, "S7P": -1, "X5P": 1, "R5P": 1},
        args=[
            "GAP",
            "S7P",
            "X5P",
            "R5P",
            "k_rapid_eq",
            "q10",
        ],
    )
    model.add_reaction(
        name="v11",
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={"R5P": -1, "RU5P": 1},
        args=[
            "R5P",
            "RU5P",
            "k_rapid_eq",
            "q11",
        ],
    )
    model.add_reaction(
        name="v12",
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={"X5P": -1, "RU5P": 1},
        args=[
            "X5P",
            "RU5P",
            "k_rapid_eq",
            "q12",
        ],
    )
    model.add_reaction(
        name="v13",
        fn=v13,
        stoichiometry={"RU5P": -1, "ATP": -1, "RUBP": 1},
        args=[
            "RU5P",
            "ATP",
            "Phosphate_pool",
            "PGA",
            "RUBP",
            "ADP",
            "Vmax_13",
            "Km_13_1",
            "Km_13_2",
            "Ki_13_1",
            "Ki_13_2",
            "Ki_13_3",
            "Ki_13_4",
            "Ki_13_5",
        ],
    )
    model.add_reaction(
        name="v14",
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={"F6P": -1, "G6P": 1},
        args=[
            "F6P",
            "G6P",
            "k_rapid_eq",
            "q14",
        ],
    )
    model.add_reaction(
        name="v15",
        fn=rapid_equilibrium_1s_1p,
        stoichiometry={"G6P": -1, "G1P": 1},
        args=[
            "G6P",
            "G1P",
            "k_rapid_eq",
            "q15",
        ],
    )
    model.add_reaction(
        name="v16",
        fn=v16,
        stoichiometry={"ATP": 1},
        args=[
            "ADP",
            "Phosphate_pool",
            "Vmax_16",
            "Km_16_1",
            "Km_16_2",
        ],
    )
    model.add_reaction(
        name="vPGA_out",
        fn=v_out,
        stoichiometry={"PGA": -1},
        args=[
            "PGA",
            "N_pool",
            "Vmax_efflux",
            "K_pga",
        ],
    )
    model.add_reaction(
        name="vGAP_out",
        fn=v_out,
        stoichiometry={"GAP": -1},
        args=[
            "GAP",
            "N_pool",
            "Vmax_efflux",
            "K_gap",
        ],
    )
    model.add_reaction(
        name="vDHAP_out",
        fn=v_out,
        stoichiometry={"DHAP": -1},
        args=[
            "DHAP",
            "N_pool",
            "Vmax_efflux",
            "K_dhap",
        ],
    )
    model.add_reaction(
        name="vSt",
        fn=starch,
        stoichiometry={"G1P": -1, "ATP": -1},
        args=[
            "G1P",
            "ATP",
            "ADP",
            "Phosphate_pool",
            "PGA",
            "F6P",
            "FBP",
            "Vmax_starch",
            "Km_starch_1",
            "Km_starch_2",
            "Ki_starch",
            "Ka_starch_1",
            "Ka_starch_2",
            "Ka_starch_3",
        ],
    )
    return model
