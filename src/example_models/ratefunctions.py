# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Collection of rate functions for enzyme kinetics."""

from __future__ import annotations

__all__ = [
    "competitive_activation",
    "competitive_inhibition",
    "constant",
    "hill",
    "mass_action_1",
    "mass_action_2",
    "mass_action_3",
    "mass_action_4",
    "mass_action_variable",
    "michaelis_menten",
    "mixed_activation",
    "mixed_inhibition",
    "noncompetitive_activation",
    "noncompetitive_inhibition",
    "ordered_2",
    "ordered_2_2",
    "ping_pong_2",
    "ping_pong_3",
    "ping_pong_4",
    "random_order_2",
    "random_order_2_2",
    "reversible_mass_action_1_1",
    "reversible_mass_action_1_2",
    "reversible_mass_action_1_3",
    "reversible_mass_action_1_4",
    "reversible_mass_action_2_1",
    "reversible_mass_action_2_2",
    "reversible_mass_action_2_3",
    "reversible_mass_action_2_4",
    "reversible_mass_action_3_1",
    "reversible_mass_action_3_2",
    "reversible_mass_action_3_3",
    "reversible_mass_action_3_4",
    "reversible_mass_action_4_1",
    "reversible_mass_action_4_2",
    "reversible_mass_action_4_3",
    "reversible_mass_action_4_4",
    "reversible_michaelis_menten",
    "reversible_michaelis_menten_keq",
    "reversible_noncompetitive_inhibition",
    "reversible_noncompetitive_inhibition_keq",
    "reversible_uncompetitive_inhibition",
    "reversible_uncompetitive_inhibition_keq",
    "uncompetitive_activation",
    "uncompetitive_inhibition",
]


from functools import reduce
from operator import mul


def constant(k: float) -> float:
    """Constant function."""
    return k


###############################################################################
# Mass action
###############################################################################


def mass_action_1(s1: float, k_fwd: float) -> float:
    """Mass action equation for one substrate."""
    return k_fwd * s1


def mass_action_2(s1: float, s2: float, k_fwd: float) -> float:
    """Mass action equation for two substrates."""
    return k_fwd * s1 * s2


def mass_action_3(s1: float, s2: float, s3: float, k_fwd: float) -> float:
    """Mass action equation for three substrates."""
    return k_fwd * s1 * s2 * s3


def mass_action_4(s1: float, s2: float, s3: float, s4: float, k_fwd: float) -> float:
    """Mass action equation for four substrates."""
    return k_fwd * s1 * s2 * s3 * s4


def mass_action_variable(*args: float) -> float:
    """Mass action equation for variable number of substrates."""
    return reduce(mul, args, 1)


###############################################################################
# Reversible Mass action
###############################################################################


def reversible_mass_action_1_1(
    s1: float,
    p1: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for one substrate and one product."""
    return k_fwd * s1 - k_bwd * p1


def reversible_mass_action_2_1(
    s1: float,
    s2: float,
    p1: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for two substrates and one product."""
    return k_fwd * s1 * s2 - k_bwd * p1


def reversible_mass_action_3_1(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for three substrates and one product."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1


def reversible_mass_action_4_1(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    p1: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for four substrates and one product."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1


def reversible_mass_action_1_2(
    s1: float,
    p1: float,
    p2: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for one substrate and two products."""
    return k_fwd * s1 - k_bwd * p1 * p2


def reversible_mass_action_2_2(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for two substrates and two products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2


def reversible_mass_action_3_2(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    p2: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for three substrates and two products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2


def reversible_mass_action_4_2(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    p1: float,
    p2: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for four substrates and two products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2


def reversible_mass_action_1_3(
    s1: float,
    p1: float,
    p2: float,
    p3: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for one substrate and three products."""
    return k_fwd * s1 - k_bwd * p1 * p2 * p3


def reversible_mass_action_2_3(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    p3: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for two substrates and three products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2 * p3


def reversible_mass_action_3_3(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    p2: float,
    p3: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for three substrates and three products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2 * p3


def reversible_mass_action_4_3(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    p1: float,
    p2: float,
    p3: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for four substrates and three products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2 * p3


def reversible_mass_action_1_4(
    s1: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for one substrate and four products."""
    return k_fwd * s1 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_2_4(
    s1: float,
    s2: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for two substrates and four products."""
    return k_fwd * s1 * s2 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_3_4(
    s1: float,
    s2: float,
    s3: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for three substrates and four products."""
    return k_fwd * s1 * s2 * s3 - k_bwd * p1 * p2 * p3 * p4


def reversible_mass_action_4_4(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    k_fwd: float,
    k_bwd: float,
) -> float:
    """Reversible mass action equation for four substrates and four products."""
    return k_fwd * s1 * s2 * s3 * s4 - k_bwd * p1 * p2 * p3 * p4


###############################################################################
# Michaelis Menten
###############################################################################


def michaelis_menten(s: float, vmax: float, km: float) -> float:
    """Irreversible Michaelis-Menten equation for one substrate."""
    return s * vmax / (s + km)


def competitive_inhibition(
    s: float, i: float, vmax: float, km: float, ki: float
) -> float:
    """Competitive inhibition."""
    return vmax * s / (s + km * (1 + i / ki))


def competitive_activation(
    s: float, a: float, vmax: float, km: float, ka: float
) -> float:
    """Competitive activation."""
    return vmax * s / (s + km * (1 + ka / a))


def uncompetitive_inhibition(
    s: float, i: float, vmax: float, km: float, ki: float
) -> float:
    """Uncompetitive inhibition."""
    return vmax * s / (s * (1 + i / ki) + km)


def uncompetitive_activation(
    s: float, a: float, vmax: float, km: float, ka: float
) -> float:
    """Uncompetitive activation."""
    return vmax * s / (s * (1 + ka / a) + km)


def noncompetitive_inhibition(
    s: float, i: float, vmax: float, km: float, ki: float
) -> float:
    """Noncompetitive inhibition."""
    return vmax * s / ((s + km) * (1 + i / ki))


def noncompetitive_activation(
    s: float, a: float, vmax: float, km: float, ka: float
) -> float:
    """Noncompetitive activation."""
    return vmax * s / ((s + km) * (1 + ka / a))


def mixed_inhibition(s: float, i: float, vmax: float, km: float, ki: float) -> float:
    """Mixed inhibition."""
    return vmax * s / (s * (1 + i / ki) + km * (1 + i / ki))


def mixed_activation(s: float, a: float, vmax: float, km: float, ka: float) -> float:
    """Mixed activation."""
    return vmax * s / (s * (1 + ka / a) + km * (1 + ka / a))


###############################################################################
# Reversible Michaelis-Menten
###############################################################################


def reversible_michaelis_menten(
    s: float,
    p: float,
    vmax_fwd: float,
    vmax_bwd: float,
    kms: float,
    kmp: float,
) -> float:
    """Reversible Michaelis-Menten equation for one substrate and one product."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (1 + s / kms + p / kmp)


def reversible_uncompetitive_inhibition(
    s: float,
    p: float,
    i: float,
    vmax_fwd: float,
    vmax_bwd: float,
    kms: float,
    kmp: float,
    ki: float,
) -> float:
    """Reversible uncompetitive inhibition."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (
        1 + (s / kms) + (p / kmp) * (1 + i / ki)
    )


def reversible_noncompetitive_inhibition(
    s: float,
    p: float,
    i: float,
    vmax_fwd: float,
    vmax_bwd: float,
    kms: float,
    kmp: float,
    ki: float,
) -> float:
    """Reversible noncompetitive inhibition."""
    return (vmax_fwd * s / kms - vmax_bwd * p / kmp) / (
        (1 + s / kms + p / kmp) * (1 + i / ki)
    )


def reversible_michaelis_menten_keq(
    s: float,
    p: float,
    vmax_fwd: float,
    kms: float,
    kmp: float,
    keq: float,
) -> float:
    """Reversible Michaelis-Menten equation for one substrate and one product."""
    return vmax_fwd / kms * (s - p / keq) / (1 + s / kms + p / kmp)


def reversible_uncompetitive_inhibition_keq(
    s: float,
    p: float,
    i: float,
    vmax_fwd: float,
    kms: float,
    kmp: float,
    ki: float,
    keq: float,
) -> float:
    """Reversible uncompetitive inhibition."""
    return vmax_fwd / kms * (s - p / keq) / (1 + (s / kms) + (p / kmp) * (1 + i / ki))


def reversible_noncompetitive_inhibition_keq(
    s: float,
    p: float,
    i: float,
    vmax_fwd: float,
    kms: float,
    kmp: float,
    ki: float,
    keq: float,
) -> float:
    """Reversible noncompetitive inhibition."""
    return vmax_fwd / kms * (s - p / keq) / ((1 + s / kms + p / kmp) * (1 + i / ki))


###############################################################################
# Multi-substrate
###############################################################################


def ordered_2(
    a: float,
    b: float,
    vmax: float,
    kma: float,
    kmb: float,
    kia: float,
) -> float:
    """Ordered Bi Bi reaction."""
    return vmax * a * b / (a * b + kmb * a + kma * b + kia * kmb)


def ordered_2_2(
    a: float,
    b: float,
    p: float,
    q: float,
    vmaxf: float,
    vmaxr: float,
    kma: float,
    kmb: float,
    kmp: float,
    kmq: float,
    kia: float,
    kib: float,
    kip: float,
    kiq: float,
) -> float:
    """Ordered Bi Bi reaction."""
    nominator = vmaxf * a * b / (kia * kmb) - vmaxr * p * q / (kmp * kiq)
    denominator = (
        1
        + (a / kia)
        + (kma * b / (kia * kmb))
        + (kmq * p / (kmp * kiq))
        + (q / kiq)
        + (a * b / (kia * kmb))
        + (kmq * a * p / (kia * kmp * kiq))
        + (kma * b * q / (kia * kmb * kiq))
        + (p * q / (kmp * kiq))
        + (a * b * p / (kia * kmb * kip))
        + (b * p * q) / (kib * kmp * kiq)
    )
    return nominator / denominator


def random_order_2(
    a: float,
    b: float,
    vmax: float,
    kma: float,
    kmb: float,
    kia: float,
) -> float:
    """Random-order reaction with two substrates."""
    return vmax * a * b / (a * b + kmb * a + kma * b + kia * kmb)


def random_order_2_2(
    a: float,
    b: float,
    p: float,
    q: float,
    vmaxf: float,
    vmaxr: float,
    kmb: float,
    kmp: float,
    kia: float,
    kib: float,
    kip: float,
    kiq: float,
) -> float:
    """Random-order reaction with two substrates and two products."""
    nominator = vmaxf * a * b / (kia * kmb) - vmaxr * p * q / (kmp * kiq)
    denominator = (
        1
        + (a / kia)
        + (b / kib)
        + (p / kip)
        + (q / kiq)
        + (a * b / (kia * kmb))
        + (p * q / (kmp * kiq))
    )
    return nominator / denominator


def ping_pong_2(
    a: float,
    b: float,
    vmax: float,
    kma: float,
    kmb: float,
) -> float:
    """Ping-pong reaction with two substrates."""
    return vmax * a * b / (a * b + kma * b + kmb * a)


def ping_pong_3(
    a: float,
    b: float,
    c: float,
    vmax: float,
    kma: float,
    kmb: float,
    kmc: float,
) -> float:
    """Ping-pong reaction with three substrates."""
    return (vmax * a * b * c) / (
        a * b * c + (kma * b * c) + (kmb * a * c) + (kmc * a * b)
    )


def ping_pong_4(
    a: float,
    b: float,
    c: float,
    d: float,
    vmax: float,
    kma: float,
    kmb: float,
    kmc: float,
    kmd: float,
) -> float:
    """Ping-pong reaction with four substrates."""
    return (vmax * a * b * c * d) / (
        a * b * c * d
        + (kma * b * c * d)
        + (kmb * a * c * d)
        + (kmc * a * b * d)
        + (kmd * a * b * c)
    )


###############################################################################
# cooperativity
###############################################################################


def hill(s: float, vmax: float, kd: float, n: float) -> float:
    """Hill equation."""
    return vmax * s**n / (kd + s**n)


###############################################################################
# Generalised
###############################################################################

# def hanekom()-> float:
#     pass

# def convenience()-> float:
#     pass
