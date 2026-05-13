"""Module containing functions for reactions and derived quatities."""

from __future__ import annotations

__all__ = [
    "add",
    "constant",
    "diffusion_1s_1p",
    "div",
    "hill_1s",
    "mass_action_1s",
    "mass_action_1s_1p",
    "mass_action_2s",
    "mass_action_2s_1p",
    "michaelis_menten_1s",
    "michaelis_menten_2s",
    "michaelis_menten_3s",
    "minus",
    "moiety_1s",
    "moiety_2s",
    "mul",
    "neg",
    "neg_div",
    "one_div",
    "proportional",
    "twice",
]


###############################################################################
# General functions
###############################################################################


def constant(x: float) -> float:
    """Return a constant value regardless of other model components.

    Parameters
    ----------
    x
        Value to return

    Returns
    -------
    Float
        The input value unchanged

    Examples
    --------
    >>> constant(5.0)
    5.0

    """
    return x


def neg(x: float) -> float:
    """Calculate the negation of a value.

    Parameters
    ----------
    x
        Value to negate

    Returns
    -------
    Float
        Negative of the input value

    Examples
    --------
    >>> neg(3.0)
    -3.0
    >>> neg(-2.5)
    2.5

    """
    return -x


def minus(x: float, y: float) -> float:
    """Calculate the difference between two values.

    Parameters
    ----------
    x
        Minuend (value to subtract from)
    y
        Subtrahend (value to subtract)

    Returns
    -------
    Float
        Difference between x and y (x - y)

    Examples
    --------
    >>> minus(5.0, 3.0)
    2.0
    >>> minus(2.0, 5.0)
    -3.0

    """
    return x - y


def mul(x: float, y: float) -> float:
    """Calculate the product of two values.

    Parameters
    ----------
    x
        First factor
    y
        Second factor

    Returns
    -------
    Float
        Product of x and y (x * y)

    Examples
    --------
    >>> mul(2.0, 3.0)
    6.0
    >>> mul(0.5, 4.0)
    2.0

    """
    return x * y


def div(x: float, y: float) -> float:
    """Calculate the quotient of two values.

    Parameters
    ----------
    x
        Numerator
    y
        Denominator

    Returns
    -------
    Float
        Quotient of x and y (x / y)

    Examples
    --------
    >>> div(6.0, 3.0)
    2.0
    >>> div(5.0, 2.0)
    2.5

    """
    return x / y


def one_div(x: float) -> float:
    """Calculate the reciprocal of a value.

    Parameters
    ----------
    x
        Value to find reciprocal of

    Returns
    -------
    Float
        Reciprocal of x (1 / x)

    Examples
    --------
    >>> one_div(2.0)
    0.5
    >>> one_div(4.0)
    0.25

    """
    return 1.0 / x


def neg_div(x: float, y: float) -> float:
    """Calculate the negative quotient of two values.

    Parameters
    ----------
    x
        Numerator
    y
        Denominator

    Returns
    -------
    Float
        Negative quotient of x and y (-x / y)

    Examples
    --------
    >>> neg_div(6.0, 3.0)
    -2.0
    >>> neg_div(-6.0, 3.0)
    2.0

    """
    return -x / y


def twice(x: float) -> float:
    """Calculate twice the value.

    Parameters
    ----------
    x
        Value to double

    Returns
    -------
    Float
        Double of the input value (x * 2)

    Examples
    --------
    >>> twice(3.5)
    7.0
    >>> twice(-1.5)
    -3.0

    """
    return x * 2


def add(x: float, y: float) -> float:
    """Calculate the sum of two values.

    Parameters
    ----------
    x
        First addend
    y
        Second addend

    Returns
    -------
    Float
        Sum of x and y (x + y)

    Examples
    --------
    >>> add(2.0, 3.0)
    5.0
    >>> add(-1.5, 3.5)
    2.0

    """
    return x + y


def proportional(x: float, y: float) -> float:
    """Calculate the product of two values.

    Common in mass-action kinetics where x represents a rate constant
    and y represents a substrate concentration.

    Parameters
    ----------
    x
        First factor (often rate constant)
    y
        Second factor (often concentration)

    Returns
    -------
    Float
        Product of x and y (x * y)

    Examples
    --------
    >>> proportional(0.5, 2.0)  # rate = 0.5 * [S]
    1.0
    >>> proportional(0.1, 5.0)
    0.5

    """
    return x * y


###############################################################################
# Derived functions
###############################################################################


def moiety_1s(
    x: float,
    x_total: float,
) -> float:
    """Calculate conservation relationship for one substrate.

    Used for creating derived variables that represent moiety conservation,
    such as calculating the free form of a species when you know the total.

    Parameters
    ----------
    x
        Concentration of one form of the species
    x_total
        Total concentration of all forms

    Returns
    -------
    Float
        Concentration of the other form (x_total - x)

    Examples
    --------
    >>> moiety_1s(0.3, 1.0)  # If total is 1.0 and one form is 0.3, other is 0.7
    0.7
    >>> # Example: If ATP + ADP = total_adenosine
    >>> moiety_1s(0.8, 1.5)  # [ADP] = total_adenosine - [ATP]
    0.7

    """
    return x_total - x


def moiety_2s(
    x1: float,
    x2: float,
    x_total: float,
) -> float:
    """Calculate conservation relationship for two substrates.

    Used for creating derived variables that represent moiety conservation
    across three species, where the third species concentration can be
    calculated from the total and the other two species.

    Parameters
    ----------
    x1
        Concentration of first form of the species
    x2
        Concentration of second form of the species
    x_total
        Total concentration of all forms

    Returns
    -------
    Float
        Concentration of the third form (x_total - x1 - x2)

    Examples
    --------
    >>> moiety_2s(0.3, 0.2, 1.0)  # If total is 1.0, first form is 0.3, second is 0.2
    0.5
    >>> # Example: If ATP + ADP + AMP = total_adenosine
    >>> moiety_2s(0.5, 0.3, 1.0)  # [AMP] = total_adenosine - [ATP] - [ADP]
    0.2

    """
    return x_total - x1 - x2


###############################################################################
# Reactions: mass action type
###############################################################################


def mass_action_1s(s1: float, k: float) -> float:
    """Calculate irreversible mass action reaction rate with one substrate.

    Rate = k * [S]

    Parameters
    ----------
    s1
        Substrate concentration
    k
        Rate constant

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> mass_action_1s(2.0, 0.5)  # Rate = 0.5 * [S]
    1.0
    >>> # Example: Simple degradation reaction S -> ∅
    >>> mass_action_1s(5.0, 0.2)  # Rate = 0.2 * [S]
    1.0

    """
    return k * s1


def mass_action_1s_1p(s1: float, p1: float, kf: float, kr: float) -> float:
    """Calculate reversible mass action reaction rate with one substrate and one product.

    Rate = kf * [S] - kr * [P]

    Parameters
    ----------
    s1
        Substrate concentration
    p1
        Product concentration
    kf
        Forward rate constant
    kr
        Reverse rate constant

    Returns
    -------
    Float
        Net reaction rate (positive for forward direction)

    Examples
    --------
    >>> # For reaction S ⇌ P
    >>> mass_action_1s_1p(2.0, 1.0, 0.5, 0.2)  # Rate = 0.5*[S] - 0.2*[P]
    0.8
    >>> # At equilibrium, rates balance
    >>> mass_action_1s_1p(2.0, 5.0, 0.5, 0.2)  # Rate = 0.5*2 - 0.2*5
    0.0

    """
    return kf * s1 - kr * p1


def mass_action_2s(s1: float, s2: float, k: float) -> float:
    """Calculate irreversible mass action reaction rate with two substrates.

    Rate = k * [S1] * [S2]

    Parameters
    ----------
    s1
        First substrate concentration
    s2
        Second substrate concentration
    k
        Rate constant

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> mass_action_2s(2.0, 3.0, 0.5)  # Rate = 0.5 * [S1] * [S2]
    3.0
    >>> # Example: Bimolecular reaction S1 + S2 -> P
    >>> mass_action_2s(1.0, 2.0, 0.25)  # Rate = 0.25 * [S1] * [S2]
    0.5

    """
    return k * s1 * s2


def mass_action_2s_1p(s1: float, s2: float, p1: float, kf: float, kr: float) -> float:
    """Calculate reversible mass action reaction rate with two substrates and one product.

    Rate = kf * [S1] * [S2] - kr * [P]

    Parameters
    ----------
    s1
        First substrate concentration
    s2
        Second substrate concentration
    p1
        Product concentration
    kf
        Forward rate constant
    kr
        Reverse rate constant

    Returns
    -------
    Float
        Net reaction rate (positive for forward direction)

    Examples
    --------
    >>> # For reaction S1 + S2 ⇌ P
    >>> mass_action_2s_1p(2.0, 1.5, 1.0, 0.5, 0.2)  # Rate = 0.5*[S1]*[S2] - 0.2*[P]
    1.3
    >>> # At equilibrium, rates balance
    >>> mass_action_2s_1p(2.0, 1.0, 5.0, 0.5, 0.5)  # Rate = 0.5*2*1 - 0.5*5
    -1.5

    """
    return kf * s1 * s2 - kr * p1


###############################################################################
# Reactions: michaelis-menten type
# For multi-molecular reactions use ping-pong kinetics as default
###############################################################################


def michaelis_menten_1s(s1: float, vmax: float, km1: float) -> float:
    """Calculate irreversible Michaelis-Menten reaction rate for one substrate.

    Rate = Vmax * [S] / (Km + [S])

    Parameters
    ----------
    s1
        Substrate concentration
    vmax
        Maximum reaction velocity
    km1
        Michaelis constant (substrate concentration at half-maximal rate)

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> michaelis_menten_1s(2.0, 10.0, 1.0)  # When [S]=2*Km, rate = 2/3 * Vmax
    6.666666666666667
    >>> michaelis_menten_1s(10.0, 5.0, 1.0)  # When [S]>>Km, rate ≈ Vmax
    4.545454545454546
    >>> michaelis_menten_1s(0.1, 5.0, 1.0)  # When [S]<<Km, rate ≈ Vmax*[S]/Km
    0.45454545454545453

    """
    return s1 * vmax / (s1 + km1)


# def michaelis_menten_1s_1i(
#     s: float,
#     i: float,
#     vmax: float,
#     km: float,
#     ki: float,
# ) -> float:
#     """Irreversible Michaelis-Menten equation for one substrate and one inhibitor."""
#     return vmax * s / (s + km * (1 + i / ki))


def michaelis_menten_2s(
    s1: float,
    s2: float,
    vmax: float,
    km1: float,
    km2: float,
) -> float:
    """Calculate Michaelis-Menten reaction rate (ping-pong) for two substrates.

    Rate = Vmax * [S1] * [S2] / ([S1]*[S2] + km1*[S2] + km2*[S1])

    This follows ping-pong kinetics, appropriate for reactions with
    multiple substrates where binding occurs in sequence.

    Parameters
    ----------
    s1
        First substrate concentration
    s2
        Second substrate concentration
    vmax
        Maximum reaction velocity
    km1
        Michaelis constant for first substrate
    km2
        Michaelis constant for second substrate

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> michaelis_menten_2s(2.0, 2.0, 10.0, 1.0, 1.0)  # Equal substrate concentrations
    5.0
    >>> michaelis_menten_2s(0.1, 10.0, 5.0, 1.0, 2.0)  # S1 is limiting
    0.4545454545454546
    >>> michaelis_menten_2s(10.0, 0.2, 5.0, 1.0, 2.0)  # S2 is limiting
    0.43478260869565216

    """
    return vmax * s1 * s2 / (s1 * s2 + km1 * s2 + km2 * s1)


def michaelis_menten_3s(
    s1: float,
    s2: float,
    s3: float,
    vmax: float,
    km1: float,
    km2: float,
    km3: float,
) -> float:
    """Calculate Michaelis-Menten reaction rate (ping-pong) for three substrates.

    Rate = Vmax * [S1] * [S2] * [S3] / ([S1]*[S2] + km1*[S2]*[S3] + km2*[S1]*[S3] + km3*[S1]*[S2])

    This follows ping-pong kinetics, appropriate for reactions with
    multiple substrates where binding occurs in sequence.

    Parameters
    ----------
    s1
        First substrate concentration
    s2
        Second substrate concentration
    s3
        Third substrate concentration
    vmax
        Maximum reaction velocity
    km1
        Michaelis constant for first substrate
    km2
        Michaelis constant for second substrate
    km3
        Michaelis constant for third substrate

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> michaelis_menten_3s(2.0, 2.0, 2.0, 10.0, 1.0, 1.0, 1.0)  # Equal substrate concentrations
    2.5
    >>> michaelis_menten_3s(0.1, 10.0, 10.0, 5.0, 1.0, 2.0, 2.0)  # S1 is limiting
    0.22727272727272727

    """
    return (
        vmax * s1 * s2 * s3 / (s1 * s2 + km1 * s2 * s3 + km2 * s1 * s3 + km3 * s1 * s2)
    )


def hill_1s(s: float, v: float, K: float, n: float) -> float:  # noqa: N803
    """Calculate Hill kinetics rate for one substrate.

    Rate = v * s^n / (K^n + s^n)

    Parameters
    ----------
    s
        Substrate concentration
    v
        Maximum reaction velocity
    K
        Half-saturation constant
    n
        Hill coefficient (cooperativity)

    Returns
    -------
    Float
        Reaction rate

    Examples
    --------
    >>> hill_1s(1.0, 10.0, 1.0, 1.0)  # Hill-1 reduces to Michaelis-Menten
    5.0
    >>> hill_1s(1.0, 10.0, 1.0, 2.0)  # Cooperative binding
    5.0
    >>> hill_1s(2.0, 10.0, 1.0, 2.0)
    8.0

    """
    sn = s**n
    return v * sn / (K**n + sn)


###############################################################################
# Reactions: diffusion type
###############################################################################


def diffusion_1s_1p(inside: float, outside: float, k: float) -> float:
    """Calculate diffusion rate between two compartments.

    Rate = k * ([outside] - [inside])

    Positive rate indicates flow from outside to inside.

    Parameters
    ----------
    inside
        Concentration inside the compartment
    outside
        Concentration outside the compartment
    k
        Diffusion rate constant

    Returns
    -------
    Float
        Net diffusion rate

    Examples
    --------
    >>> diffusion_1s_1p(1.0, 3.0, 0.5)  # Flow into the compartment
    1.0
    >>> diffusion_1s_1p(5.0, 2.0, 0.5)  # Flow out of the compartment
    -1.5
    >>> diffusion_1s_1p(3.0, 3.0, 0.5)  # No net flow at equilibrium
    0.0

    """
    return k * (outside - inside)
