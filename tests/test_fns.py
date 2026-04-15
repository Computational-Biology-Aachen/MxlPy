"""Tests for the mxlpy.fns module."""

from __future__ import annotations

import pytest

from mxlpy.fns import (
    add,
    constant,
    diffusion_1s_1p,
    div,
    mass_action_1s,
    mass_action_1s_1p,
    mass_action_2s,
    mass_action_2s_1p,
    michaelis_menten_1s,
    michaelis_menten_2s,
    michaelis_menten_3s,
    minus,
    moiety_1s,
    moiety_2s,
    mul,
    neg,
    neg_div,
    one_div,
    proportional,
    twice,
)


def test_constant() -> None:
    """Test the constant function."""
    assert constant(5.0) == 5.0
    assert constant(0.0) == 0.0
    assert constant(-2.0) == -2.0


def test_neg() -> None:
    """Test the negation function."""
    assert neg(5.0) == -5.0
    assert neg(0.0) == 0.0
    assert neg(-2.0) == 2.0


def test_minus() -> None:
    """Test the subtraction function."""
    assert minus(5.0, 3.0) == 2.0
    assert minus(5.0, 5.0) == 0.0
    assert minus(5.0, 10.0) == -5.0


def test_mul() -> None:
    """Test the multiplication function."""
    assert mul(5.0, 3.0) == 15.0
    assert mul(5.0, 0.0) == 0.0
    assert mul(-2.0, 3.0) == -6.0


def test_div() -> None:
    """Test the division function."""
    assert div(6.0, 3.0) == 2.0
    assert div(5.0, 2.0) == 2.5
    assert div(0.0, 5.0) == 0.0
    with pytest.raises(ZeroDivisionError):
        div(5.0, 0.0)


def test_one_div() -> None:
    """Test the reciprocal function."""
    assert one_div(2.0) == 0.5
    assert one_div(4.0) == 0.25
    assert one_div(1.0) == 1.0
    with pytest.raises(ZeroDivisionError):
        one_div(0.0)


def test_neg_div() -> None:
    """Test the negated division function."""
    assert neg_div(6.0, 3.0) == -2.0
    assert neg_div(0.0, 5.0) == 0.0
    assert neg_div(-6.0, 3.0) == 2.0
    with pytest.raises(ZeroDivisionError):
        neg_div(5.0, 0.0)


def test_twice() -> None:
    """Test the twice function."""
    assert twice(5.0) == 10.0
    assert twice(0.0) == 0.0
    assert twice(-3.0) == -6.0


def test_add() -> None:
    """Test the addition function."""
    assert add(5.0, 3.0) == 8.0
    assert add(5.0, 0.0) == 5.0
    assert add(-2.0, 5.0) == 3.0


def test_proportional() -> None:
    """Test the proportional function."""
    assert proportional(5.0, 3.0) == 15.0
    assert proportional(5.0, 0.0) == 0.0
    assert proportional(-2.0, 3.0) == -6.0


def test_moiety_1s() -> None:
    """Test the moiety_1s function."""
    assert moiety_1s(3.0, 10.0) == 7.0
    assert moiety_1s(0.0, 10.0) == 10.0
    assert moiety_1s(10.0, 10.0) == 0.0


def test_moiety_2s() -> None:
    """Test the moiety_2s function."""
    assert moiety_2s(3.0, 2.0, 10.0) == 5.0
    assert moiety_2s(0.0, 0.0, 10.0) == 10.0
    assert moiety_2s(5.0, 5.0, 10.0) == 0.0


def test_mass_action_1s() -> None:
    """Test the mass_action_1s function."""
    assert mass_action_1s(5.0, 2.0) == 10.0
    assert mass_action_1s(0.0, 2.0) == 0.0
    assert mass_action_1s(5.0, 0.0) == 0.0


def test_mass_action_1s_1p() -> None:
    """Test the mass_action_1s_1p function."""
    assert mass_action_1s_1p(5.0, 2.0, 3.0, 1.0) == 13.0
    assert mass_action_1s_1p(5.0, 2.0, 0.0, 1.0) == -2.0
    assert mass_action_1s_1p(5.0, 2.0, 3.0, 0.0) == 15.0
    assert mass_action_1s_1p(0.0, 0.0, 3.0, 1.0) == 0.0


def test_mass_action_2s() -> None:
    """Test the mass_action_2s function."""
    assert mass_action_2s(5.0, 2.0, 3.0) == 30.0
    assert mass_action_2s(0.0, 2.0, 3.0) == 0.0
    assert mass_action_2s(5.0, 0.0, 3.0) == 0.0
    assert mass_action_2s(5.0, 2.0, 0.0) == 0.0


def test_mass_action_2s_1p() -> None:
    """Test the mass_action_2s_1p function."""
    assert mass_action_2s_1p(5.0, 2.0, 3.0, 0.5, 1.0) == 2.0
    assert mass_action_2s_1p(0.0, 2.0, 3.0, 0.5, 1.0) == -3.0
    assert mass_action_2s_1p(5.0, 0.0, 3.0, 0.5, 1.0) == -3.0
    assert mass_action_2s_1p(5.0, 2.0, 0.0, 0.5, 0.0) == 5.0


def test_michaelis_menten_1s() -> None:
    """Test the michaelis_menten_1s function."""
    assert michaelis_menten_1s(10.0, 5.0, 2.0) == 50.0 / 12.0
    assert michaelis_menten_1s(0.0, 5.0, 2.0) == 0.0
    assert michaelis_menten_1s(10.0, 0.0, 2.0) == 0.0
    assert michaelis_menten_1s(10.0, 5.0, 0.0) == 5.0


def test_michaelis_menten_2s() -> None:
    """Test the michaelis_menten_2s function."""
    s1, s2 = 10.0, 5.0
    vmax, km1, km2 = 5.0, 2.0, 3.0
    expected = vmax * s1 * s2 / (s1 * s2 + km1 * s2 + km2 * s1)
    assert michaelis_menten_2s(s1, s2, vmax, km1, km2) == expected
    assert michaelis_menten_2s(0.0, s2, vmax, km1, km2) == 0.0
    assert michaelis_menten_2s(s1, 0.0, vmax, km1, km2) == 0.0
    assert michaelis_menten_2s(s1, s2, 0.0, km1, km2) == 0.0


def test_michaelis_menten_3s() -> None:
    """Test the michaelis_menten_3s function."""
    s1, s2, s3 = 10.0, 5.0, 2.0
    vmax, km1, km2, km3 = 5.0, 2.0, 3.0, 1.0
    expected = (
        vmax * s1 * s2 * s3 / (s1 * s2 + km1 * s2 * s3 + km2 * s1 * s3 + km3 * s1 * s2)
    )
    assert michaelis_menten_3s(s1, s2, s3, vmax, km1, km2, km3) == expected
    assert michaelis_menten_3s(0.0, s2, s3, vmax, km1, km2, km3) == 0.0
    assert michaelis_menten_3s(s1, 0.0, s3, vmax, km1, km2, km3) == 0.0
    assert michaelis_menten_3s(s1, s2, 0.0, vmax, km1, km2, km3) == 0.0
    assert michaelis_menten_3s(s1, s2, s3, 0.0, km1, km2, km3) == 0.0


def test_diffusion_1s_1p() -> None:
    """Test the diffusion_1s_1p function."""
    assert diffusion_1s_1p(5.0, 10.0, 2.0) == 10.0
    assert diffusion_1s_1p(10.0, 5.0, 2.0) == -10.0
    assert diffusion_1s_1p(5.0, 5.0, 2.0) == 0.0
    assert diffusion_1s_1p(5.0, 10.0, 0.0) == 0.0


###############################################################################
# Edge cases
###############################################################################


def test_michaelis_menten_1s_zero_substrate_and_km() -> None:
    """s=0 and km=0 simultaneously: 0/0 raises ZeroDivisionError.

    Formula: s * vmax / (s + km).  When both are zero the denominator is zero.
    The existing tests cover (s=0, km>0) and (s>0, km=0) separately but not
    the simultaneous case.
    """
    with pytest.raises(ZeroDivisionError):
        michaelis_menten_1s(0.0, 5.0, 0.0)


def test_michaelis_menten_2s_zero_both_substrates() -> None:
    """s1=0 and s2=0: denominator collapses to zero → ZeroDivisionError.

    Formula: vmax * s1 * s2 / (s1*s2 + km1*s2 + km2*s1).
    With s1=s2=0 every term in the denominator is zero.
    """
    with pytest.raises(ZeroDivisionError):
        michaelis_menten_2s(0.0, 0.0, 5.0, 1.0, 1.0)


def test_mass_action_1s_negative_concentration() -> None:
    """Negative concentration returns a negative rate (no guard).

    This is mathematically valid but physically impossible.  The function
    makes no non-negativity assertion.  Documenting the behavior prevents
    silent surprise when an integrator steps below zero.
    """
    assert mass_action_1s(-1.0, 2.0) == pytest.approx(-2.0)


def test_moiety_1s_negative_result() -> None:
    """x1 > total gives negative moiety value; no guard is applied."""
    assert moiety_1s(15.0, 10.0) == pytest.approx(-5.0)
