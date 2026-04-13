"""Tests for generate_model_components_py with an explicit outputs subset."""

from __future__ import annotations

import pytest

from mxlpy import Model, fns
from mxlpy.meta.codegen_model import generate_model_components_py


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _model_1v_1p_1d_1r() -> Model:
    """v1, p1 → d1 = v1+p1 → r1 = d1*v1."""
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_derived("d1", fn=fns.add, args=["v1", "p1"])
        .add_reaction("r1", fn=fns.mass_action_1s, args=["v1", "d1"], stoichiometry={"v1": -1.0})
    )


def _model_2d_1r() -> Model:
    """v1 → d1 → d2 → r1.  Two-level derived chain."""
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_derived("d1", fn=fns.constant, args=["v1"])
        .add_derived("d2", fn=fns.constant, args=["d1"])
        .add_reaction("r1", fn=fns.mass_action_1s, args=["v1", "d2"], stoichiometry={"v1": -1.0})
    )


def _model_with_readout() -> Model:
    """v1, v2 → r1, r2; readout ro1 depends on r1 and r2."""

    def ratio(a: float, b: float) -> float:
        return a / b

    return (
        Model()
        .add_variables({"v1": 1.0, "v2": 2.0})
        .add_parameter("k", 0.1)
        .add_reaction("r1", fn=fns.mass_action_1s, args=["k", "v1"], stoichiometry={"v1": -1.0})
        .add_reaction("r2", fn=fns.mass_action_1s, args=["k", "v2"], stoichiometry={"v2": -1.0})
        .add_readout("ro1", fn=ratio, args=["r1", "r2"])
    )


# ---------------------------------------------------------------------------
# outputs=None (default) — baseline, all components emitted
# ---------------------------------------------------------------------------


def test_outputs_none_emits_all_components() -> None:
    code = generate_model_components_py(_model_1v_1p_1d_1r(), outputs=None)
    lines = code.split("\n")
    assert "    d1: float = p1 + v1" in lines
    assert "    r1: float = d1*v1" in lines
    assert "    return d1, r1" in lines


# ---------------------------------------------------------------------------
# Single output — reaction only, no intermediate deps
# ---------------------------------------------------------------------------


def test_outputs_single_reaction_no_intermediate_deps() -> None:
    """r1 depends only on k and v1 (leaves) so nothing extra is emitted."""
    m = (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("k", 0.5)
        .add_reaction("r1", fn=fns.mass_action_1s, args=["k", "v1"], stoichiometry={"v1": -1.0})
    )
    lines = generate_model_components_py(m, outputs=["r1"]).split("\n")
    assert "    r1: float = k*v1" in lines
    assert "    return r1" in lines


# ---------------------------------------------------------------------------
# Single output — derived value
# ---------------------------------------------------------------------------


def test_outputs_single_derived() -> None:
    lines = generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["d1"]).split("\n")
    assert "    d1: float = p1 + v1" in lines
    assert "    return d1" in lines
    # r1 must NOT be emitted
    assert not any("r1" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Transitive dependency: reaction depends on a derived value
# ---------------------------------------------------------------------------


def test_outputs_reaction_pulls_in_derived_dep() -> None:
    """Requesting r1 must also emit d1 as an intermediate."""
    lines = generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["r1"]).split("\n")
    # d1 is a dep — it must appear before r1
    assert "    d1: float = p1 + v1" in lines
    assert "    r1: float = d1*v1" in lines
    # only r1 in the return
    assert "    return r1" in lines
    d1_idx = next(i for i, ln in enumerate(lines) if "d1: float" in ln)
    r1_idx = next(i for i, ln in enumerate(lines) if "r1: float" in ln)
    assert d1_idx < r1_idx


# ---------------------------------------------------------------------------
# Transitive dependency: two-level derived chain
# ---------------------------------------------------------------------------


def test_outputs_transitive_two_level_derived_chain() -> None:
    """Requesting r1 from d1→d2→r1 chain must emit d1 and d2 in order."""
    lines = generate_model_components_py(_model_2d_1r(), outputs=["r1"]).split("\n")
    assert any("d1" in ln for ln in lines)
    assert any("d2" in ln for ln in lines)
    assert "    return r1" in lines
    d1_idx = next(i for i, ln in enumerate(lines) if "d1: float" in ln)
    d2_idx = next(i for i, ln in enumerate(lines) if "d2: float" in ln)
    r1_idx = next(i for i, ln in enumerate(lines) if "r1: float" in ln)
    assert d1_idx < d2_idx < r1_idx


def test_outputs_intermediate_dep_not_in_return() -> None:
    """d1 and d2 are intermediates for r1 — they must not appear in the return."""
    lines = generate_model_components_py(_model_2d_1r(), outputs=["r1"]).split("\n")
    return_line = next(ln for ln in lines if ln.strip().startswith("return"))
    assert "d1" not in return_line
    assert "d2" not in return_line
    assert "r1" in return_line


# ---------------------------------------------------------------------------
# Readout with reaction dependencies
# ---------------------------------------------------------------------------


def test_outputs_readout_pulls_in_reaction_deps() -> None:
    """Requesting ro1 must emit r1 and r2 as intermediates."""
    lines = generate_model_components_py(_model_with_readout(), outputs=["ro1"]).split("\n")
    assert any("r1: float" in ln for ln in lines)
    assert any("r2: float" in ln for ln in lines)
    assert any("ro1: float" in ln for ln in lines)
    assert "    return ro1" in lines
    # reactions come before readout
    r1_idx = next(i for i, ln in enumerate(lines) if "r1: float" in ln)
    ro1_idx = next(i for i, ln in enumerate(lines) if "ro1: float" in ln)
    assert r1_idx < ro1_idx


# ---------------------------------------------------------------------------
# Multiple outputs
# ---------------------------------------------------------------------------


def test_outputs_multiple_returns_in_requested_order() -> None:
    lines = generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["d1", "r1"]).split("\n")
    assert "    return d1, r1" in lines


def test_outputs_multiple_reversed_order() -> None:
    """Return order follows the outputs list, not emission order."""
    lines = generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["r1", "d1"]).split("\n")
    assert "    return r1, d1" in lines


# ---------------------------------------------------------------------------
# Subset omits unrelated components
# ---------------------------------------------------------------------------


def test_outputs_unrelated_components_not_emitted() -> None:
    """Requesting only d1 must not emit r1 at all."""
    m = _model_1v_1p_1d_1r()
    lines = generate_model_components_py(m, outputs=["d1"]).split("\n")
    assert not any("r1" in ln for ln in lines)


def test_outputs_independent_reactions_not_emitted() -> None:
    """Two independent reactions; requesting one omits the other."""
    m = (
        Model()
        .add_variables({"v1": 1.0, "v2": 2.0})
        .add_parameter("k", 0.1)
        .add_reaction("r1", fn=fns.mass_action_1s, args=["k", "v1"], stoichiometry={"v1": -1.0})
        .add_reaction("r2", fn=fns.mass_action_1s, args=["k", "v2"], stoichiometry={"v2": -1.0})
    )
    lines = generate_model_components_py(m, outputs=["r1"]).split("\n")
    assert any("r1: float" in ln for ln in lines)
    assert not any("r2: float" in ln for ln in lines)
    assert "    return r1" in lines


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_outputs_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown output components"):
        generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["nonexistent"])


def test_outputs_partially_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown output components"):
        generate_model_components_py(_model_1v_1p_1d_1r(), outputs=["d1", "ghost"])
