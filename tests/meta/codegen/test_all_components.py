"""Matrix tests: every model-component x every codegen function."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from mxlpy import Model, fns, units
from mxlpy.meta import (
    generate_latex_code,
    generate_model_code_jl,
    generate_model_code_py,
    generate_model_code_rs,
    generate_model_code_ts,
    generate_mxlpy_code,
    generate_mxlpy_code_raw,
    generate_mxlweb_page,
)
from mxlpy.surrogates import qss
from mxlpy.types import InitialAssignment

# ---------------------------------------------------------------------------
# Shared model factories
# ---------------------------------------------------------------------------


def _m_vars() -> Model:
    return Model().add_variable("v1", 1.0)


def _m_pars() -> Model:
    return Model().add_parameter("p1", 1.0)


def _m_derived() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_derived("d1", fn=fns.add, args=["v1", "p1"])
    )


def _m_reactions() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_reaction(
            "r1",
            fn=fns.mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
        )
    )


def _qss_two(v1: float, p1: float) -> tuple[float, float]:
    return v1 * p1, v1 + p1


def _m_surrogates() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_surrogate(
            "qss1",
            qss.Surrogate(
                model=_qss_two,
                args=["v1", "p1"],
                outputs=["out1", "out2"],
                stoichiometries={"out1": {"v1": -1.0}},
            ),
        )
    )


def _readout_ratio(a: float, b: float) -> float:
    return a / b


def _m_readouts() -> Model:
    return (
        Model()
        .add_variables({"v1": 1.0, "v2": 2.0})
        .add_parameter("p1", 1.0)
        .add_reaction(
            "r1",
            fn=fns.mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
        )
        .add_reaction(
            "r2",
            fn=fns.mass_action_1s,
            args=["v2", "p1"],
            stoichiometry={"v2": -1.0},
        )
        .add_readout("ro1", fn=_readout_ratio, args=["r1", "r2"])
    )


def _ia_init(p1: float) -> float:
    return p1 * 2.0


def _m_ia_var() -> Model:
    return (
        Model()
        .add_parameter("p1", 1.0)
        .add_variable("v1", InitialAssignment(fn=_ia_init, args=["p1"]))
    )


def _p_init(v1: float) -> float:
    return v1 + 1.0


def _m_ia_par() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", InitialAssignment(fn=_p_init, args=["v1"]))
    )


# ===========================================================================
# ROW: vars
# ===========================================================================


def test_vars_latex() -> None:
    code = generate_latex_code(_m_vars())
    assert r"\subsection*{Variables}" in code
    assert "v1 & 1.00e+00" in code


def test_vars_jl() -> None:
    lines = generate_model_code_jl(_m_vars()).split("\n")
    assert "    v1 = variables" in lines


def test_vars_py() -> None:
    lines = generate_model_code_py(_m_vars()).split("\n")
    assert "    v1 = variables" in lines


def test_vars_rs() -> None:
    lines = generate_model_code_rs(_m_vars()).split("\n")
    assert "    let [v1] = *variables;" in lines


def test_vars_ts() -> None:
    lines = generate_model_code_ts(_m_vars()).split("\n")
    assert "    let [v1] = variables;" in lines


def test_vars_mxlpy() -> None:
    code = generate_mxlpy_code(_m_vars())
    assert ".add_variable('v1', initial_value=1.0)" in code


def test_vars_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_vars())
    assert ".add_variable('v1', initial_value=1.0)" in code


def test_vars_mxlweb() -> None:
    code = generate_mxlweb_page(_m_vars())
    assert '.addVariable("v1", { value: 1.0,' in code


# ===========================================================================
# ROW: pars
# ===========================================================================


def test_pars_latex() -> None:
    code = generate_latex_code(_m_pars())
    assert r"\subsection*{Parameters}" in code
    assert "p1 & 1.00e+00" in code


def test_pars_jl() -> None:
    lines = generate_model_code_jl(_m_pars()).split("\n")
    assert "    p1 = 1.0" in lines


def test_pars_py() -> None:
    lines = generate_model_code_py(_m_pars()).split("\n")
    assert "    p1: float = 1.0" in lines


def test_pars_rs() -> None:
    lines = generate_model_code_rs(_m_pars()).split("\n")
    assert "    let p1: f64 = 1.0;" in lines


def test_pars_ts() -> None:
    lines = generate_model_code_ts(_m_pars()).split("\n")
    assert "    let p1: number = 1.0;" in lines


def test_pars_mxlpy() -> None:
    code = generate_mxlpy_code(_m_pars())
    assert ".add_parameter('p1', value=1.0)" in code


def test_pars_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_pars())
    assert ".add_parameter('p1', value=1.0)" in code


def test_pars_mxlweb() -> None:
    code = generate_mxlweb_page(_m_pars())
    assert '.addParameter("p1", { value: 1.0,' in code


# ===========================================================================
# ROW: derived
# ===========================================================================


def test_derived_latex() -> None:
    code = generate_latex_code(_m_derived())
    assert r"\subsection*{Derived}" in code
    assert r"\mathrm{d1} &=" in code


def test_derived_jl() -> None:
    lines = generate_model_code_jl(_m_derived()).split("\n")
    assert any(ln.startswith("    d1 = ") for ln in lines)


def test_derived_py() -> None:
    lines = generate_model_code_py(_m_derived()).split("\n")
    assert any(ln.startswith("    d1: float = ") for ln in lines)


def test_derived_rs() -> None:
    lines = generate_model_code_rs(_m_derived()).split("\n")
    assert any(ln.startswith("    let d1: f64 = ") for ln in lines)


def test_derived_ts() -> None:
    lines = generate_model_code_ts(_m_derived()).split("\n")
    assert any(ln.startswith("    let d1: number = ") for ln in lines)


def test_derived_mxlpy() -> None:
    code = generate_mxlpy_code(_m_derived())
    assert ".add_derived(" in code
    assert "'d1'" in code
    assert "fn=add" in code


def test_derived_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_derived())
    assert ".add_derived(" in code
    assert "'d1'" in code
    assert "fn=add" in code


def test_derived_mxlweb() -> None:
    code = generate_mxlweb_page(_m_derived())
    assert '.addAssignment("d1",' in code


# ===========================================================================
# ROW: reactions
# ===========================================================================


def test_reactions_latex() -> None:
    code = generate_latex_code(_m_reactions())
    assert r"\subsection*{Reactions}" in code
    assert r"\mathrm{r1} &=" in code
    # Differential equation section present (spaces → underscores via _name_to_latex)
    assert r"\subsection*{Differential\_equations}" in code


def test_reactions_jl() -> None:
    lines = generate_model_code_jl(_m_reactions()).split("\n")
    assert any(ln.startswith("    r1 = ") for ln in lines)
    assert "    dv1dt = -r1" in lines
    assert "    return [dv1dt]" in lines


def test_reactions_py() -> None:
    lines = generate_model_code_py(_m_reactions()).split("\n")
    assert any(ln.startswith("    r1: float = ") for ln in lines)
    assert "    dv1dt: float = -r1" in lines
    assert "    return dv1dt" in lines


def test_reactions_rs() -> None:
    lines = generate_model_code_rs(_m_reactions()).split("\n")
    assert any(ln.startswith("    let r1: f64 = ") for ln in lines)
    assert "    let dv1dt: f64 = -r1;" in lines
    assert "    return [dv1dt]" in lines


def test_reactions_ts() -> None:
    lines = generate_model_code_ts(_m_reactions()).split("\n")
    assert any(ln.startswith("    let r1: number = ") for ln in lines)
    assert "    let dv1dt: number = -r1;" in lines
    assert "    return [dv1dt];" in lines


def test_reactions_mxlpy() -> None:
    code = generate_mxlpy_code(_m_reactions())
    assert ".add_reaction(" in code
    assert '"r1"' in code
    assert "fn=mass_action_1s" in code


def test_reactions_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_reactions())
    assert ".add_reaction(" in code
    assert '"r1"' in code
    assert "fn=mass_action_1s" in code


def test_reactions_mxlweb() -> None:
    code = generate_mxlweb_page(_m_reactions())
    assert '.addReaction("r1",' in code


# ===========================================================================
# ROW: surrogates
# ===========================================================================


def test_surrogates_latex() -> None:
    # Surrogates silently ignored in LaTeX export (has FIXME in to_tex_export)
    code = generate_latex_code(_m_surrogates())
    assert "out1" not in code
    assert "out2" not in code


def test_surrogates_jl() -> None:
    lines = generate_model_code_jl(_m_surrogates()).split("\n")
    assert any(ln.startswith("    out1 = ") for ln in lines)
    assert any(ln.startswith("    out2 = ") for ln in lines)
    assert "    dv1dt = -out1" in lines


def test_surrogates_py() -> None:
    lines = generate_model_code_py(_m_surrogates()).split("\n")
    assert any(ln.startswith("    out1: float = ") for ln in lines)
    assert any(ln.startswith("    out2: float = ") for ln in lines)
    assert "    dv1dt: float = -out1" in lines


def test_surrogates_rs() -> None:
    lines = generate_model_code_rs(_m_surrogates()).split("\n")
    assert any(ln.startswith("    let out1: f64 = ") for ln in lines)
    assert any(ln.startswith("    let out2: f64 = ") for ln in lines)
    assert "    let dv1dt: f64 = -out1;" in lines


def test_surrogates_ts() -> None:
    lines = generate_model_code_ts(_m_surrogates()).split("\n")
    assert any(ln.startswith("    let out1: number = ") for ln in lines)
    assert any(ln.startswith("    let out2: number = ") for ln in lines)
    assert "    let dv1dt: number = -out1;" in lines


def test_surrogates_mxlpy(caplog: pytest.LogCaptureFixture) -> None:
    # Surrogates are warned and skipped; surrogate must not appear in output
    with caplog.at_level(logging.WARNING):
        code = generate_mxlpy_code(_m_surrogates())
    assert "Surrogates" in caplog.text
    assert "add_surrogate" not in code
    assert "qss1" not in code


def test_surrogates_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_surrogates())
    assert ".add_surrogate(" in code
    assert "'qss1'" in code
    assert "qss.Surrogate(" in code
    assert "outputs=['out1', 'out2']" in code


def test_surrogates_mxlweb() -> None:
    code = generate_mxlweb_page(_m_surrogates())
    # out1 has stoichiometry → addReaction; out2 is plain → addAssignment
    assert '"out1"' in code
    assert '"out2"' in code


# ===========================================================================
# ROW: readouts
# ===========================================================================


def test_readouts_latex() -> None:
    # Readouts absent from LaTeX export; reactions still appear
    code = generate_latex_code(_m_readouts())
    assert "ro1" not in code
    assert r"\mathrm{r1} &=" in code


def test_readouts_jl() -> None:
    lines = generate_model_code_jl(_m_readouts()).split("\n")
    assert not any("ro1" in ln for ln in lines)
    # Reactions and their diff eqs ARE present
    assert any("r1" in ln for ln in lines)


def test_readouts_py() -> None:
    lines = generate_model_code_py(_m_readouts()).split("\n")
    assert not any("ro1" in ln for ln in lines)
    assert any("r1" in ln for ln in lines)


def test_readouts_rs() -> None:
    lines = generate_model_code_rs(_m_readouts()).split("\n")
    assert not any("ro1" in ln for ln in lines)
    assert any("r1" in ln for ln in lines)


def test_readouts_ts() -> None:
    lines = generate_model_code_ts(_m_readouts()).split("\n")
    assert not any("ro1" in ln for ln in lines)
    assert any("r1" in ln for ln in lines)


def test_readouts_mxlpy() -> None:
    code = generate_mxlpy_code(_m_readouts())
    assert "ro1" not in code
    assert "add_reaction" in code


def test_readouts_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_readouts())
    assert "ro1" not in code
    assert "add_reaction" in code


def test_readouts_mxlweb() -> None:
    code = generate_mxlweb_page(_m_readouts())
    assert "ro1" not in code
    assert "addReaction" in code


# ===========================================================================
# ROW: ia_var (variable with InitialAssignment)
# ===========================================================================


def test_ia_var_latex() -> None:
    # Variable is evaluated to float; appears in Variables table
    code = generate_latex_code(_m_ia_var())
    assert r"\subsection*{Variables}" in code
    assert "v1" in code


def test_ia_var_jl() -> None:
    # Variable evaluated at init time; appears in ODE unpack, no expression
    lines = generate_model_code_jl(_m_ia_var()).split("\n")
    assert "    v1 = variables" in lines


def test_ia_var_py() -> None:
    lines = generate_model_code_py(_m_ia_var()).split("\n")
    assert "    v1 = variables" in lines


def test_ia_var_rs() -> None:
    lines = generate_model_code_rs(_m_ia_var()).split("\n")
    assert "    let [v1] = *variables;" in lines


def test_ia_var_ts() -> None:
    lines = generate_model_code_ts(_m_ia_var()).split("\n")
    assert "    let [v1] = variables;" in lines


def test_ia_var_mxlpy() -> None:
    # generate_mxlpy_code reconstructs InitialAssignment via symbolic repr
    code = generate_mxlpy_code(_m_ia_var())
    assert "InitialAssignment" in code
    assert "add_variable(" in code
    assert "'v1'" in code


def test_ia_var_mxlpy_raw() -> None:
    # generate_mxlpy_code_raw preserves InitialAssignment using inspect.getsource
    code = generate_mxlpy_code_raw(_m_ia_var())
    assert "InitialAssignment" in code
    assert "add_variable(" in code
    assert "'v1'" in code


def test_ia_var_mxlweb() -> None:
    # Variable with InitialAssignment emitted with computed expression (not bare float)
    code = generate_mxlweb_page(_m_ia_var())
    # value must be a TS AST expression, not a plain number
    assert '.addVariable("v1", { value: new ' in code


# ===========================================================================
# ROW: ia_par (parameter with InitialAssignment)
# ===========================================================================


def test_ia_par_latex() -> None:
    # InitialAssignment params excluded from get_parameter_values() → absent from table
    code = generate_latex_code(_m_ia_par())
    assert "p1" not in code


def test_ia_par_jl() -> None:
    # InitialAssignment params excluded from get_parameter_values() → absent from ODE body
    lines = generate_model_code_jl(_m_ia_par()).split("\n")
    assert not any("p1" in ln for ln in lines)


def test_ia_par_py() -> None:
    lines = generate_model_code_py(_m_ia_par()).split("\n")
    assert not any("p1" in ln for ln in lines)


def test_ia_par_rs() -> None:
    lines = generate_model_code_rs(_m_ia_par()).split("\n")
    assert not any("p1" in ln for ln in lines)


def test_ia_par_ts() -> None:
    lines = generate_model_code_ts(_m_ia_par()).split("\n")
    assert not any("p1" in ln for ln in lines)


def test_ia_par_mxlpy() -> None:
    # generate_mxlpy_code uses get_raw_parameters() so InitialAssignment is preserved
    code = generate_mxlpy_code(_m_ia_par())
    assert "InitialAssignment" in code
    assert "add_parameter(" in code
    assert "'p1'" in code


def test_ia_par_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_ia_par())
    assert "InitialAssignment" in code
    assert "add_parameter(" in code
    assert "'p1'" in code


def test_ia_par_mxlweb() -> None:
    # InitialAssignment params excluded from get_parameter_values() → absent from page
    code = generate_mxlweb_page(_m_ia_par())
    assert "addParameter" not in code
    assert "p1" not in code


# ===========================================================================
# ROW: vars_unit  (variable with unit annotation)
# ===========================================================================


def _m_vars_unit() -> Model:
    return Model().add_variable("v1", 1.0, unit=units.liter)


def test_vars_unit_latex() -> None:
    # Units not exported in LaTeX; variable still appears in table
    code = generate_latex_code(_m_vars_unit())
    assert r"\subsection*{Variables}" in code
    assert "v1 & 1.00e+00" in code


def test_vars_unit_jl() -> None:
    # Units irrelevant to ODE body
    lines = generate_model_code_jl(_m_vars_unit()).split("\n")
    assert "    v1 = variables" in lines


def test_vars_unit_py() -> None:
    lines = generate_model_code_py(_m_vars_unit()).split("\n")
    assert "    v1 = variables" in lines


def test_vars_unit_rs() -> None:
    lines = generate_model_code_rs(_m_vars_unit()).split("\n")
    assert "    let [v1] = *variables;" in lines


def test_vars_unit_ts() -> None:
    lines = generate_model_code_ts(_m_vars_unit()).split("\n")
    assert "    let [v1] = variables;" in lines


def test_vars_unit_mxlpy() -> None:
    # generate_mxlpy_code preserves units for variables
    code = generate_mxlpy_code(_m_vars_unit())
    assert "from mxlpy import" in code
    assert "units" in code
    assert "unit=units.liter" in code


def test_vars_unit_mxlpy_raw() -> None:
    # generate_mxlpy_code_raw does NOT preserve units (known limitation)
    code = generate_mxlpy_code_raw(_m_vars_unit())
    assert "add_variable('v1'" in code
    assert "unit=" not in code


def test_vars_unit_mxlweb() -> None:
    # Units not part of mxlweb page format
    code = generate_mxlweb_page(_m_vars_unit())
    assert '.addVariable("v1",' in code


# ===========================================================================
# ROW: pars_unit  (parameter with unit annotation)
# ===========================================================================


def _m_pars_unit() -> Model:
    return Model().add_parameter("p1", 1.0, unit=units.second)


def test_pars_unit_latex() -> None:
    code = generate_latex_code(_m_pars_unit())
    assert r"\subsection*{Parameters}" in code
    assert "p1 & 1.00e+00" in code


def test_pars_unit_jl() -> None:
    lines = generate_model_code_jl(_m_pars_unit()).split("\n")
    assert "    p1 = 1.0" in lines


def test_pars_unit_py() -> None:
    lines = generate_model_code_py(_m_pars_unit()).split("\n")
    assert "    p1: float = 1.0" in lines


def test_pars_unit_rs() -> None:
    lines = generate_model_code_rs(_m_pars_unit()).split("\n")
    assert "    let p1: f64 = 1.0;" in lines


def test_pars_unit_ts() -> None:
    lines = generate_model_code_ts(_m_pars_unit()).split("\n")
    assert "    let p1: number = 1.0;" in lines


def test_pars_unit_mxlpy() -> None:
    # generate_mxlpy_code preserves units for parameters
    code = generate_mxlpy_code(_m_pars_unit())
    assert "units" in code
    assert "unit=units.second" in code


def test_pars_unit_mxlpy_raw() -> None:
    # generate_mxlpy_code_raw does NOT preserve units
    code = generate_mxlpy_code_raw(_m_pars_unit())
    assert "add_parameter('p1'" in code
    assert "unit=" not in code


def test_pars_unit_mxlweb() -> None:
    code = generate_mxlweb_page(_m_pars_unit())
    assert '.addParameter("p1",' in code


# ===========================================================================
# ROW: derived_unit  (derived with unit annotation)
# ===========================================================================


def _m_derived_unit() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_derived("d1", fn=fns.add, args=["v1", "p1"], unit=units.liter)
    )


def test_derived_unit_latex() -> None:
    code = generate_latex_code(_m_derived_unit())
    assert r"\mathrm{d1} &=" in code


def test_derived_unit_jl() -> None:
    lines = generate_model_code_jl(_m_derived_unit()).split("\n")
    assert any(ln.startswith("    d1 = ") for ln in lines)


def test_derived_unit_py() -> None:
    lines = generate_model_code_py(_m_derived_unit()).split("\n")
    assert any(ln.startswith("    d1: float = ") for ln in lines)


def test_derived_unit_rs() -> None:
    lines = generate_model_code_rs(_m_derived_unit()).split("\n")
    assert any(ln.startswith("    let d1: f64 = ") for ln in lines)


def test_derived_unit_ts() -> None:
    lines = generate_model_code_ts(_m_derived_unit()).split("\n")
    assert any(ln.startswith("    let d1: number = ") for ln in lines)


def test_derived_unit_mxlpy() -> None:
    # SymbolicFn has no unit field; unit is NOT preserved for derived
    code = generate_mxlpy_code(_m_derived_unit())
    assert "add_derived(" in code
    assert "'d1'" in code
    assert "unit=" not in code


def test_derived_unit_mxlpy_raw() -> None:
    # unit NOT preserved for derived
    code = generate_mxlpy_code_raw(_m_derived_unit())
    assert "add_derived(" in code
    assert "'d1'" in code
    assert "unit=" not in code


def test_derived_unit_mxlweb() -> None:
    code = generate_mxlweb_page(_m_derived_unit())
    assert '.addAssignment("d1",' in code


# ===========================================================================
# ROW: rxn_unit  (reaction with unit annotation)
# ===========================================================================


def _m_rxn_unit() -> Model:
    return (
        Model()
        .add_variable("v1", 1.0)
        .add_parameter("p1", 1.0)
        .add_reaction(
            "r1",
            fn=fns.mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
            unit=units.liter,
        )
    )


def test_rxn_unit_latex() -> None:
    code = generate_latex_code(_m_rxn_unit())
    assert r"\mathrm{r1} &=" in code


def test_rxn_unit_jl() -> None:
    lines = generate_model_code_jl(_m_rxn_unit()).split("\n")
    assert any(ln.startswith("    r1 = ") for ln in lines)
    assert "    dv1dt = -r1" in lines


def test_rxn_unit_py() -> None:
    lines = generate_model_code_py(_m_rxn_unit()).split("\n")
    assert any(ln.startswith("    r1: float = ") for ln in lines)
    assert "    dv1dt: float = -r1" in lines


def test_rxn_unit_rs() -> None:
    lines = generate_model_code_rs(_m_rxn_unit()).split("\n")
    assert any(ln.startswith("    let r1: f64 = ") for ln in lines)
    assert "    let dv1dt: f64 = -r1;" in lines


def test_rxn_unit_ts() -> None:
    lines = generate_model_code_ts(_m_rxn_unit()).split("\n")
    assert any(ln.startswith("    let r1: number = ") for ln in lines)
    assert "    let dv1dt: number = -r1;" in lines


def test_rxn_unit_mxlpy() -> None:
    # unit NOT preserved for reactions
    code = generate_mxlpy_code(_m_rxn_unit())
    assert "add_reaction(" in code
    assert '"r1"' in code
    assert "unit=" not in code


def test_rxn_unit_mxlpy_raw() -> None:
    # unit NOT preserved for reactions
    code = generate_mxlpy_code_raw(_m_rxn_unit())
    assert "add_reaction(" in code
    assert '"r1"' in code
    assert "unit=" not in code


def test_rxn_unit_mxlweb() -> None:
    code = generate_mxlweb_page(_m_rxn_unit())
    assert '.addReaction("r1",' in code


# ===========================================================================
# ROW: readout_unit  (readout with unit annotation)
# ===========================================================================


def _m_readout_unit() -> Model:
    return (
        Model()
        .add_variables({"v1": 1.0, "v2": 2.0})
        .add_parameter("p1", 1.0)
        .add_reaction(
            "r1",
            fn=fns.mass_action_1s,
            args=["v1", "p1"],
            stoichiometry={"v1": -1.0},
        )
        .add_reaction(
            "r2",
            fn=fns.mass_action_1s,
            args=["v2", "p1"],
            stoichiometry={"v2": -1.0},
        )
        .add_readout("ro1", fn=_readout_ratio, args=["r1", "r2"], unit=units.liter)
    )


def test_readout_unit_latex() -> None:
    # Readouts not exported; no crash
    code = generate_latex_code(_m_readout_unit())
    assert "ro1" not in code


def test_readout_unit_jl() -> None:
    lines = generate_model_code_jl(_m_readout_unit()).split("\n")
    assert not any("ro1" in ln for ln in lines)


def test_readout_unit_py() -> None:
    lines = generate_model_code_py(_m_readout_unit()).split("\n")
    assert not any("ro1" in ln for ln in lines)


def test_readout_unit_rs() -> None:
    lines = generate_model_code_rs(_m_readout_unit()).split("\n")
    assert not any("ro1" in ln for ln in lines)


def test_readout_unit_ts() -> None:
    lines = generate_model_code_ts(_m_readout_unit()).split("\n")
    assert not any("ro1" in ln for ln in lines)


def test_readout_unit_mxlpy() -> None:
    code = generate_mxlpy_code(_m_readout_unit())
    assert "ro1" not in code


def test_readout_unit_mxlpy_raw() -> None:
    code = generate_mxlpy_code_raw(_m_readout_unit())
    assert "ro1" not in code


def test_readout_unit_mxlweb() -> None:
    code = generate_mxlweb_page(_m_readout_unit())
    assert "ro1" not in code
