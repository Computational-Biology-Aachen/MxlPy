from __future__ import annotations

import sympy

from mxlpy import KineticModelBuilder, OdeModelBuilder, SteadyStateModelBuilder, fns

# sympy.Expr support has exactly one binding rule everywhere it appears: an
# expression's free symbol names are used *literally* to look up model
# components. There is no separate `args` list to remap them through -
# `add_derived`/`add_reaction`/`add_readout` keep taking plain functions
# with an explicit `args` list, and `add_derived_from_expr`/
# `add_reaction_from_expr`/`add_readout_from_expr` take a sympy expression
# with no `args` list at all. This avoids the ambiguity of binding an
# expression's (unordered) free symbols to an external `args` list.


def test_add_parameter_from_expr_uses_literal_names() -> None:
    k1, km = sympy.symbols("k1 km")
    m = (
        KineticModelBuilder()
        .add_parameters({"k1": 2.0, "km": 4.0})
        .add_parameter("keq", k1 / km)
    )
    assert m.get_args()["keq"] == 0.5


def test_add_variable_from_expr_uses_literal_names() -> None:
    k1 = sympy.symbols("k1")
    m = KineticModelBuilder().add_parameters({"k1": 3.0}).add_variables({"v1": k1 * 2})
    assert m.get_initial_conditions()["v1"] == 6.0


def test_stoichiometry_expr_uses_literal_names() -> None:
    stoich = sympy.symbols("stoich")
    m = (
        KineticModelBuilder()
        .add_parameters({"stoich": -2.0, "k": 1.0})
        .add_variables({"x": 1.0})
        .add_reaction(
            "v1",
            fn=fns.proportional,
            args=["x", "k"],
            stoichiometry={"x": stoich},
        )
    )
    assert m.get_stoichiometries().loc["x", "v1"] == -2.0


def test_add_derived_from_expr_uses_literal_names() -> None:
    """The expression's own symbols must exist as model components."""
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        KineticModelBuilder()
        .add_parameters({"k1": 3.0, "k2": 5.0})
        .add_derived_from_expr("diff", k1 - k2)
    )
    assert m.get_args()["diff"] == 3.0 - 5.0


def test_add_reaction_from_expr_uses_literal_names() -> None:
    s, vmax, km = sympy.symbols("s vmax km")
    m = (
        KineticModelBuilder()
        .add_parameters({"vmax": 2.0, "km": 0.5})
        .add_variables({"s": 1.0})
        .add_reaction_from_expr(
            "v1",
            vmax * s / (km + s),
            stoichiometry={"s": -1},
        )
    )
    expected = 2.0 * 1.0 / (0.5 + 1.0)
    assert m.get_fluxes()["v1"] == expected


def test_add_readout_from_expr_uses_literal_names() -> None:
    nadph, nadp_total = sympy.symbols("nadph nadp_total")
    m = (
        KineticModelBuilder()
        .add_parameters({"nadph": 1.0, "nadp_total": 4.0})
        .add_readout_from_expr("energy_state", nadph / nadp_total)
    )
    readout = m.get_raw_readouts()["energy_state"]
    values = m.get_args()
    assert readout.fn(*(values[a] for a in readout.args)) == 0.25


def test_update_derived_from_expr_replaces_fn_and_args() -> None:
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        KineticModelBuilder()
        .add_parameters({"k1": 3.0, "k2": 5.0})
        .add_derived_from_expr("d", k1 + k2)
    )
    assert m.get_args()["d"] == 3.0 + 5.0

    m.update_derived_from_expr("d", k1 - k2)
    assert m.get_args()["d"] == 3.0 - 5.0


def test_update_reaction_from_expr_replaces_fn_and_args() -> None:
    s, k = sympy.symbols("s k")
    m = (
        KineticModelBuilder()
        .add_parameters({"k": 2.0})
        .add_variables({"s": 3.0})
        .add_reaction_from_expr("v1", k * s, stoichiometry={"s": -1})
    )
    assert m.get_fluxes()["v1"] == 6.0

    m.update_reaction_from_expr("v1", k / s)
    assert m.get_fluxes()["v1"] == 2.0 / 3.0


def test_ode_builder_add_derived_from_expr_uses_literal_names() -> None:
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        OdeModelBuilder()
        .add_parameters({"k1": 3.0, "k2": 5.0})
        .add_derived_from_expr("diff", k1 - k2)
    )
    derived = m.get_raw_derived()["diff"]
    values = m.get_parameter_values()
    assert derived.fn(*(values[a] for a in derived.args)) == 3.0 - 5.0


def test_ode_builder_add_readout_from_expr_uses_literal_names() -> None:
    nadph, nadp_total = sympy.symbols("nadph nadp_total")
    m = (
        OdeModelBuilder()
        .add_parameters({"nadph": 1.0, "nadp_total": 4.0})
        .add_readout_from_expr("energy_state", nadph / nadp_total)
    )
    readout = m.get_raw_readouts()["energy_state"]
    values = m.get_parameter_values()
    assert readout.fn(*(values[a] for a in readout.args)) == 0.25


def test_steady_state_builder_add_derived_from_expr_uses_literal_names() -> None:
    # SteadyStateModelBuilder has no cache-backed evaluation yet, so check
    # the raw Derived directly against the raw parameter values instead.
    k1, k2 = sympy.symbols("k1 k2")
    m = (
        SteadyStateModelBuilder()
        .add_parameters({"k1": 3.0, "k2": 5.0})
        .add_derived_from_expr("diff", k1 - k2)
    )
    derived = m.get_raw_derived()["diff"]
    values = {name: p.value for name, p in m.get_raw_parameters().items()}
    assert derived.fn(*(values[a] for a in derived.args)) == 3.0 - 5.0
