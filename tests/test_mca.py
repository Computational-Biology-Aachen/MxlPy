# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Tests for the MCA module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mxlpy import (
    KineticModelBuilder,
    KineticRunner,
    Simulator,
    SteadyStateModelBuilder,
    fns,
)
from mxlpy.mca import (
    RateCharacteristics,
    SteadyStateStability,
    parameter_elasticities,
    rate_characteristics,
    response_coefficients,
    steady_state_stability,
    variable_elasticities,
)
from mxlpy.types import InitialAssignment

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.simulation import Simulation
    from mxlpy.types import Result


def linear_chain() -> KineticModelBuilder:
    """A -> B -> C open linear chain with mass-action kinetics."""
    return (
        KineticModelBuilder()
        .add_variables({"A": 5.0, "B": 2.0})
        .add_parameters({"k1": 1.0, "k2": 0.5, "k_in": 2.0})
        .add_reaction(
            "v_in",
            fn=fns.constant,
            args=["k_in"],
            stoichiometry={"A": 1.0},
        )
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["A", "k1"],
            stoichiometry={"A": -1.0, "B": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["B", "k2"],
            stoichiometry={"B": -1.0},
        )
    )


###############################################################################
# variable_elasticities
###############################################################################


def test_parameter_elasticities() -> None:
    # FIXME: implement this
    assert True


def test_parameter_elasticities_cross_terms_are_zero() -> None:
    """k1 does not appear in v2; elasticity must be zero."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v2", "k1"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc["v1", "k2"] == pytest.approx(0.0, abs=1e-4)


def test_parameter_elasticities_mass_action_normalized_equals_one() -> None:
    """For v = k*S, normalized elasticity w.r.t. k is exactly 1."""
    result = parameter_elasticities(linear_chain(), normalized=True)
    assert result.loc["v1", "k1"] == pytest.approx(1.0, rel=1e-3)
    assert result.loc["v2", "k2"] == pytest.approx(1.0, rel=1e-3)


def test_parameter_elasticities_returns_dataframe() -> None:
    result = parameter_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"k1", "k2", "k_in"}
    assert set(result.index) == {"v_in", "v1", "v2"}


def _reversible_with_outputs() -> KineticModelBuilder:
    """A <-> B (A + B conserved), with a derived quantity and a readout."""
    return (
        KineticModelBuilder()
        .add_parameters({"kf": 1.0, "kr": 0.5, "c": 2.0})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_derived("tot", fns.add, args=["A", "B"])
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "kf"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
        .add_readout("ro", fns.proportional, args=["A", "c"])
    )


@pytest.mark.parametrize("model_fn", [linear_chain, _reversible_with_outputs])
@pytest.mark.parametrize("normalized", [True, False])
def test_response_coefficients_forward_matches_finite_differences(
    model_fn: Callable[[], KineticModelBuilder], *, normalized: bool
) -> None:
    forward = response_coefficients(
        model_fn(), normalized=normalized, parallel=False
    ).combined
    fd = response_coefficients(
        model_fn(),
        normalized=normalized,
        parallel=False,
        runner=KineticRunner(derivatives="finite_difference", displacement=1e-4),
    ).combined
    assert list(forward.index) == list(fd.index)
    assert list(forward.columns) == list(fd.columns)
    np.testing.assert_allclose(forward, fd, atol=1e-5)


def test_response_coefficients_use_a_single_steady_state() -> None:
    calls: list[int] = []

    class Counting(KineticRunner):
        def steady_state(self, model: KineticModelBuilder) -> Result[Simulation]:
            calls.append(1)
            return super().steady_state(model)

    runner = Counting()
    response_coefficients(linear_chain(), parallel=False, runner=runner)
    # The forward sensitivities simulate through Simulator, not the runner
    assert calls == []
    response_coefficients(
        linear_chain(),
        parallel=False,
        runner=Counting(derivatives="finite_difference"),
    )
    # 2 per parameter + 1 for the normalisation, shared by all
    assert len(calls) == 2 * 3 + 1


def _interp_rate(x: float, k: float) -> float:
    # np.interp can't be parsed symbolically
    return k * float(np.interp(x, [0.0, 10.0], [0.0, 10.0]))


def test_response_coefficients_fall_back_to_finite_differences() -> None:
    model = linear_chain().update_reaction("v1", fn=_interp_rate)
    fd = response_coefficients(model, parallel=False).combined
    expected = response_coefficients(linear_chain(), parallel=False).combined
    np.testing.assert_allclose(fd, expected, atol=1e-4)


def test_response_coefficients_finite() -> None:
    """All response coefficients must be finite for a well-behaved model."""
    result = response_coefficients(linear_chain(), parallel=False)
    assert np.isfinite(result.variables.values).all()
    assert np.isfinite(result.fluxes.values).all()


def test_response_coefficients_returns_named_result() -> None:
    result = response_coefficients(linear_chain(), parallel=False)
    assert result.variables is not None
    assert result.fluxes is not None
    assert set(result.variables.columns) == {"k1", "k2", "k_in"}


def test_response_coefficients_zero_parameter() -> None:
    """A parameter at 0 gets an absolute finite-difference step, not a zero one."""
    model = (
        KineticModelBuilder()
        .add_variable("S", 1.0)
        .add_parameters({"k_active": 1.0, "k_zero": 0.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k_active"],
            stoichiometry={"S": -1.0},
        )
    )
    # S doesn't depend on k_zero
    for runner in (None, KineticRunner(derivatives="finite_difference")):
        result = response_coefficients(
            model, to_scan=["k_zero"], parallel=False, normalized=False, runner=runner
        )
        assert result.variables["k_zero"].to_numpy() == pytest.approx(0.0)


###############################################################################
# Time courses
###############################################################################

T = np.linspace(0, 5, 11)
FD = KineticRunner(derivatives="finite_difference", displacement=1e-4)


def _assert_same(a: pd.DataFrame, b: pd.DataFrame, atol: float) -> None:
    assert a.index.equals(b.index)
    assert list(a.columns) == list(b.columns)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=atol)


@pytest.mark.parametrize("model_fn", [linear_chain, _reversible_with_outputs])
@pytest.mark.parametrize("normalized", [True, False])
def test_time_course_response_coefficients_forward_matches_finite_differences(
    model_fn: Callable[[], KineticModelBuilder], *, normalized: bool
) -> None:
    forward = response_coefficients(
        model_fn(), time_points=T, normalized=normalized, parallel=False
    ).combined
    fd = response_coefficients(
        model_fn(), time_points=T, normalized=normalized, parallel=False, runner=FD
    ).combined
    _assert_same(forward, fd, atol=1e-5)


def test_time_course_response_coefficients_match_analytic() -> None:
    # dA/dt = k_in - k1 A, so dA(t)/dk_in = (1 - exp(-k1 t)) / k1 with k1 = 1
    rc = response_coefficients(
        linear_chain(), time_points=T, normalized=False, parallel=False
    )
    da = rc.for_parameter("k_in")
    assert isinstance(da, pd.DataFrame)
    assert list(da.index) == list(T)
    assert list(da.columns) == ["A", "B", "v_in", "v1", "v2"]
    np.testing.assert_allclose(da["A"], 1 - np.exp(-T), atol=1e-6)
    # Rows are (time, output); the variable / flux split works per time point
    assert list(rc.variables.loc[2.0].index) == ["A", "B"]
    assert list(rc.fluxes.loc[2.0].index) == ["v_in", "v1", "v2"]


def test_time_course_response_coefficients_approach_steady_state() -> None:
    late = response_coefficients(
        linear_chain(), time_points=[0.0, 60.0], parallel=False
    ).combined.loc[60.0]
    steady = response_coefficients(linear_chain(), parallel=False).combined
    _assert_same(late, steady, atol=1e-6)


def test_time_course_response_coefficients_fall_back_to_finite_differences() -> None:
    model = linear_chain().update_reaction("v1", fn=_interp_rate)
    fd = response_coefficients(model, time_points=T, parallel=False).combined
    expected = response_coefficients(
        linear_chain(), time_points=T, parallel=False
    ).combined
    np.testing.assert_allclose(fd, expected, atol=1e-4)


def test_time_course_response_coefficients_need_dynamics() -> None:
    model = SteadyStateModelBuilder().add_parameter("a", 1.0)
    with pytest.raises(TypeError, match="no dynamics"):
        response_coefficients(model, time_points=T)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    "elasticities", [variable_elasticities, parameter_elasticities]
)
@pytest.mark.parametrize("normalized", [True, False])
def test_elasticities_over_time_match_finite_differences(
    elasticities: Callable[..., pd.DataFrame], *, normalized: bool
) -> None:
    exact = elasticities(linear_chain(), time_points=T, normalized=normalized)
    fd = elasticities(
        linear_chain(),
        time_points=T,
        normalized=normalized,
        runner=KineticRunner(derivatives="finite_difference"),
    )
    _assert_same(exact, fd, atol=1e-8)
    assert exact.index.names == ["time", None]
    assert list(exact.index.get_level_values(1).unique()) == ["v_in", "v1", "v2"]


def test_elasticities_over_time_are_those_of_the_trajectory() -> None:
    model = _reversible_with_outputs()
    exact = variable_elasticities(model, time_points=T, normalized=True)
    states = (
        Simulator(model).simulate_time_course(T).get_result().unwrap_or_err().variables
    )
    t = T[4]
    at_t = variable_elasticities(
        model,
        variables={k: float(states.loc[t, k]) for k in ("A", "B")},
        time=t,
    )
    _assert_same(exact.loc[t], at_t, atol=1e-6)


def test_elasticities_over_time_fall_back_to_finite_differences() -> None:
    model = linear_chain().update_reaction("v1", fn=_interp_rate)
    fd = variable_elasticities(model, time_points=T)
    expected = variable_elasticities(linear_chain(), time_points=T)
    np.testing.assert_allclose(fd, expected, atol=1e-4)


def _assigned_rate_model() -> KineticModelBuilder:
    """v1 = k3 A with k3 = 2 k1 an initial assignment."""
    return (
        KineticModelBuilder()
        .add_variables({"A": 5.0})
        .add_parameters({"k1": 1.0, "k3": InitialAssignment(fn=fns.twice, args=["k1"])})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["A", "k3"], stoichiometry={"A": -1}
        )
    )


def test_parameter_elasticities_over_time_follow_initial_assignments() -> None:
    times = [0.0, 0.5, 1.0]
    exact = parameter_elasticities(
        _assigned_rate_model(), to_scan=["k1"], time_points=times, normalized=False
    )
    # dv1/dk1 = 2 A(t) with A(t) = 5 exp(-2 t)
    np.testing.assert_allclose(
        exact["k1"], 10 * np.exp(-2 * np.array(times)), rtol=1e-4
    )


def test_duplicate_time_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        response_coefficients(linear_chain(), time_points=[0, 1, 1], parallel=False)
    with pytest.raises(ValueError, match="duplicates"):
        variable_elasticities(linear_chain(), time_points=[0, 1, 1])


def test_time_courses_need_a_dynamic_runner() -> None:
    class SteadyOnly:
        def steady_state(self, model: KineticModelBuilder) -> Result[Simulation]:
            return KineticRunner().steady_state(model)

    with pytest.raises(TypeError, match="time courses"):
        response_coefficients(
            linear_chain(),
            time_points=T,
            runner=SteadyOnly(),  # type: ignore[arg-type]
        )


def test_elasticities_of_a_single_state_take_no_runner() -> None:
    with pytest.raises(TypeError, match="time_points"):
        variable_elasticities(linear_chain(), runner=KineticRunner())


def test_elasticities_take_a_time_or_time_points() -> None:
    with pytest.raises(ValueError, match="either"):
        variable_elasticities(linear_chain(), time=1.0, time_points=T)


def test_responsecoefficients_combined() -> None:
    # FIXME: implement this
    assert True


def test_responsecoefficientsbypars_combined() -> None:
    # FIXME: implement this
    assert True


def test_variable_elasticities() -> None:
    # FIXME: implement this
    assert True


def test_variable_elasticities_constant_reaction_is_zero() -> None:
    """Constant influx v_in does not depend on any variable."""
    result = variable_elasticities(linear_chain(), normalized=False)
    assert result.loc["v_in", "A"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc["v_in", "B"] == pytest.approx(0.0, abs=1e-6)


def test_variable_elasticities_mass_action_equals_k_unnormalized() -> None:
    """For mass-action v = k*S, unnormalized elasticity ≈ k (the derivative)."""
    model = linear_chain()
    result = variable_elasticities(model, normalized=False)
    k1 = model.get_parameter_values()["k1"]
    assert result.loc["v1", "A"] == pytest.approx(k1, rel=1e-3)
    assert result.loc["v2", "B"] == pytest.approx(
        model.get_parameter_values()["k2"], rel=1e-3
    )


def test_variable_elasticities_normalized_mass_action_equals_one() -> None:
    """For mass-action v = k*S, normalized elasticity = 1."""
    result = variable_elasticities(linear_chain(), normalized=True)
    assert result.loc["v1", "A"] == pytest.approx(1.0, rel=1e-3)
    assert result.loc["v2", "B"] == pytest.approx(1.0, rel=1e-3)


def test_variable_elasticities_returns_dataframe() -> None:
    result = variable_elasticities(linear_chain(), normalized=False)
    assert set(result.columns) == {"A", "B"}
    assert set(result.index) == {"v_in", "v1", "v2"}


def test_variable_elasticities_at_zero() -> None:
    """A variable at 0 gets an absolute finite-difference step, not a zero one."""
    model = (
        KineticModelBuilder()
        .add_variable("S", 0.0)  # zero initial condition
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k"],
            stoichiometry={"S": -1.0},
        )
    )
    result = variable_elasticities(model, normalized=False)
    # v1 = k S, so dv1/dS = k
    assert result.loc["v1", "S"] == pytest.approx(1.0)
    # Normalised elasticities are undefined at v1 = 0
    normalized = variable_elasticities(model)
    assert not np.isfinite(normalized["S"].to_numpy()).all()


###############################################################################
# steady_state_stability
###############################################################################


def test_steady_state_stability_returns_correct_type() -> None:
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    assert isinstance(result, SteadyStateStability)
    assert isinstance(result.eigenvalues, np.ndarray)
    assert isinstance(result.spectral_abscissa, float)
    assert isinstance(result.is_stable, bool)
    assert isinstance(result.has_oscillatory, bool)
    assert isinstance(result.classification, str)


def test_steady_state_stability_linear_chain_is_stable() -> None:
    """Open linear chain A→B→sink is unconditionally stable."""
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    assert result.is_stable
    assert result.spectral_abscissa < 0
    assert result.classification in {"stable node", "stable spiral"}


def test_steady_state_stability_unstable_model() -> None:
    """Autocatalytic model dS/dt = k*S has positive eigenvalue → unstable."""
    model = (
        KineticModelBuilder()
        .add_variable("S", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k"],
            stoichiometry={"S": 1.0},
        )
    )
    result = steady_state_stability(model, {"S": 1.0})
    assert not result.is_stable
    assert result.spectral_abscissa > 0
    assert result.classification in {"unstable node", "unstable spiral", "saddle point"}


def test_steady_state_stability_spectral_abscissa_matches_eigenvalues() -> None:
    ss = {"A": 2.0, "B": 4.0}
    result = steady_state_stability(linear_chain(), ss)
    max_real = max(float(ev.real) for ev in result.eigenvalues)
    assert result.spectral_abscissa == pytest.approx(max_real)


###############################################################################
# rate_characteristics
###############################################################################


def test_rate_characteristics_returns_result_type() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert isinstance(result, RateCharacteristics)
    assert isinstance(result.steady_state_conc, float)


def test_rate_characteristics_supply_demand_columns() -> None:
    """B is produced by v1 (supply) and consumed by v2 (demand)."""
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert list(result.supply_fluxes.columns) == ["v1"]
    assert list(result.demand_fluxes.columns) == ["v2"]


def test_rate_characteristics_n_points() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=32, parallel=False)
    assert len(result.supply_fluxes) == 32
    assert len(result.demand_fluxes) == 32


def test_rate_characteristics_steady_state_conc() -> None:
    """At steady state k1*A = v_in and k2*B = k1*A, so A=2, B=4."""
    model = linear_chain()
    ss = (
        Simulator(model)
        .simulate_to_steady_state()
        .get_result()
        .unwrap_or_err()
        .get_new_y0()
    )
    result = rate_characteristics(model, "B", n_points=16, parallel=False)
    assert result.steady_state_conc == pytest.approx(ss["B"])
    assert result.steady_state_conc == pytest.approx(4.0, rel=1e-3)


def test_rate_characteristics_scan_bounds() -> None:
    """Scan spans [ss / min_factor, ss * max_factor] on a log scale."""
    result = rate_characteristics(
        linear_chain(),
        "B",
        min_factor=10.0,
        max_factor=100.0,
        n_points=16,
        parallel=False,
    )
    concs = result.demand_fluxes.index.to_numpy()
    ss = result.steady_state_conc
    assert concs[0] == pytest.approx(ss / 10.0)
    assert concs[-1] == pytest.approx(ss * 100.0)


def test_rate_characteristics_demand_increases_with_concentration() -> None:
    """Demand flux v2 = k2 * B is monotonically increasing in B."""
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    demand = result.total_demand.to_numpy()
    assert np.all(np.diff(demand) > 0)


def test_rate_characteristics_totals_are_series() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    assert result.total_supply.shape == (16,)
    assert result.total_demand.shape == (16,)


def test_rate_characteristics_no_reaction_raises() -> None:
    """A variable that participates in no reaction is rejected."""
    model = (
        KineticModelBuilder()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v1",
            fn=fns.constant,
            args=["k"],
            stoichiometry={},
        )
    )
    with pytest.raises(ValueError, match="does not participate"):
        rate_characteristics(model, "x", n_points=4, parallel=False)


def test_rate_characteristics_plot_supply_demand() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    fig, ax = result.plot_supply_demand()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_rate_characteristics_plot() -> None:
    result = rate_characteristics(linear_chain(), "B", n_points=16, parallel=False)
    fig, axs = result.plot()
    assert isinstance(fig, Figure)
    assert len(axs) == 3
    plt.close(fig)
