# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Sundials (CVODE via sundials4py) must agree with Scipy and behave like Assimulo."""

from __future__ import annotations

import gc
import weakref
from functools import partial
from typing import Literal

import numpy as np
import pytest

pytest.importorskip("sundials4py")

from mxlpy import Model, Scipy, Simulator
from mxlpy.fns import mass_action_2s
from mxlpy.integrators.int_sundials import Sundials, SundialsError
from mxlpy.types import IntegrationFailure, NoSteadyState, Result

from .test_int_native import chain, driven, lotka, robertson


def _tight_scipy() -> partial:
    return partial(Scipy, method="Radau", rtol=1e-11, atol=1e-11)


def _values(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
    res = sim.get_result().unwrap_or_err().variables
    return res.index.to_numpy(), res.to_numpy()


def _blow_up() -> Model:
    """y' = y^2 reaches infinity at t = 1."""
    return (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction("r", mass_action_2s, stoichiometry={"x": 1}, args=["x", "x", "k"])
    )


@pytest.mark.parametrize("method", ["BDF", "Adams"])
@pytest.mark.parametrize(
    ("factory", "t_end"),
    [(lotka, 20), (chain, 50), (robertson, 100), (driven, 10)],
)
def test_matches_scipy(method: Literal["BDF", "Adams"], factory, t_end: float) -> None:  # noqa: ANN001
    _, expected = _values(
        Simulator(factory(), integrator=_tight_scipy()).simulate(t_end)
    )
    time, got = _values(
        Simulator(
            factory(),
            integrator=partial(Sundials, method=method, rtol=1e-10, atol=1e-10),
        ).simulate(t_end)
    )
    assert time[-1] == t_end
    assert got[-1] == pytest.approx(expected[-1], rel=1e-5, abs=1e-6)


def test_steps_are_equidistant() -> None:
    time, _ = _values(Simulator(lotka(), integrator=Sundials).simulate(10, steps=5))
    np.testing.assert_allclose(time, [0, 2, 4, 6, 8, 10])


def test_requested_time_points_are_returned_exactly() -> None:
    time_points = np.array([0.0, 0.3, 1.7, 5.0])
    time, got = _values(
        Simulator(lotka(), integrator=Sundials).simulate_time_course(time_points)
    )
    np.testing.assert_array_equal(time, time_points)
    _, expected = _values(
        Simulator(lotka(), integrator=_tight_scipy()).simulate_time_course(time_points)
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_continues_across_calls() -> None:
    _, once = _values(Simulator(chain(), integrator=Sundials).simulate(10))
    time, twice = _values(
        Simulator(chain(), integrator=Sundials).simulate(4).simulate(10)
    )
    assert np.all(np.diff(time) > 0)
    assert twice[-1] == pytest.approx(once[-1], rel=1e-6)


def test_mixed_calls_continue() -> None:
    """Time courses and free-step integrations restart from each other's end state."""
    time, got = _values(
        Simulator(lotka(), integrator=Sundials)
        .simulate_time_course([0, 2.5])
        .simulate(2.5000001)
        .simulate(5)
    )
    assert time[-1] == 5
    _, expected = _values(Simulator(lotka(), integrator=_tight_scipy()).simulate(5))
    assert got[-1] == pytest.approx(expected[-1], rel=1e-5)


def test_reset() -> None:
    integrator = Sundials(lotka(), (2.0, 1.0))
    first = integrator.integrate(t_end=5).unwrap_or_err()
    integrator.reset()
    second = integrator.integrate(t_end=5).unwrap_or_err()
    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.values, second.values)


def test_analytic_jacobian_matches_finite_differences() -> None:
    _, fd = _values(Simulator(robertson(), integrator=Sundials).simulate(100))
    _, jac = _values(
        Simulator(robertson(), integrator=Sundials, use_jacobian=True).simulate(100)
    )
    assert jac[-1] == pytest.approx(fd[-1], rel=1e-5, abs=1e-8)


def test_steady_state() -> None:
    got = (
        Simulator(chain(), integrator=Sundials)
        .simulate_to_steady_state(tolerance=1e-8)
        .get_result()
        .unwrap_or_err()
        .variables.iloc[-1]
    )
    # Inflow balances loss: out * p = feed
    assert got["p"] == pytest.approx(0.2 / 0.3, rel=1e-6)


def test_no_steady_state_reported() -> None:
    res = Sundials(lotka(), (2.0, 1.0)).integrate_to_steady_state(
        tolerance=1e-12, rel_norm=False, oscillation_detector=None, t_max=3000
    )
    assert isinstance(res, Result)
    assert isinstance(res.value, NoSteadyState)


def test_blow_up_is_not_reported_as_success() -> None:
    res = Sundials(_blow_up(), (1.0,)).integrate(t_end=2)
    assert isinstance(res.value, IntegrationFailure)


def test_too_many_steps_fails() -> None:
    res = Sundials(lotka(), (2.0, 1.0), maxsteps=10).integrate(t_end=100, steps=1)
    assert isinstance(res.value, SundialsError)


def test_rhs_exception_is_returned_as_cause() -> None:
    def rhs(_t: float, _y: np.ndarray) -> tuple[float, ...]:
        msg = "bad rhs"
        raise ZeroDivisionError(msg)

    res = Sundials(rhs, (1.0,)).integrate(t_end=1)  # type: ignore[arg-type]
    assert isinstance(res.value, SundialsError)
    assert isinstance(res.value.__cause__, ZeroDivisionError)


def test_integrator_is_garbage_collected() -> None:
    """Scans create one integrator per model, so they must not leak."""
    integrator = Sundials(lotka(), (2.0, 1.0), jacobian=lambda _t, _y: np.eye(2))
    integrator.integrate(t_end=1)
    ref = weakref.ref(integrator)
    del integrator
    gc.collect()
    assert ref() is None


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        Sundials(lotka(), (2.0, 1.0), method="RK45")  # type: ignore[arg-type]


def test_decreasing_time_points_raise() -> None:
    with pytest.raises(ValueError, match="increasing"):
        Sundials(lotka(), (2.0, 1.0)).integrate_time_course(
            time_points=np.array([0.0, 2.0, 1.0])
        )


def test_matches_assimulo() -> None:
    """Both wrap CVODE with the same defaults."""
    pytest.importorskip("assimulo")
    from mxlpy.integrators.int_assimulo import Assimulo  # noqa: PLC0415

    time_points = np.linspace(0, 20, 41)
    _, expected = _values(
        Simulator(lotka(), integrator=Assimulo).simulate_time_course(time_points)
    )
    _, got = _values(
        Simulator(lotka(), integrator=Sundials).simulate_time_course(time_points)
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-8)
