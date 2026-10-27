# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import numpy as np
import pandas as pd
import pytest

from mxlpy import fit
from mxlpy.minimizers import FixedNMinimizer


def mock_residual_fn(
    updates: dict[str, float],  # noqa: ARG001
) -> float:
    return 0.0


def test_default_minimizer() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_fit = fit.LocalScipyMinimizer()(
        mock_residual_fn,
        p_true,
        bounds={},
    ).unwrap_or_err()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.parameters), pd.Series(p_true), rtol=0.1)


def quadratic_residual(updates: dict[str, float]) -> float:
    """Simple quadratic bowl: minimum at x=0, y=0."""
    return updates["x"] ** 2 + updates["y"] ** 2


def quadratic_residual_1d(updates: dict[str, float]) -> float:
    """Single-parameter quadratic: minimum at x=0."""
    return updates["x"] ** 2


def test_fixed_n_minimizer_returns_valid_result() -> None:
    p0 = {"x": 2.0, "y": 3.0}
    state = FixedNMinimizer(n_steps=5)(
        quadratic_residual, p0, bounds={}
    ).unwrap_or_err()
    assert state is not None
    assert set(state.parameters) == {"x", "y"}
    assert np.isfinite(state.residual)


def test_fixed_n_minimizer_reduces_loss() -> None:
    p0 = {"x": 2.0, "y": 3.0}
    initial_loss = quadratic_residual(p0)
    state = FixedNMinimizer(n_steps=20)(
        quadratic_residual, p0, bounds={}
    ).unwrap_or_err()
    assert state.residual < initial_loss


def test_fixed_n_minimizer_always_succeeds() -> None:
    """FixedNMinimizer must never return FitFailure, even with n_steps=1."""
    p0 = {"x": 5.0}
    result = FixedNMinimizer(n_steps=1)(quadratic_residual_1d, p0, bounds={})
    # Result wraps a success value, not an exception
    assert result.unwrap_or_err() is not None


@pytest.mark.parametrize("method", ["L-BFGS-B", "BFGS", "CG", "TNC", "SLSQP"])
def test_fixed_n_minimizer_methods(method: str) -> None:
    p0 = {"x": 1.0}
    state = FixedNMinimizer(n_steps=10, method=method)(  # type: ignore[arg-type]
        quadratic_residual_1d,
        p0,
        bounds={},
    ).unwrap_or_err()
    assert state is not None
    assert np.isfinite(state.residual)


def test_fixed_n_minimizer_with_tol() -> None:
    p0 = {"x": 2.0}
    state = FixedNMinimizer(n_steps=50, tol=1e-3)(
        quadratic_residual_1d,
        p0,
        bounds={},
    ).unwrap_or_err()
    assert state is not None
    assert np.isfinite(state.residual)


def test_fixed_n_minimizer_accessible_from_fit_module() -> None:
    assert fit.FixedNMinimizer is FixedNMinimizer


def residual_infeasible_below(updates: dict[str, float]) -> float:
    """Minimum at x=1, but the "simulation" fails for x < 0.9."""
    x = updates["x"]
    if x < 0.9:
        return np.inf
    return (x - 1.0) ** 2


def always_inf_residual(updates: dict[str, float]) -> float:  # noqa: ARG001
    return np.inf


@pytest.mark.parametrize(
    "method", [None, "L-BFGS-B", "TNC", "SLSQP", "Nelder-Mead", "Powell"]
)
def test_local_minimizer_single_parameter_with_infeasible_region(
    method: str | None,
) -> None:
    # The first unit-length gradient step from x=1.2 lands in the infeasible region
    state = fit.LocalScipyMinimizer(method=method)(  # type: ignore[arg-type]
        residual_infeasible_below,
        {"x": 1.2},
        bounds={},
    ).unwrap_or_err()
    assert state.parameters["x"] == pytest.approx(1.0, abs=1e-3)


def test_local_minimizer_all_infinite_residuals_fail() -> None:
    res = fit.LocalScipyMinimizer()(always_inf_residual, {"x": 1.0}, bounds={})
    assert isinstance(res.value, Exception)


def test_fixed_n_minimizer_infeasible_region() -> None:
    state = FixedNMinimizer(n_steps=50)(
        residual_infeasible_below,
        {"x": 1.2},
        bounds={},
    ).unwrap_or_err()
    assert state.parameters["x"] == pytest.approx(1.0, abs=1e-3)


def test_fixed_n_minimizer_all_infinite_residuals() -> None:
    state = FixedNMinimizer(n_steps=5)(
        always_inf_residual,
        {"x": 1.0},
        bounds={},
    ).unwrap_or_err()
    assert state.residual == np.inf


def test_global_minimizer_infeasible_region() -> None:
    state = fit.GlobalScipyMinimizer(method="basinhopping")(
        residual_infeasible_below,
        {"x": 1.2},
        bounds={},
    ).unwrap_or_err()
    assert state.parameters["x"] == pytest.approx(1.0, abs=1e-3)
