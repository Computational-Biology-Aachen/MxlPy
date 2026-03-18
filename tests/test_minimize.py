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
