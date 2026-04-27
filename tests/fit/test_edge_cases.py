"""Edge case tests for the fit module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mxlpy import Model, fit, fns
from mxlpy.fit import losses
from mxlpy.fit.abstract import _Settings
from mxlpy.fit.residuals import steady_state_residual
from mxlpy.minimizers.abstract import mock_minimizer


def simple_model() -> Model:
    return (
        Model()
        .add_variables({"S": 10.0, "P": 0.0})
        .add_parameters({"k1": 1.0, "k2": 2.0})
        .add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "k1"],
            stoichiometry={"S": -1.0, "P": 1.0},
        )
        .add_reaction(
            "v2",
            fn=fns.mass_action_1s,
            args=["P", "k2"],
            stoichiometry={"P": -1.0},
        )
    )


def test_fit_steady_state_all_nan_data() -> None:
    """All-NaN data must not crash; residual becomes nan/inf and fit returns.

    The steady_state_residual tries to compute loss between model output and
    data.  With all-NaN data the loss function receives NaN columns and should
    return NaN or inf, causing the minimizer to converge nowhere useful but
    not raise an unhandled exception.
    """
    data = pd.Series({"S": float("nan"), "P": float("nan")})

    result = fit.steady_state(
        simple_model(),
        p0={"k1": 1.0, "k2": 2.0},
        data=data,
        minimizer=mock_minimizer,
    )
    # mock_minimizer always returns p0 regardless of residual, so result succeeds
    assert result.unwrap_or_err() is not None


def test_fit_steady_state_empty_data() -> None:
    """Empty pd.Series as data: fit should not crash."""
    data = pd.Series(dtype=float)

    result = fit.steady_state(
        simple_model(),
        p0={"k1": 1.0, "k2": 2.0},
        data=data,
        minimizer=mock_minimizer,
    )
    assert result.unwrap_or_err() is not None


def test_fit_time_course_all_nan_data() -> None:
    """All-NaN DataFrame does not crash the time-course fit."""
    data = pd.DataFrame(
        {"S": [float("nan"), float("nan")], "P": [float("nan"), float("nan")]},
        index=[0.0, 1.0],
    )

    result = fit.time_course(
        simple_model(),
        p0={"k1": 1.0, "k2": 2.0},
        data=data,
        minimizer=mock_minimizer,
    )
    assert result.unwrap_or_err() is not None


def test_steady_state_residual_nan_data_returns_inf_or_nan() -> None:
    """steady_state_residual with NaN data returns a non-finite residual.

    This tests the residual function in isolation, which should handle NaN
    gracefully by returning inf (simulation failure sentinel) or propagating NaN.
    """
    model = simple_model()
    data = pd.Series({"S": float("nan"), "P": float("nan")})

    settings = _Settings(
        model=model,
        data=data,
        y0=None,
        loss_fn=losses.rmse,
        p_names=["k1", "k2"],
        v_names=[],
        integrator=None,
        oscillation_detector=None,
        standard_scale=False,
    )
    residual = steady_state_residual({"k1": 1.0, "k2": 2.0}, settings)
    # NaN data → nan loss; inf is also acceptable (integration-failure sentinel)
    assert not np.isfinite(residual) or np.isnan(residual)


def test_fit_with_bounds_p0_inside_bounds_succeeds() -> None:
    """p0 inside bounds is the normal case; should succeed."""
    data = pd.Series({"S": 5.0, "P": 2.0})
    bounds: dict[str, tuple[float | None, float | None]] = {
        "k1": (0.1, 10.0),
        "k2": (0.1, 10.0),
    }

    result = fit.steady_state(
        simple_model(),
        p0={"k1": 1.0, "k2": 2.0},
        data=data,
        minimizer=mock_minimizer,
        bounds=bounds,
    )
    assert result.unwrap_or_err() is not None
