# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Derivatives of joint and mixed fits: sums of their parts' derivatives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns
from mxlpy.fit._joint import _JointGradient, _sum_of_residuals, _with_derivatives
from mxlpy.fit.abstract import _joint_settings
from mxlpy.fit.residuals import steady_state_residual, time_course_residual
from mxlpy.runners import KineticRunner

TIMES = np.linspace(0.5, 5.0, 10)


def _chain(*, k3: bool) -> KineticModelBuilder:
    model = (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )
    if k3:
        # Only in the second model
        model.add_parameter("k3", 0.3).add_reaction(
            "v3", fns.mass_action_1s, args=["x", "k3"], stoichiometry={"x": -1}
        )
    return model


def _data(model: KineticModelBuilder) -> pd.DataFrame:
    res = Simulator(model).simulate_time_course(TIMES).get_result().unwrap_or_err()
    return res.get_variables().loc[:, ["x", "y"]]


def _settings(derivatives: str = "auto", loss_fn: Any = fit.rmse) -> list[Any]:
    return [
        fit.FitSettings(
            model=m,
            data=_data(m),
            runner=KineticRunner(derivatives=derivatives),  # type: ignore[arg-type]
            loss_fn=loss_fn,
        )
        for m in (_chain(k3=False), _chain(k3=True))
    ]


def _joint(settings: list[Any], p0: dict[str, float]) -> _JointGradient:
    full = [
        _joint_settings(
            s,
            p0=p0,
            y0=None,
            runner=None,
            loss_fn=s.loss_fn,
            postprocess_fn=None,
            as_deepcopy=True,
            standard_scale=True,
            needs_dynamics=True,
        )
        for s in settings
    ]

    def loss(updates: dict[str, float]) -> float:
        return _sum_of_residuals(
            updates, residual_fn=time_course_residual, fits=full, pool=None
        )

    residual = _with_derivatives(loss, [(time_course_residual, f) for f in full], p0)
    assert isinstance(residual, _JointGradient)
    return residual


def _fd_gradient(residual: _JointGradient, p: dict[str, float]) -> np.ndarray:
    out = []
    for k, v in p.items():
        h = 1e-6 * max(1.0, abs(v))
        out.append((residual(p | {k: v + h}) - residual(p | {k: v - h})) / (2 * h))
    return np.array(out)


@pytest.mark.parametrize("derivatives", ["forward", "adjoint"])
@pytest.mark.parametrize(
    "loss_fn", [fit.rmse, fit.GaussianNll(0.1)], ids=["rmse", "gaussian_nll"]
)
def test_joint_gradient_matches_finite_differences(
    derivatives: str, loss_fn: Any
) -> None:
    p = {"k1": 0.6, "k2": 1.7, "k3": 0.4}
    residual = _joint(_settings(derivatives, loss_fn), p)
    np.testing.assert_allclose(
        residual.gradient(p), _fd_gradient(residual, p), rtol=1e-4, atol=1e-7
    )


def test_joint_residual_vector() -> None:
    p = {"k1": 0.6, "k2": 1.7, "k3": 0.4}
    residual = _joint(_settings(loss_fn=fit.GaussianNll(0.1)), p)
    r = residual.residuals(p)
    # Its sum of squares / 2 is the loss
    assert residual.squares_factor() == 1.0
    assert float(r @ r) / 2 == pytest.approx(residual(p), rel=1e-8)
    jac = residual.jacobian(p)
    np.testing.assert_allclose(r @ jac, residual.gradient(p), rtol=1e-6)
    assert jac.shape == (len(r), 3)
    # k3 only affects the second model's block
    assert np.all(jac[: len(r) // 2, 2] == 0)
    hess = residual.hessian(p)
    assert hess is not None
    np.testing.assert_allclose(hess, jac.T @ jac, rtol=1e-8)

    rmse = _joint(_settings(loss_fn=fit.rmse), p)
    assert rmse.squares_factor() is None
    with pytest.raises(TypeError, match="sum-of-squares"):
        rmse.residuals(p)


def test_joint_fits_use_the_derivatives() -> None:
    p0 = {"k1": 0.3, "k2": 3.0, "k3": 0.6}
    for minimizer, loss_fn in [
        (fit.LocalScipyMinimizer(method="L-BFGS-B"), fit.rmse),
        (fit.LeastSquaresMinimizer(), fit.GaussianNll(0.1)),
    ]:
        res = fit.joint_time_course(
            _settings("adjoint", loss_fn),
            p0=p0,
            minimizer=minimizer,
            loss_fn=loss_fn,
            parallel=False,
        ).unwrap_or_err()
        assert res.best_pars["k1"] == pytest.approx(0.5, rel=1e-3)
        assert res.best_pars["k2"] == pytest.approx(2.0, rel=1e-3)
        assert res.best_pars["k3"] == pytest.approx(0.3, rel=1e-3)


def test_mixed_fit_gradient() -> None:
    reversible = (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "kr": 0.25})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "k1"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
    )
    to_fit = [
        fit.MixedSettings(
            model=_chain(k3=False),
            data=_data(_chain(k3=False)),
            residual_fn=time_course_residual,
        ),
        fit.MixedSettings(
            model=reversible,
            # A* = kr / (k1 + kr)
            data=pd.Series({"A": 1 / 3, "B": 2 / 3}),
            residual_fn=steady_state_residual,
        ),
    ]
    res = fit.joint_mixed(
        to_fit,
        p0={"k1": 0.3, "k2": 3.0, "kr": 0.4},
        minimizer=fit.LocalScipyMinimizer(method="L-BFGS-B"),
        parallel=False,
    ).unwrap_or_err()
    assert res.best_pars["k1"] == pytest.approx(0.5, rel=1e-3)
    assert res.best_pars["kr"] == pytest.approx(0.25, rel=1e-3)


def test_native_least_squares_falls_back_for_joint_fits() -> None:
    pytest.importorskip("mxlpy_native")
    res = fit.joint_time_course(
        _settings("adjoint", fit.GaussianNll(0.1)),
        p0={"k1": 0.3, "k2": 3.0, "k3": 0.6},
        minimizer=fit.LeastSquaresMinimizer(method="lm", backend="native"),
        loss_fn=fit.GaussianNll(0.1),
        parallel=False,
    ).unwrap_or_err()
    assert res.best_pars["k1"] == pytest.approx(0.5, rel=1e-3)
