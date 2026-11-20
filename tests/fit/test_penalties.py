# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns
from mxlpy.distributions import (
    Laplace,
    LogNormal,
    LogUniform,
    Normal,
    SoftBounds,
    Uniform,
)
from mxlpy.fit._gradient import Transform
from mxlpy.fit.penalties import PriorPenalty, collect_penalties
from mxlpy.types import Array

NLL = fit.GaussianNll(1.0)
DATA = pd.Series({"x": 2.0})
MINIMIZERS = {
    "lbfgs": fit.LocalScipyMinimizer(method="L-BFGS-B"),
    "nelder-mead": fit.LocalScipyMinimizer(method="Nelder-Mead"),
    "trust-exact": fit.LocalScipyMinimizer(method="trust-exact"),
    "least-squares": fit.LeastSquaresMinimizer(),
}


def _source_model(p: float = 3.0) -> KineticModelBuilder:
    """Steady state x = p, so priors and data combine analytically."""
    return (
        KineticModelBuilder()
        .add_parameters({"p": p, "k": 1.0})
        .add_variables({"x": 1.0})
        .add_reaction("vin", fns.constant, args=["p"], stoichiometry={"x": 1})
        .add_reaction(
            "vout", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1}
        )
    )


def _decay_model() -> KineticModelBuilder:
    return (
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


def _decay_data() -> pd.DataFrame:
    t = np.linspace(0.2, 6.0, 20)
    res = Simulator(_decay_model()).simulate_time_course(t).get_result().unwrap_or_err()
    return res.get_combined().loc[t, ["x", "y"]]


@pytest.mark.parametrize("transform", [None, "log"])
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_normal_prior_gives_analytic_map(
    minimizer: str, transform: Transform | None
) -> None:
    # (p - 2)² / 2 + p² / 2 is minimal at p = 1, independent of the transform
    res = fit.steady_state(
        _source_model(),
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS[minimizer],
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        transform=transform,
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(1.0, rel=1e-4)
    assert res.loss == pytest.approx(0.5, rel=1e-4)
    assert res.penalty == pytest.approx(0.5, rel=1e-4)
    assert res.objective == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("minimizer", ["lbfgs", "least-squares"])
def test_lognormal_prior_pulls_towards_its_median(minimizer: str) -> None:
    # Penalty on log(p): (p - 2)² / 2 + (log p / s)² / 2, stationary where
    # p - 2 + log(p) / (s² p) = 0
    s = 0.5
    res = fit.steady_state(
        _source_model(),
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS[minimizer],
        loss_fn=NLL,
        prior={"p": LogNormal(0.0, s)},
    ).unwrap_or_err()
    p = res.best_pars["p"]
    assert p - 2 + math.log(p) / (s**2 * p) == pytest.approx(0.0, abs=1e-4)
    assert res.penalty == pytest.approx(0.5 * (math.log(p) / s) ** 2, rel=1e-6)


@pytest.mark.parametrize("minimizer", ["lbfgs", "least-squares"])
def test_soft_bounds(minimizer: str) -> None:
    # Data at 2 with sigma 0.1, soft upper bound 1.5 with scale 0.1: halfway
    res = fit.steady_state(
        _source_model(1.0),
        p0={"p": 1.0},
        data=DATA,
        minimizer=MINIMIZERS[minimizer],
        loss_fn=fit.GaussianNll(0.1),
        prior={"p": SoftBounds(0.1, 1.5, 0.1)},
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(1.75, rel=1e-4)

    # No penalty inside the bounds
    res = fit.steady_state(
        _source_model(1.0),
        p0={"p": 1.0},
        data=DATA,
        minimizer=MINIMIZERS[minimizer],
        loss_fn=NLL,
        prior={"p": SoftBounds(1.0, 3.0, 0.1, log=True)},
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(2.0, rel=1e-4)
    assert res.penalty == pytest.approx(0.0, abs=1e-12)


def test_laplace_prior_is_an_l1_penalty() -> None:
    # (p - 2)² / 2 + |p| / b is minimal at p = 2 - 1 / b
    res = fit.steady_state(
        _source_model(),
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": Laplace(0.0, 2.0)},
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(1.5, rel=1e-4)


def test_unidentifiable_parameter_stays_at_prior() -> None:
    # k2 alone only enters through y; with y missing, its prior decides
    data = _decay_data()
    data["y"] = np.nan
    res = fit.time_course(
        _decay_model(),
        p0={"k1": 1.0, "k2": 1.0},
        data=data,
        minimizer=fit.LeastSquaresMinimizer(),
        loss_fn=fit.GaussianNll(0.01),
        prior={"k2": LogNormal(math.log(3.0), 0.1)},
    ).unwrap_or_err()
    assert res.best_pars["k1"] == pytest.approx(0.5, rel=1e-4)
    assert res.best_pars["k2"] == pytest.approx(3.0, rel=1e-4)


def test_minimizers_agree_with_heuristic_loss() -> None:
    # The penalty is added to the loss as is, whatever the loss
    prior = {"k1": Normal(0.3, 0.1), "k2": LogNormal(0.0, 0.5)}
    results = {
        name: fit.time_course(
            _decay_model(),
            p0={"k1": 1.0, "k2": 1.0},
            data=_decay_data(),
            minimizer=minimizer,
            loss_fn=fit.mean_squared,
            prior=prior,
        ).unwrap_or_err()
        for name, minimizer in [
            ("lbfgs", fit.LocalScipyMinimizer()),
            ("trust-constr", fit.LocalScipyMinimizer(method="trust-constr")),
            ("least-squares", fit.LeastSquaresMinimizer()),
        ]
    }
    for name in ["trust-constr", "least-squares"]:
        for k in prior:
            assert results[name].best_pars[k] == pytest.approx(
                results["lbfgs"].best_pars[k], rel=1e-3
            )
        assert results[name].objective == pytest.approx(
            results["lbfgs"].objective, rel=1e-6
        )


def test_least_squares_needs_sum_of_squares_loss() -> None:
    # rmse isn't a sum of squares, so least squares can't weight the prior
    with pytest.raises(TypeError, match="sum of squares"):
        fit.time_course(
            _decay_model(),
            p0={"k1": 1.0, "k2": 1.0},
            data=_decay_data(),
            minimizer=fit.LeastSquaresMinimizer(),
            loss_fn=fit.rmse,
            prior={"k1": Normal(0.3, 0.1)},
        )


@pytest.mark.parametrize("x", [1e-10, 1e-6, 1.0, 1e6])
def test_prior_jacobian_for_any_magnitude(x: float) -> None:
    prior = {
        "k": LogNormal(math.log(x), 1.0),
        "s": SoftBounds(1e-20, 1e-19, 0.5, log=True),
    }
    jac = PriorPenalty(prior).jacobian(np.array([2 * x, 2 * x]))
    assert jac[0, 0] == pytest.approx(1 / (2 * x), rel=1e-6)
    assert jac[1, 1] == pytest.approx(1 / (2 * x * math.log(10) * 0.5), rel=1e-6)


def test_weighted_prior_penalty() -> None:
    weighted = fit.steady_state(
        _source_model(),
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS["least-squares"],
        loss_fn=NLL,
        penalty=PriorPenalty({"p": Normal(0.0, 1.0)}, weight=3.0),
    ).unwrap_or_err()
    # (p - 2)² / 2 + 3 p² / 2 is minimal at p = 0.5
    assert weighted.best_pars["p"] == pytest.approx(0.5, rel=1e-4)
    assert weighted.penalty == pytest.approx(3 * 0.5 * 0.25, rel=1e-4)


class _Difference:
    """Custom penalty coupling two parameters, without residuals."""

    names = ("k1", "k2")

    def __call__(self, values: Array) -> float:
        return 100.0 * float(values[0] - values[1]) ** 2


class _DifferenceResiduals(_Difference):
    def residuals(self, values: Array) -> Array:
        return np.array([math.sqrt(200.0) * (values[0] - values[1])])


@pytest.mark.parametrize("penalty", [_Difference(), _DifferenceResiduals()])
def test_custom_penalty(penalty: fit.Penalty) -> None:
    res = fit.time_course(
        _decay_model(),
        p0={"k1": 1.0, "k2": 1.5},
        data=_decay_data(),
        minimizer=fit.LocalScipyMinimizer(),
        penalty=penalty,
    ).unwrap_or_err()
    assert abs(res.best_pars["k1"] - res.best_pars["k2"]) < 0.5
    assert res.penalty == pytest.approx(
        penalty(np.array([res.best_pars["k1"], res.best_pars["k2"]]))
    )


def test_least_squares_needs_penalty_residuals() -> None:
    with pytest.raises(TypeError, match="residuals"):
        fit.time_course(
            _decay_model(),
            p0={"k1": 1.0, "k2": 1.5},
            data=_decay_data(),
            minimizer=fit.LeastSquaresMinimizer(),
            loss_fn=fit.mean_squared,
            penalty=_Difference(),
        )


def test_prior_needs_fitted_parameter() -> None:
    with pytest.raises(ValueError, match="aren't in p0"):
        collect_penalties({"k": Normal(0, 1)}, None, ["p"])
    with pytest.raises(ValueError, match="aren't in p0"):
        fit.steady_state(
            _source_model(),
            p0={"p": 3.0},
            data=DATA,
            minimizer=MINIMIZERS["lbfgs"],
            prior={"k": Normal(0, 1)},
        )


def test_no_prior_keeps_the_loss() -> None:
    res = fit.steady_state(
        _source_model(),
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
    ).unwrap_or_err()
    assert res.penalty == 0.0
    assert res.objective == res.loss


def test_bounded_prior_outside_support() -> None:
    # The penalty is infinite outside, so the minimizer stays inside
    res = fit.steady_state(
        _source_model(1.0),
        p0={"p": 1.0},
        data=DATA,
        minimizer=MINIMIZERS["nelder-mead"],
        loss_fn=NLL,
        prior={"p": Uniform(0.5, 1.5)},
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(1.5, abs=1e-3)
    res = fit.steady_state(
        _source_model(1.0),
        p0={"p": 1.0},
        data=DATA,
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": LogUniform(0.1, 10.0)},
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(2.0, rel=1e-4)
    assert res.penalty == pytest.approx(0.0, abs=1e-12)


def test_group_picks_lowest_objective() -> None:
    res = fit.group_steady_state(
        _source_model(),
        p0=pd.DataFrame({"p": [0.5, 3.0]}),
        data=DATA,
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        parallel=False,
    )
    assert len(res.fits) == 2
    assert res.get_best_fit().best_pars["p"] == pytest.approx(1.0, rel=1e-4)
    np.testing.assert_allclose(res.get_objectives(), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(res.get_losses(), [0.5, 0.5], rtol=1e-4)


def test_ensemble() -> None:
    res = fit.ensemble_steady_state(
        [_source_model(), _source_model()],
        p0={"p": 3.0},
        data=DATA,
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        parallel=False,
    )
    assert res.get_best_fit().best_pars["p"] == pytest.approx(1.0, rel=1e-4)


def test_joint_and_mixed() -> None:
    # Two data sets and the prior: (2 + 2 + 0) / 3
    settings = [fit.FitSettings(_source_model(), DATA) for _ in range(2)]
    res = fit.joint_steady_state(
        settings,
        p0={"p": 3.0},
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        parallel=False,
    ).unwrap_or_err()
    assert res.best_pars["p"] == pytest.approx(4 / 3, rel=1e-4)
    assert res.penalty == pytest.approx(0.5 * (4 / 3) ** 2, rel=1e-4)

    mixed = fit.joint_mixed(
        [
            fit.MixedSettings(
                _source_model(), DATA, residual_fn=fit.steady_state_residual
            )
            for _ in range(2)
        ],
        p0={"p": 3.0},
        minimizer=MINIMIZERS["lbfgs"],
        loss_fn=NLL,
        prior={"p": Normal(0.0, 1.0)},
        parallel=False,
    ).unwrap_or_err()
    assert mixed.best_pars["p"] == pytest.approx(4 / 3, rel=1e-4)


@pytest.mark.parametrize(
    "loss_fn",
    [fit.GaussianNll(pd.Series({"x": 0.01, "y": 0.05})), fit.rmse, fit.mae],
)
def test_penalised_gradient_matches_finite_differences(loss_fn: fit.LossFn) -> None:
    from mxlpy.fit._gradient import GradientResidual  # noqa: PLC0415
    from mxlpy.fit.abstract import _Settings  # noqa: PLC0415
    from mxlpy.fit.penalties import penalised  # noqa: PLC0415
    from mxlpy.fit.residuals import time_course_residual  # noqa: PLC0415
    from mxlpy.runners import KineticRunner  # noqa: PLC0415

    settings = _Settings(
        model=_decay_model(),
        data=_decay_data() * 1.05,
        y0=None,
        runner=KineticRunner(derivatives="forward"),
        loss_fn=loss_fn,
        p_names=["k1", "k2"],
        v_names=[],
        standard_scale=True,
    )
    residual = penalised(
        GradientResidual(time_course_residual, settings),
        [
            PriorPenalty({"k1": Normal(0.3, 0.1), "k2": SoftBounds(0.1, 1.0, 0.5)}),
            _Difference(),
        ],
    )
    at = {"k1": 0.6, "k2": 1.7}
    numeric = []
    for k, v in at.items():
        h = 1e-5 * v
        numeric.append(
            (residual(at | {k: v + h}) - residual(at | {k: v - h})) / (2 * h)
        )
    np.testing.assert_allclose(residual.gradient(at), numeric, rtol=1e-4)  # type: ignore[attr-defined]
    # A penalty without residuals has no Gauss-Newton Hessian
    assert residual.hessian(at) is None  # type: ignore[attr-defined]
