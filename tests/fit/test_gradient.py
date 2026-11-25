# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

from mxlpy import (
    KineticModelBuilder,
    Simulator,
    SteadyStateModelBuilder,
    fit,
    fns,
    make_protocol,
)
from mxlpy.fit import LeastSquaresMinimizer, LocalScipyMinimizer
from mxlpy.fit._gradient import GradientResidual, _TransformedGradient, transformed
from mxlpy.fit.abstract import _Settings
from mxlpy.fit.residuals import steady_state_residual, time_course_residual
from mxlpy.forward_sensitivities import ForwardSensitivities
from mxlpy.minimizers import FixedNMinimizer, GlobalScipyMinimizer
from mxlpy.minimizers.abstract import MinimizerProtocol
from mxlpy.runners import KineticRunner, default_runner
from mxlpy.types import ArrayLike, Result

TRUE = {"k1": 0.5, "k2": 2.0}
P0 = {"k1": 1.5, "k2": 0.7}
T = np.linspace(0.2, 6.0, 30)


def _model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters(TRUE)
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            args=["x", "k1"],
            stoichiometry={"x": -1, "y": 1},
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )


def _data() -> pd.DataFrame:
    res = Simulator(_model()).simulate_time_course(T).get_result().unwrap_or_err()
    return res.get_combined().loc[T, ["x", "y", "v2"]]


def _settings(
    loss_fn: fit.LossFn = fit.rmse,
    data: pd.DataFrame | None = None,
    derivatives: Literal["auto", "forward", "finite_difference"] = "forward",
) -> _Settings:
    model = _model()
    return _Settings(
        model=model,
        data=_data() if data is None else data,
        y0=None,
        runner=KineticRunner(derivatives=derivatives),
        loss_fn=loss_fn,
        p_names=list(P0),
        v_names=[],
        standard_scale=True,
    )


def _numeric_gradient(residual: GradientResidual, at: dict[str, float]) -> np.ndarray:
    out = []
    for k, v in at.items():
        h = 1e-5 * v
        out.append((residual(at | {k: v + h}) - residual(at | {k: v - h})) / (2 * h))
    return np.array(out)


def _custom_loss(y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
    return float(np.mean(np.abs(y_pred - y_true) ** 1.5))


@pytest.mark.parametrize("loss_fn", [fit.rmse, fit.mean_squared, fit.mae, _custom_loss])
def test_gradient_matches_numeric_loss_gradient(loss_fn: fit.LossFn) -> None:
    residual = GradientResidual(time_course_residual, _settings(loss_fn))
    np.testing.assert_allclose(
        residual.gradient(P0), _numeric_gradient(residual, P0), rtol=1e-4
    )


def test_forward_and_finite_difference_jacobians_agree() -> None:
    fw = GradientResidual(time_course_residual, _settings())
    fd = GradientResidual(
        time_course_residual, _settings(derivatives="finite_difference")
    )
    assert fw.forward
    assert not fd.forward
    np.testing.assert_allclose(fw.jacobian(P0), fd.jacobian(P0), atol=1e-4)
    np.testing.assert_allclose(fw.residuals(P0), fd.residuals(P0))
    # Residual vector: scaled prediction minus data, flattened without gaps
    assert fw.residuals(P0).shape == (len(T) * 3,)
    assert fw(P0) == pytest.approx(np.sqrt(np.mean(fw.residuals(P0) ** 2)))


def test_missing_data_is_left_out() -> None:
    data = _data()
    data.iloc[3, 0] = np.nan
    residual = GradientResidual(time_course_residual, _settings(data=data))
    assert residual.residuals(P0).shape == (len(T) * 3 - 1,)
    assert residual.jacobian(P0).shape == (len(T) * 3 - 1, 2)
    np.testing.assert_allclose(
        residual.gradient(P0), _numeric_gradient(residual, P0), rtol=1e-4
    )


def test_constant_data_columns_are_left_out() -> None:
    # Zero std: the scalar loss skips the column (NaN after scaling), so must
    # the residual vector, else least squares only sees the penalty
    data = _data()
    data["y"] = 0.5
    residual = GradientResidual(time_course_residual, _settings(data=data))
    assert residual.residuals(P0).shape == (len(T) * 2,)
    assert np.isfinite(residual.jacobian(P0)).all()
    np.testing.assert_allclose(
        residual.gradient(P0), _numeric_gradient(residual, P0), rtol=1e-4
    )


@pytest.mark.parametrize(
    ("minimizer", "derivatives"),
    [
        (LeastSquaresMinimizer(), "forward"),
        (LeastSquaresMinimizer(), "finite_difference"),
        (LeastSquaresMinimizer(method="lm"), "auto"),
        (LocalScipyMinimizer(), "auto"),
        (FixedNMinimizer(n_steps=200), "auto"),
    ],
)
def test_time_course_fits_recover_parameters(
    minimizer: MinimizerProtocol,
    derivatives: Literal["auto", "forward", "finite_difference"],
) -> None:
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=minimizer,
        runner=KineticRunner(derivatives=derivatives),
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


def test_log_transform_least_squares() -> None:
    res = fit.time_course(
        _model(),
        p0={"k1": 20.0, "k2": 0.01},
        data=_data(),
        minimizer=LeastSquaresMinimizer(),
        transform="log",
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-4)


def test_transformed_maps_values_and_bounds() -> None:
    residual = GradientResidual(time_course_residual, _settings())
    wrapped, p0, bounds, back = transformed(
        residual, P0, {"k1": (1e-2, 1e2)}, {"k1": "log", "k2": "linear"}
    )
    assert p0 == pytest.approx({"k1": np.log10(1.5), "k2": 0.7})
    assert bounds == {"k1": pytest.approx((-2.0, 2.0))}
    assert back(p0) == pytest.approx(P0)
    # chain rule: d/du = d/dk * k ln 10
    expected = residual.gradient(P0) * np.array([1.5 * np.log(10), 1.0])
    np.testing.assert_allclose(wrapped.gradient(p0), expected)  # type: ignore[attr-defined]
    # unbounded log parameters get explicit default bounds
    _, _, bounds, _ = transformed(residual, P0, {}, "log")
    assert bounds == {"k1": (-6.0, 6.0), "k2": (-6.0, 6.0)}


def test_transform_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        transformed(_quadratic, {"a": -1.0}, {}, "log")
    with pytest.raises(ValueError, match="not in p0"):
        transformed(_quadratic, {"a": 1.0}, {}, {"b": "log"})


def test_least_squares_needs_default_residual_fn() -> None:
    def custom(updates: dict[str, float], settings: _Settings) -> float:
        return time_course_residual(updates, settings)

    with pytest.raises(TypeError, match="residual functions"):
        fit.time_course(
            _model(),
            p0=P0,
            data=_data(),
            minimizer=LeastSquaresMinimizer(),
            residual_fn=custom,
        )
    with pytest.raises(ValueError, match="residual functions"):
        GradientResidual(custom, _settings())  # type: ignore[arg-type]


def test_custom_residual_fn_keeps_gradient_free_minimizers_working() -> None:
    def custom(updates: dict[str, float], settings: _Settings) -> float:
        return time_course_residual(updates, settings)

    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=LocalScipyMinimizer(),
        residual_fn=custom,
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


class _NoSensitivitiesRunner:
    """Custom runner without sensitivity methods, delegating simulations."""

    def __init__(self) -> None:
        self.inner = KineticRunner()

    def steady_state(self, model: KineticModelBuilder) -> object:
        return self.inner.steady_state(model)

    def time_course(self, model: KineticModelBuilder, time_points: object) -> object:
        return self.inner.time_course(model, time_points)  # type: ignore[arg-type]


def test_runner_without_sensitivities_uses_finite_differences() -> None:
    settings = _settings()
    settings.runner = _NoSensitivitiesRunner()  # type: ignore[assignment]
    residual = GradientResidual(time_course_residual, settings)
    assert not residual.forward
    np.testing.assert_allclose(
        residual.jacobian(P0),
        GradientResidual(time_course_residual, _settings()).jacobian(P0),
        atol=1e-4,
    )
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        minimizer=LeastSquaresMinimizer(),
        runner=_NoSensitivitiesRunner(),  # type: ignore[arg-type]
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


def test_derivatives_come_from_the_runner() -> None:
    calls: list[list[str]] = []

    class Counting(KineticRunner):
        def time_course_sensitivities(
            self, model: KineticModelBuilder, time_points: ArrayLike, wrt: list[str]
        ) -> Result[ForwardSensitivities]:
            calls.append(wrt)
            return super().time_course_sensitivities(model, time_points, wrt)

    settings = _settings()
    settings.runner = Counting()
    GradientResidual(time_course_residual, settings).jacobian(P0)
    assert calls == [list(P0)]


def test_postprocess_falls_back_to_finite_differences() -> None:
    def double(df: pd.DataFrame) -> pd.DataFrame:
        return df * 2

    data = _data() * 2
    res = fit.time_course(
        _model(),
        p0=P0,
        data=data,
        minimizer=LeastSquaresMinimizer(),
        postprocess_fn=double,
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


def test_steady_state_fits() -> None:
    model = (
        KineticModelBuilder()
        .add_parameters({"kf": 1.0, "kr": 0.5})
        .add_variables({"A": 1.0, "B": 0.0})
        .add_reaction(
            "vf", fns.mass_action_1s, args=["A", "kf"], stoichiometry={"A": -1, "B": 1}
        )
        .add_reaction(
            "vr", fns.mass_action_1s, args=["B", "kr"], stoichiometry={"A": 1, "B": -1}
        )
    )
    # A* = kr / (kf + kr), vf* = kf A*
    data = pd.Series({"A": 1 / 3, "vf": 2 / 3})
    res = fit.steady_state(
        model,
        p0={"kf": 1.0, "kr": 2.0},
        data=data,
        minimizer=LeastSquaresMinimizer(),
        standard_scale=False,
    ).unwrap_or_err()
    assert res.best_pars["kf"] == pytest.approx(2.0, rel=1e-4)
    assert res.best_pars["kr"] == pytest.approx(1.0, rel=1e-4)

    settings = _Settings(
        model=model,
        data=data,
        y0=None,
        runner=default_runner(model),
        loss_fn=fit.rmse,
        p_names=["kf", "kr"],
        v_names=[],
        standard_scale=False,
    )
    fw = GradientResidual(steady_state_residual, settings)
    settings.runner = KineticRunner(derivatives="finite_difference")
    fd = GradientResidual(steady_state_residual, settings)
    at = {"kf": 1.0, "kr": 2.0}
    np.testing.assert_allclose(fw.jacobian(at), fd.jacobian(at), atol=1e-5)


def test_steady_state_model_builder_uses_finite_differences() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("k", 1.0)
        .add_derived("d", fn=fns.twice, args=["k"])
    )
    res = fit.steady_state(
        model,
        p0={"k": 3.0},
        data=pd.Series({"d": 5.0}),
        minimizer=LeastSquaresMinimizer(),
    ).unwrap_or_err()
    assert res.best_pars["k"] == pytest.approx(2.5, rel=1e-6)


def test_protocol_time_course_least_squares() -> None:
    protocol = make_protocol([(2.0, {"k1": 0.5}), (4.0, {"k1": 1.0})])
    model = _model()
    data = (
        Simulator(model)
        .simulate_protocol_time_course(protocol, T)
        .get_result()
        .unwrap_or_err()
        .variables.reindex(T, method="nearest", tolerance=1e-9)
        .loc[:, ["x", "y"]]
    )
    res = fit.protocol_time_course(
        model,
        p0={"k2": 0.7},
        data=data,
        protocol=protocol,
        minimizer=LeastSquaresMinimizer(),
    ).unwrap_or_err()
    assert res.best_pars["k2"] == pytest.approx(2.0, rel=1e-3)


def test_group_fit_with_least_squares() -> None:
    res = fit.group_time_course(
        _model(),
        p0=pd.DataFrame([P0, {"k1": 0.2, "k2": 5.0}]),
        data=_data(),
        minimizer=LeastSquaresMinimizer(),
        parallel=False,
    )
    assert len(res.fits) == 2
    for f in res.fits.values():
        assert f.best_pars["k2"] == pytest.approx(2.0, rel=1e-3)


def _quadratic(updates: dict[str, float]) -> float:
    return (updates["x"] - 1.0) ** 2 + (updates["y"] + 2.0) ** 2


@pytest.mark.parametrize("method", ["differential_evolution", "dual_annealing"])
def test_global_minimizer_accepts_bounds_dict(method: str) -> None:
    state = GlobalScipyMinimizer(method=method)(  # type: ignore[arg-type]
        _quadratic, {"x": 0.0, "y": 0.0}, {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}
    ).unwrap_or_err()
    assert state.parameters["x"] == pytest.approx(1.0, abs=1e-3)
    assert state.parameters["y"] == pytest.approx(-2.0, abs=1e-3)


def _interp_rate(x: float, k: float) -> float:
    # np.interp can't be parsed symbolically
    return k * float(np.interp(x, [0.0, 10.0], [0.0, 10.0]))


def test_unparsable_model_falls_back_to_finite_differences() -> None:
    model = _model().update_reaction("v1", fn=_interp_rate)
    res = fit.time_course(
        model,
        p0=P0,
        data=_data(),
        minimizer=LeastSquaresMinimizer(),
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


def test_non_finite_predictions_are_skipped() -> None:
    def drop_one(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.iloc[10, 0] = np.nan
        return df

    for minimizer in (LeastSquaresMinimizer(), LocalScipyMinimizer()):
        res = fit.time_course(
            _model(),
            p0=P0,
            data=_data(),
            minimizer=minimizer,
            postprocess_fn=drop_one,
            runner=KineticRunner(derivatives="finite_difference"),
        ).unwrap_or_err()
        for k, v in TRUE.items():
            assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


def test_replicate_time_points() -> None:
    data = _data()
    data = pd.concat([data, data * 1.01]).sort_index()
    for minimizer in (LeastSquaresMinimizer(), LocalScipyMinimizer()):
        res = fit.time_course(
            _model(),
            p0=P0,
            data=data,
            minimizer=minimizer,
        ).unwrap_or_err()
        assert res.best_pars["k2"] == pytest.approx(2.0, rel=2e-2)
    fw = GradientResidual(time_course_residual, _settings(data=data))
    fd = GradientResidual(
        time_course_residual, _settings(data=data, derivatives="finite_difference")
    )
    np.testing.assert_allclose(fw.jacobian(P0), fd.jacobian(P0), atol=1e-4)


def test_loss_matches_residual_function() -> None:
    settings = _settings()
    residual = GradientResidual(time_course_residual, settings)
    assert residual(P0) == pytest.approx(time_course_residual(P0, settings))
    residual.gradient(P0)
    assert residual(P0) == pytest.approx(time_course_residual(P0, settings))


def test_invalid_transform_values_rejected() -> None:
    with pytest.raises(ValueError, match="'log' or 'linear'"):
        transformed(_quadratic, {"a": 1.0}, {}, {"a": "Log"})  # type: ignore[dict-item]


def test_log_bounds_stay_finite_for_non_positive_lower_bounds() -> None:
    _, _, bounds, _ = transformed(_quadratic, {"a": 1.0}, {"a": (0.0, 10.0)}, "log")
    assert bounds == {"a": pytest.approx((-6.0, 1.0))}


def test_least_squares_clips_infeasible_start() -> None:
    res = fit.time_course(
        _model(),
        p0={"k1": 0.0, "k2": 0.7},
        data=_data(),
        minimizer=LeastSquaresMinimizer(),
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-3)


###############################################################################
# Gauss-Newton Hessians
###############################################################################


def test_gauss_newton_hessian_matches_numeric_close_to_optimum() -> None:
    residual = GradientResidual(time_course_residual, _settings(fit.mean_squared))
    at = {k: v * 1.0001 for k, v in TRUE.items()}
    numeric = []
    for k, v in at.items():
        h = 1e-5 * v
        numeric.append(
            (residual.gradient(at | {k: v + h}) - residual.gradient(at | {k: v - h}))
            / (2 * h)
        )
    hessian = residual.hessian(at)
    assert hessian is not None
    np.testing.assert_allclose(hessian, np.array(numeric), rtol=1e-3)
    assert np.all(np.linalg.eigvalsh(cast(np.ndarray, residual.hessian(P0))) >= 0)
    # No Gauss-Newton Hessian for other losses
    mae = GradientResidual(time_course_residual, _settings(fit.mae))
    assert mae.hessian(P0) is None


def test_transformed_hessian_is_gauss_newton_in_log_space() -> None:
    residual = GradientResidual(time_course_residual, _settings())
    wrapped, u, _, _ = transformed(residual, P0, {}, {"k1": "log"})
    assert isinstance(wrapped, _TransformedGradient)
    f = np.array([P0["k1"] * np.log(10), 1.0])
    expected = cast(np.ndarray, residual.hessian(P0)) * np.outer(f, f)
    hessian = cast(np.ndarray, wrapped.hessian(u))
    np.testing.assert_allclose(hessian, expected)
    assert np.all(np.linalg.eigvalsh(hessian) >= 0)


@pytest.mark.parametrize("method", ["trust-constr", "trust-exact"])
@pytest.mark.parametrize("loss_fn", [fit.rmse, fit.mean_squared])
def test_gauss_newton_fits(method: str, loss_fn: fit.LossFn) -> None:
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        loss_fn=loss_fn,
        minimizer=LocalScipyMinimizer(method=method),  # type: ignore[arg-type]
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-6)


def test_gauss_newton_other_losses_use_bfgs_updates() -> None:
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        loss_fn=fit.mae,
        minimizer=LocalScipyMinimizer(method="trust-constr"),
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-4)
    with pytest.raises(ValueError, match="trust-constr"):
        fit.time_course(
            _model(),
            p0=P0,
            data=_data(),
            loss_fn=fit.mae,
            minimizer=LocalScipyMinimizer(method="trust-exact"),
        )


def test_gauss_newton_respects_bounds() -> None:
    res = fit.time_course(
        _model(),
        p0=P0,
        data=_data(),
        bounds={"k2": (0.1, 1.5)},
        minimizer=LocalScipyMinimizer(method="trust-constr"),
    ).unwrap_or_err()
    assert 0.1 <= res.best_pars["k2"] <= 1.5
    assert res.best_pars["k2"] == pytest.approx(1.5, rel=1e-4)


def test_gauss_newton_with_log_transform() -> None:
    res = fit.time_course(
        _model(),
        p0={"k1": 5.0, "k2": 0.2},
        data=_data(),
        minimizer=LocalScipyMinimizer(method="trust-constr"),
        transform="log",
    ).unwrap_or_err()
    for k, v in TRUE.items():
        assert res.best_pars[k] == pytest.approx(v, rel=1e-5)


def test_native_backend_only_implements_lm() -> None:
    with pytest.raises(ValueError, match="lm"):
        LeastSquaresMinimizer(backend="native")
