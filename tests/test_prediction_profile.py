# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mxlpy import Simulator, fit, identify, make_protocol, plot
from mxlpy.fit._problem import Constraint, setup_problem
from mxlpy.identify._profile2d import _Axis, _repair
from tests.test_profile_likelihood import (
    SIGMA,
    _decay,
    _decay_data,
    _decay_fit,
    _source,
)

mpl.use("Agg")

NLL = fit.GaussianNll(1.0)
THRESHOLD = float(stats.chi2.ppf(0.95, 1) / 2)
THRESHOLD_2D = float(stats.chi2.ppf(0.95, 2) / 2)
HALF = math.sqrt(2 * THRESHOLD)
STEADY = identify.Prediction(target="x")


def _source_profile(**kwargs: object) -> identify.PredictionProfile:
    """Data x = 2 at steady state x = p (k = 1), so Δ(z) = (z - 2)² / 2."""
    return identify.prediction_profile(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5},
        loss_fn=NLL,
        parallel=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_analytic_prediction_interval() -> None:
    pp = _source_profile(prediction=STEADY)
    assert pp.value == pytest.approx(2.0, rel=1e-6)
    assert (pp.lower_status, pp.upper_status) == ("threshold", "threshold")
    assert pp.ci[0] == pytest.approx(2 - HALF, rel=2e-3)
    assert pp.ci[1] == pytest.approx(2 + HALF, rel=2e-3)
    assert pp.alpha == 0.95
    assert pp.delta.index.is_monotonic_increasing
    # Every point is an exact point of the profile, whatever the pseudo data
    x = pp.delta.index.to_numpy()
    np.testing.assert_allclose(pp.delta, (x - 2) ** 2 / 2, atol=1e-8)
    assert not np.allclose(pp.z.to_numpy(), x)
    assert pp.parameters["p"].to_numpy() == pytest.approx(x, rel=1e-6)


def test_prediction_in_another_condition() -> None:
    # x = p / k with k = 2 in the predicted condition: half the interval
    pp = _source_profile(prediction=identify.Prediction("x", parameters={"k": 2.0}))
    assert pp.value == pytest.approx(1.0, rel=1e-6)
    assert pp.ci[0] == pytest.approx((2 - HALF) / 2, rel=2e-3)
    assert pp.ci[1] == pytest.approx((2 + HALF) / 2, rel=2e-3)


def test_merged_and_separate_prediction_agree() -> None:
    # The same condition, once merged into the data, once simulated apart
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 1.5})
    kwargs = {"loss_fn": fit.GaussianNll(SIGMA), "parallel": False}
    merged = identify.prediction_profile(
        _decay(), data, res, prediction=identify.Prediction("y", time=8.0), **kwargs
    )
    # Setting the (unchanged) initial value makes it a separate condition
    apart = identify.prediction_profile(
        _decay(),
        data,
        res,
        prediction=identify.Prediction("y", time=8.0, y0={"x": 2.0}),
        **kwargs,
    )
    np.testing.assert_allclose(merged.ci, apart.ci, rtol=1e-4)


def test_merged_data() -> None:
    data = _decay_data()
    problem = setup_problem(
        _decay(),
        data,
        {"k1": 0.5, "k2": 2.0},
        loss_fn=fit.GaussianNll(SIGMA),
        minimizer=None,
        y0=None,
        protocol=None,
        runner=None,
        bounds=None,
        prior=None,
        penalty=None,
        transform=None,
        postprocess_fn=None,
        standard_scale=True,
    )

    def merged(prediction: identify.Prediction) -> object:
        constraint = Constraint(
            prediction=prediction,
            model=problem.condition(prediction),
            z=0.1,
            sigma=1e-3,
        )
        return problem._merged(constraint)

    new_time = merged(identify.Prediction("y", time=8.0))
    assert new_time is not None
    values, loss = new_time  # type: ignore[misc]
    assert values.loc[8.0, "y"] == 0.1
    assert np.isnan(values.loc[8.0, "x"])
    assert loss.sigma.loc[8.0, "y"] == 1e-3
    assert loss.sigma.loc[data.index[0], "x"] == SIGMA
    # A flux isn't measured, so it gets a new column
    assert merged(identify.Prediction("v1", time=float(data.index[3]))) is not None
    # Measured entries, other conditions and steady states are simulated apart
    assert merged(identify.Prediction("y", time=float(data.index[3]))) is None
    assert merged(identify.Prediction("y", time=8.0, parameters={"k2": 1.0})) is None
    assert merged(identify.Prediction("y")) is None


def test_bounded_prediction() -> None:
    # p can't exceed 2.5, so neither can the prediction
    pp = _source_profile(prediction=STEADY, bounds={"p": (0.01, 2.5)})
    assert pp.upper_status == "bound"
    assert pp.ci[1] == pytest.approx(2.5, rel=1e-4)
    assert pp.ci[0] == pytest.approx(2 - HALF, rel=2e-3)


def test_analytic_validation_interval() -> None:
    # Δ(z) = (z - 2)² / (2 (1 + sigma²))
    sigma = 0.5
    val = identify.validation_profile(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5},
        prediction=STEADY,
        sigma=sigma,
        loss_fn=NLL,
        parallel=False,
    )
    half = HALF * math.sqrt(1 + sigma**2)
    lower, upper = val.validation_ci
    assert lower == pytest.approx(2 - half, abs=2e-3 * half)
    assert upper == pytest.approx(2 + half, abs=2e-3 * half)
    z = val.delta.index.to_numpy()
    np.testing.assert_allclose(
        val.delta, (z - 2) ** 2 / (2 * (1 + sigma**2)), atol=1e-8
    )
    # Without the data point, the points lie on the prediction profile
    g = val.predicted.to_numpy()
    np.testing.assert_allclose(val.data_delta, (g - 2) ** 2 / 2, atol=1e-8)
    assert val.is_consistent(2 + 0.9 * half)
    assert not val.is_consistent(2 + 1.1 * half)
    assert not val.is_consistent(2 - 1.1 * half)


def test_band_methods_agree() -> None:
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 1.5})
    times = np.linspace(0, 10, 6)
    bands = {
        method: identify.prediction_band(
            _decay(),
            data,
            res,
            prediction="y",
            times=times,
            method=method,
            loss_fn=fit.GaussianNll(SIGMA),
            parallel=False,
        )
        for method in ("integrated", "pointwise")
    }
    integrated, pointwise = bands["integrated"], bands["pointwise"]
    assert integrated.profiles is None
    assert pointwise.profiles is not None
    assert set(pointwise.profiles) == set(times)
    np.testing.assert_allclose(integrated.value, pointwise.value)
    np.testing.assert_allclose(integrated.lower, pointwise.lower, rtol=1e-3)
    np.testing.assert_allclose(integrated.upper, pointwise.upper, rtol=1e-3)
    # y(0) = 0 is fixed, so the band has no width there
    assert integrated.lower.iloc[0] == 0.0
    assert integrated.upper.iloc[0] == 0.0
    assert (integrated.lower_status.iloc[1:] == "threshold").all()
    inner = integrated.value.iloc[1:]
    assert (integrated.lower.iloc[1:] < inner).all()
    assert (integrated.upper.iloc[1:] > inner).all()
    # The band matches the prediction profile at one of its times
    pp = identify.prediction_profile(
        _decay(),
        data,
        res,
        prediction=identify.Prediction("y", time=float(times[2])),
        loss_fn=fit.GaussianNll(SIGMA),
        parallel=False,
    )
    assert pp.ci == pytest.approx(
        (integrated.lower.iloc[2], integrated.upper.iloc[2]), rel=1e-3
    )


def test_band_close_to_linearised_band() -> None:
    # Nearly linear for small noise: close to the delta-method band
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 1.5})
    fim = identify.fisher_information(_decay(), data, res, sigma=SIGMA)
    band = identify.prediction_band(
        _decay(),
        data,
        res,
        prediction="y",
        times=[2.0, 5.0],
        loss_fn=fit.GaussianNll(SIGMA),
        parallel=False,
    )
    cov = fim.covariance.to_numpy()
    problem = setup_problem(
        _decay(),
        data,
        dict(res.best_pars),
        loss_fn=fit.GaussianNll(SIGMA),
        minimizer=None,
        y0=None,
        protocol=None,
        runner=None,
        bounds=None,
        prior=None,
        penalty=None,
        transform=None,
        postprocess_fn=None,
        standard_scale=True,
    )
    prediction = identify.Prediction("y", time=2.0)
    grad = problem.prediction_jacobian(
        prediction,
        problem.condition(prediction),
        {str(k): float(v) for k, v in fim.parameters.items()},
        [2.0, 5.0],
    )
    sd = np.sqrt(np.einsum("ti,ij,tj->t", grad, cov, grad))
    half = (band.upper - band.lower).to_numpy() / 2
    np.testing.assert_allclose(half, 1.96 * sd, rtol=0.05)


def test_two_parameter_region() -> None:
    # x = p / k: Δ = 2 (p / k - 2)², a valley along k = p / 2
    data, nll = pd.Series({"x": 2.0}), fit.GaussianNll(0.5)
    width = 0.5 * math.sqrt(2 * THRESHOLD_2D)
    for method in ("slices", "grid"):
        pl2d = identify.profile_likelihood_2d(
            _source(),
            data,
            {"p": 2.0, "k": 1.0},
            x="p",
            y="k",
            loss_fn=nll,
            method=method,
            bounds={"p": (1.0, 4.0), "k": (0.1, 10.0)},
            transform="log",
            n_steps=16,
            parallel=False,
        )
        assert pl2d.threshold == pytest.approx(THRESHOLD_2D)
        assert pl2d.log_x
        assert pl2d.log_y
        points = pl2d.points
        p, k = points["x"].to_numpy(), points["y"].to_numpy()
        # Nothing left to fit: every point is exact
        np.testing.assert_allclose(points["delta"], 2 * (p / k - 2) ** 2, atol=1e-8)
        assert len(pl2d.parameters) == len(points)
        region = pl2d.region()
        assert len(region) >= 3
        rtol = 1e-2 if method == "slices" else 0.1
        np.testing.assert_allclose(
            region["lower"], region.index / (2 + width), rtol=rtol
        )
        np.testing.assert_allclose(
            region["upper"], region.index / (2 - width), rtol=rtol
        )
    narrower = pl2d.region(alpha=0.5)
    width_narrow = narrower["upper"].to_numpy() - narrower["lower"].to_numpy()
    width_wide = region["upper"].to_numpy() - region["lower"].to_numpy()
    assert np.mean(width_narrow) < np.mean(width_wide)


def test_slices_cant_widen_region() -> None:
    pl2d = identify.profile_likelihood_2d(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.0, "k": 1.0},
        x="p",
        y="k",
        loss_fn=fit.GaussianNll(0.5),
        bounds={"p": (1.0, 4.0), "k": (0.1, 10.0)},
        transform="log",
        parallel=False,
    )
    with pytest.raises(ValueError, match="can't be wider"):
        pl2d.region(alpha=0.99)
    assert bool((pl2d.slices["lower_status"] == "threshold").all())


def test_parameter_and_prediction_region() -> None:
    data = _decay_data()
    res = _decay_fit(data, {"k1": 0.4, "k2": 1.5})
    prediction = identify.Prediction("y", time=8.0)
    pl2d = identify.profile_likelihood_2d(
        _decay(),
        data,
        res,
        x="k1",
        y=prediction,
        loss_fn=fit.GaussianNll(SIGMA),
        parallel=False,
    )
    assert pl2d.y_name == "y"
    # The pseudo data points lead the predictions away from each slice's start
    off = ~np.isclose(pl2d.points["z"], pl2d.points["y"], rtol=1e-6)
    assert off.mean() > 0.5
    region = pl2d.region()
    # The region's projection onto the prediction is the prediction's
    # interval at the threshold of the region
    pp = identify.prediction_profile(
        _decay(),
        data,
        res,
        prediction=prediction,
        loss_fn=fit.GaussianNll(SIGMA),
        threshold=THRESHOLD_2D,
        parallel=False,
    )
    assert np.min(region["lower"].to_numpy()) == pytest.approx(pp.ci[0], rel=2e-3)
    assert np.max(region["upper"].to_numpy()) == pytest.approx(pp.ci[1], rel=2e-3)


def test_repair_refits_kinks() -> None:
    data = pd.Series({"x": 2.0})
    start = {"p": 2.0, "k": 1.0}
    problem = setup_problem(
        _source(),
        data,
        start,
        loss_fn=NLL,
        minimizer=None,
        y0=None,
        protocol=None,
        runner=None,
        bounds=None,
        prior=None,
        penalty=None,
        transform=None,
        postprocess_fn=None,
        standard_scale=True,
    )
    axis = _Axis(problem=problem, x="k", y="p", objective=0.0, sigma=1.0)
    # Refits of p at fixed k = 1: the kink at p = 2.2 is a failed fit
    points = [
        {
            "x": 1.0,
            "y": y,
            "delta": (y - 2) ** 2 / 2,
            "z": math.nan,
            "pars": {"p": y, "k": 1.0},
        }
        for y in (1.6, 1.8, 2.0, 2.4, 2.6)
    ]
    points.insert(3, {"x": 1.0, "y": 2.2, "delta": 5.0, "z": math.nan, "pars": start})
    # There is nothing left to fit at fixed (k, p), so the repair only
    # re-evaluates, which still fixes the stored objective
    n = _repair(
        points,
        axis=axis,
        threshold=THRESHOLD_2D,
        log_x=False,
        log_y=False,
        parallel=False,
        max_workers=None,
    )
    assert n == 1
    assert points[3]["delta"] == pytest.approx(0.02, abs=1e-8)


def test_input_errors() -> None:
    data = pd.Series({"x": 2.0})
    with pytest.raises(ValueError, match="needs a time"):
        _source_profile(prediction=identify.Prediction("x", protocol=pd.DataFrame()))
    with pytest.raises(ValueError, match="non-negative"):
        identify.Prediction("x", time=-1.0)
    with pytest.raises(ValueError, match="Unknown prediction target"):
        _source_profile(prediction=identify.Prediction("nope"))
    with pytest.raises(ValueError, match="likelihood loss"):
        identify.prediction_profile(
            _source(), data, {"p": 2.0}, prediction=STEADY, loss_fn=fit.rmse
        )
    with pytest.raises(ValueError, match="sigma has to be positive"):
        identify.validation_profile(
            _source(), data, {"p": 2.0}, prediction=STEADY, sigma=0.0, loss_fn=NLL
        )
    with pytest.raises(ValueError, match="times="):
        identify.prediction_band(
            _decay(),
            _decay_data(),
            {"k1": 0.5},
            prediction=identify.Prediction("y", time=1.0),
            times=[1.0],
            loss_fn=NLL,
        )
    with pytest.raises(ValueError, match="grid method needs two parameters"):
        identify.profile_likelihood_2d(
            _source(), data, {"p": 2.0}, x="p", y=STEADY, loss_fn=NLL, method="grid"
        )
    with pytest.raises(ValueError, match="fitted ones"):
        identify.profile_likelihood_2d(
            _source(), data, {"p": 2.0}, x="p", y="k", loss_fn=NLL
        )


def test_plots() -> None:
    data = pd.Series({"x": 2.0})
    pp = _source_profile(prediction=STEADY)
    plot.prediction_profile(pp)
    val = identify.validation_profile(
        _source(),
        data,
        {"p": 2.0},
        prediction=STEADY,
        sigma=0.5,
        loss_fn=NLL,
        parallel=False,
    )
    _, ax = plot.prediction_profile(val)
    assert "new data point" in ax.get_xlabel()
    band = identify.prediction_band(
        _decay(),
        _decay_data(),
        _decay_fit(_decay_data(), {"k1": 0.4, "k2": 1.5}),
        prediction="y",
        times=[0.0, 2.0, 4.0],
        loss_fn=fit.GaussianNll(SIGMA),
        parallel=False,
    )
    plot.prediction_band(band)
    pl2d = identify.profile_likelihood_2d(
        _source(),
        data,
        {"p": 2.0, "k": 1.0},
        x="p",
        y="k",
        loss_fn=fit.GaussianNll(0.5),
        method="grid",
        bounds={"p": (1.0, 4.0), "k": (0.1, 10.0)},
        transform="log",
        n_steps=5,
        parallel=False,
    )
    _, ax = plot.profile_likelihood_2d(pl2d, show_points=True)
    assert ax.get_xlabel() == "log10 p"


def test_prediction_defaults_to_the_fits_protocol() -> None:
    protocol = make_protocol([(2, {"k1": 0.5}), (2, {"k1": 1.5})])
    t = np.linspace(0.2, 4.0, 15)
    sim = (
        Simulator(_decay())
        .simulate_protocol_time_course(protocol, time_points=[*t, 3.05])
        .get_result()
        .unwrap_or_err()
        .get_combined()
    )
    pp = identify.prediction_profile(
        _decay(),
        sim.loc[t, ["y"]],
        {"k2": 2.0},
        prediction=identify.Prediction("y", time=3.05),
        loss_fn=fit.GaussianNll(SIGMA),
        protocol=protocol,
        parallel=False,
    )
    assert pp.value == pytest.approx(float(sim.loc[3.05, "y"]), rel=1e-4)
    assert pp.ci[0] < pp.value < pp.ci[1]


def test_inconclusive_validation_raises() -> None:
    val = identify.validation_profile(
        _source(),
        pd.Series({"x": 2.0}),
        {"p": 2.5},
        prediction=STEADY,
        sigma=0.5,
        loss_fn=NLL,
        max_steps=1,
        parallel=False,
    )
    assert val.upper_status == "max_steps"
    assert val.is_consistent(2.0)
    with pytest.raises(ValueError, match="increase max_steps"):
        val.is_consistent(10.0)


def test_prediction_type_is_checked() -> None:
    with pytest.raises(TypeError, match=r"identify\.Prediction"):
        identify.prediction_profile(
            _source(),
            pd.Series({"x": 2.0}),
            {"p": 2.0},
            prediction="x",  # type: ignore[arg-type]
            loss_fn=NLL,
        )
