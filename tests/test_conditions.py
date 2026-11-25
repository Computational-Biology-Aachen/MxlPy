# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Several data sets (`fit.Condition`) in the identifiability and selection analyses."""

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns, identify, select
from mxlpy.fit.abstract import as_conditions

if TYPE_CHECKING:
    from mxlpy.fit.callbacks import FitEvent

SIGMA = 0.02
TIMES = np.linspace(0.2, 6.0, 15)


def _decay() -> KineticModelBuilder:
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


def _data(seed: int, *, k2: float = 2.0, x0: float = 2.0) -> pd.DataFrame:
    model = _decay().update_parameter("k2", k2).update_variable("x", x0)
    res = Simulator(model).simulate_time_course(TIMES).get_result().unwrap_or_err()
    data = res.get_combined().loc[TIMES, ["x", "y"]]
    return data + np.random.default_rng(seed).normal(0, SIGMA, data.shape)


@pytest.fixture(scope="module")
def conditions() -> dict[str, fit.Condition]:
    return {
        "low": fit.Condition(_data(1, k2=1.0), parameters={"k2": 1.0}, sigma=SIGMA),
        "high": fit.Condition(
            _data(2, k2=4.0, x0=3.0), parameters={"k2": 4.0}, y0={"x": 3.0}, sigma=SIGMA
        ),
    }


def test_as_conditions_forms() -> None:
    df = _data(0)
    assert list(as_conditions(df)) == ["data"]
    single = as_conditions(df, y0={"x": 1.0})["data"]
    assert single.y0 == {"x": 1.0}
    both = as_conditions(
        {"a": df, "b": fit.Condition(df, y0={"x": 3.0})}, y0={"x": 1.0}
    )
    assert both["a"].y0 == {"x": 1.0}
    assert both["b"].y0 == {"x": 3.0}
    with pytest.raises(ValueError, match="No data"):
        as_conditions({})
    with pytest.raises(TypeError, match="needs a time course"):
        as_conditions(fit.Condition(df.iloc[-1], protocol=pd.DataFrame()))


def test_condition_loss() -> None:
    assert isinstance(
        fit.Condition(_data(0), sigma=0.1).loss(fit.rmse), fit.GaussianNll
    )
    assert fit.Condition(_data(0)).loss(fit.rmse) is fit.rmse
    with pytest.raises(ValueError, match="sigma for every condition"):
        fit.Condition(_data(0)).loss(None)


def test_fisher_information_of_copies_adds_up() -> None:
    df = _data(0)
    pars = {"k1": 0.5, "k2": 2.0}
    one = identify.fisher_information(_decay(), df, pars, sigma=SIGMA)
    two = identify.fisher_information(_decay(), {"a": df, "b": df}, pars, sigma=SIGMA)
    np.testing.assert_allclose(two.matrix, 2 * one.matrix, rtol=1e-8)
    assert two.design.index.names == ["condition", "time", "observable"]
    assert set(two.design.index.get_level_values("condition")) == {"a", "b"}


def test_fisher_information_condition_parameters(
    conditions: dict[str, fit.Condition],
) -> None:
    fim = identify.fisher_information(_decay(), conditions, {"k1": 0.5})
    assert not fim.sigma_estimated
    assert fim.rank == 1
    with pytest.raises(ValueError, match="all data sets or for none"):
        identify.fisher_information(
            _decay(),
            {"a": _data(0), "b": fit.Condition(_data(1), sigma=SIGMA)},
            {"k1": 0.5},
        )


def test_profile_of_copies_is_narrower() -> None:
    df = _data(0)
    nll = fit.GaussianNll(SIGMA)
    one = identify.profile_likelihood(
        _decay(), df, {"k1": 0.5, "k2": 2.0}, loss_fn=nll, parallel=False
    )
    two = identify.profile_likelihood(
        _decay(),
        {"a": df, "b": df},
        {"k1": 0.5, "k2": 2.0},
        loss_fn=nll,
        parallel=False,
    )
    assert two.objective == pytest.approx(2 * one.objective, rel=1e-6)
    assert two.optimum["k1"] == pytest.approx(one.optimum["k1"], rel=1e-4)
    w1 = one["k1"].ci_upper - one["k1"].ci_lower
    w2 = two["k1"].ci_upper - two["k1"].ci_lower
    # twice the data: sqrt(2) narrower (close to linear)
    assert w1 / w2 == pytest.approx(math.sqrt(2), rel=0.05)


def test_profile_with_conditions_and_sigma(
    conditions: dict[str, fit.Condition],
) -> None:
    events: list[FitEvent] = []
    pl = identify.profile_likelihood(
        _decay(), conditions, {"k1": 0.6}, parallel=False, callback=events.append
    )
    assert pl.optimum["k1"] == pytest.approx(0.5, rel=0.05)
    assert pl["k1"].identifiability == "identifiable"
    assert any(e.kind == "finished" for e in events)


def test_every_condition_setting_a_parameter_raises(
    conditions: dict[str, fit.Condition],
) -> None:
    with pytest.raises(ValueError, match="can't be estimated"):
        identify.profile_likelihood(
            _decay(), conditions, {"k1": 0.5, "k2": 2.0}, parallel=False
        )


def test_unknown_condition_parameters_raise() -> None:
    with pytest.raises(ValueError, match="unknown parameters"):
        identify.profile_likelihood(
            _decay(),
            {"a": fit.Condition(_data(0), parameters={"nope": 1.0}, sigma=SIGMA)},
            {"k1": 0.5},
            parallel=False,
        )


def test_missing_loss_raises() -> None:
    with pytest.raises(ValueError, match="sigma for every condition"):
        identify.profile_likelihood(
            _decay(), {"a": _data(0), "b": _data(1)}, {"k1": 0.5}, parallel=False
        )


def test_identifiability_test_with_conditions(
    conditions: dict[str, fit.Condition],
) -> None:
    test = identify.identifiability_test(
        _decay(), conditions, {"k1": 0.5}, n_fits=3, parallel=False
    )
    assert test.is_identifiable


def test_prediction_band_with_conditions(conditions: dict[str, fit.Condition]) -> None:
    band = identify.prediction_band(
        _decay(),
        conditions,
        {"k1": 0.5},
        prediction=identify.Prediction("y", parameters={"k2": 4.0}, y0={"x": 3.0}),
        times=[1.0, 3.0],
        parallel=False,
    )
    lower = band.lower.to_numpy(dtype=float)
    upper = band.upper.to_numpy(dtype=float)
    assert (lower < upper).all()


def test_select_compare_with_conditions(conditions: dict[str, fit.Condition]) -> None:
    def wrong(m: KineticModelBuilder) -> KineticModelBuilder:
        return m.update_reaction("v1", fn=fns.constant, args=["k1"])

    sel = select.compare(
        {"true": _decay(), "wrong": wrong(_decay())},
        conditions,
        {"k1": 0.6},
        n_starts=2,
        parallel=False,
    )
    assert sel.best_id == "true"
    n_data = sum(c.data.size for c in conditions.values())
    assert sel["true"].n_data == n_data


def test_local_estimates_differ_per_condition() -> None:
    conds = {
        "wt": fit.Condition(_data(1), sigma=SIGMA),
        "mut": fit.Condition(_data(2, k2=4.0), local={"k2": "k2_mut"}, sigma=SIGMA),
    }
    fim = identify.fisher_information(
        _decay(), conds, {"k1": 0.5, "k2": 2.0, "k2_mut": 4.0}
    )
    assert fim.rank == 3
    pl = identify.profile_likelihood(
        _decay(),
        conds,
        {"k1": 0.6, "k2": 1.5, "k2_mut": 3.0},
        names=["k2_mut"],
        parallel=False,
    )
    assert pl.optimum["k2"] == pytest.approx(2.0, rel=0.05)
    assert pl.optimum["k2_mut"] == pytest.approx(4.0, rel=0.1)
    assert pl["k2_mut"].ci_lower < 4.0 < pl["k2_mut"].ci_upper


def test_local_names_have_to_be_estimated() -> None:
    conds = {
        "wt": fit.Condition(_data(1), sigma=SIGMA),
        "mut": fit.Condition(_data(2), local={"k2": "k2_mut"}, sigma=SIGMA),
    }
    with pytest.raises(ValueError, match="aren't estimated"):
        identify.profile_likelihood(_decay(), conds, {"k1": 0.5}, parallel=False)


def test_select_every_condition_setting_a_parameter_raises(
    conditions: dict[str, fit.Condition],
) -> None:
    with pytest.raises(ValueError, match="can't be estimated"):
        select.compare(
            {"m": _decay()},
            conditions,
            {"k1": 0.5, "k2": 2.0},
            n_starts=0,
            parallel=False,
        )


def test_several_conditions_with_other_losses() -> None:
    data = {"a": _data(1), "b": _data(2)}
    pl = identify.profile_likelihood(
        _decay(),
        data,
        {"k1": 0.5, "k2": 2.0},
        loss_fn=fit.rmse,
        threshold=0.05,
        names=["k1"],
        parallel=False,
    )
    assert pl.optimum["k1"] == pytest.approx(0.5, rel=0.05)
    with pytest.raises(ValueError, match="sum-of-squares"):
        identify.profile_likelihood(
            _decay(),
            data,
            {"k1": 0.5, "k2": 2.0},
            loss_fn=fit.rmse,
            threshold=0.05,
            minimizer=fit.LeastSquaresMinimizer(),
            parallel=False,
        )


def test_identify_refit_events_are_labelled(
    conditions: dict[str, fit.Condition],
) -> None:
    events: list[FitEvent] = []
    identify.prediction_band(
        _decay(),
        conditions,
        {"k1": 0.5},
        prediction=identify.Prediction("y", parameters={"k2": 4.0}),
        times=[1.0],
        parallel=False,
        callback=events.append,
    )
    starts = {e.start for e in events}
    assert None not in starts
    assert len(starts) > 1
    ended = {e.start for e in events if e.kind in ("finished", "failed")}
    assert ended == starts
