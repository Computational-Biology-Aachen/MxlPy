# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mxlpy import KineticModelBuilder, Simulator, fit, fns, plot, select
from mxlpy.distributions import LogUniform

mpl.use("Agg")

SIGMA = 0.01
NLL = fit.GaussianNll(SIGMA)
BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "k1": (1e-6, 1e3),
    "k2": (0.0, 1e3),
    "kx": (0.0, 1e3),
}
PARS = {
    "k1": LogUniform(1e-2, 1e1),
    "k2": LogUniform(1e-2, 1e1),
    "kx": LogUniform(1e-3, 1e0),
}


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


def _add_x_degradation(m: KineticModelBuilder) -> None:
    m.add_parameter("kx", 0.1)
    m.add_reaction("vx", fns.mass_action_1s, args=["x", "kx"], stoichiometry={"x": -1})


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    t = np.linspace(0.2, 6.0, 20)
    res = Simulator(_decay()).simulate_time_course(t).get_result().unwrap_or_err()
    df = res.get_combined().loc[t, ["x", "y"]]
    return df + np.random.default_rng(0).normal(0, SIGMA, df.shape)


@pytest.fixture
def space() -> select.ModelSpace[KineticModelBuilder]:
    # True model: y degrades (k2 free), x doesn't
    return select.ModelSpace(
        base=_decay(),
        switches={
            "deg_y": select.free_parameter("k2", 0.0),
            "deg_x": _add_x_degradation,
        },
    )


def _run(space, data, **kwargs):  # noqa: ANN001, ANN003, ANN202
    kwargs = {"n_starts": 2, "bounds": BOUNDS, "parallel": False, **kwargs}
    return select.run(space, data, PARS, loss_fn=NLL, **kwargs)


###############################################################################
# Criteria
###############################################################################


def test_criteria_formulas() -> None:
    assert select.aic(-10.0, 3) == pytest.approx(26.0)
    assert select.bic(-10.0, 3, 50) == pytest.approx(3 * math.log(50) + 20)
    assert select.aicc(-10.0, 3, 50) == pytest.approx(26.0 + 24 / 46)
    assert select.aicc(-10.0, 3, 4) == math.inf
    assert select.information_criterion("bic", -1.0, 2, 10) == select.bic(-1.0, 2, 10)
    with pytest.raises(ValueError, match="Unknown information criterion"):
        select.information_criterion("foo", -1.0, 2, 10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_parameters"):
        select.aic(0.0, -1)
    with pytest.raises(ValueError, match="n_data"):
        select.bic(0.0, 1, 0)


def test_model_weights() -> None:
    w = select.model_weights(pd.Series({"a": 0.0, "b": 2.0, "c": math.inf}))
    assert w.sum() == pytest.approx(1.0)
    wd = w.to_dict()
    assert wd["c"] == 0.0
    assert wd["a"] / wd["b"] == pytest.approx(math.e)
    assert (select.model_weights(pd.Series({"a": math.inf})) == 0).all()


def test_lrt() -> None:
    res = select.lrt(-10.0, -7.0, 2)
    assert res.statistic == pytest.approx(6.0)
    assert res.p_value == pytest.approx(stats.chi2.sf(6.0, 2))
    assert res.rejects(0.05)
    assert not res.rejects(0.01)
    # A worse fit of the larger model is clipped
    assert select.lrt(-7.0, -10.0, 1).p_value == 1.0
    with pytest.raises(ValueError, match="df"):
        select.lrt(-7.0, -6.0, 0)


###############################################################################
# Model space
###############################################################################


def test_space_ids_and_build(space) -> None:  # noqa: ANN001
    assert space.names == ["deg_y", "deg_x"]
    assert space.id([]) == select.BASE
    assert space.id(["deg_x", "deg_y"]) == "deg_y+deg_x"
    base, fixed = space.build([])
    assert fixed == {"k2": 0.0}
    assert base.get_parameter_values()["k2"] == 0.0
    assert "vx" not in base.get_reaction_names()
    full, fixed = space.build(["deg_x", "deg_y"])
    assert fixed == {}
    assert full.get_parameter_values()["k2"] == 2.0
    assert "vx" in full.get_reaction_names()
    # The base model isn't modified
    assert "kx" not in space.base.get_parameter_names()
    with pytest.raises(ValueError, match="Unknown switches"):
        space.build(["nope"])


def test_space_switch_validation() -> None:
    for name in ("a+b", "base", ""):
        with pytest.raises(ValueError, match="Switch names"):
            select.ModelSpace(base=_decay(), switches={name: lambda m: m})


def test_space_switch_on_off_returning_model() -> None:
    def replace(_: KineticModelBuilder) -> KineticModelBuilder:
        return _decay().update_parameter("k1", 9.0)

    space = select.ModelSpace(
        base=_decay(),
        switches={
            "s": select.Switch(
                on=replace,
                off=lambda m: m.update_parameter("k1", 3.0),
                fix_on={"k2": 1.0},
            )
        },
    )
    on, fixed_on = space.build(["s"])
    off, fixed_off = space.build([])
    assert on.get_parameter_values()["k1"] == 9.0
    assert on.get_parameter_values()["k2"] == 1.0
    assert fixed_on == {"k2": 1.0}
    assert off.get_parameter_values()["k1"] == 3.0
    assert fixed_off == {}


def test_space_neighbours_and_allowed() -> None:
    space = select.ModelSpace(
        base=_decay(),
        switches={"a": lambda m: m, "b": lambda m: m, "c": lambda m: m},
        allowed=lambda on: not {"a", "b"} <= on,
    )
    assert set(space.neighbours({"a"}, "forward")) == {frozenset({"a", "c"})}
    assert set(space.neighbours({"a", "c"}, "backward")) == {
        frozenset({"a"}),
        frozenset({"c"}),
    }
    assert set(space.neighbours({"a"}, "swap")) == {
        frozenset({"b"}),
        frozenset({"c"}),
    }
    assert len(space.candidates()) == 6
    assert not space.is_allowed({"a", "b"})


###############################################################################
# Search
###############################################################################


@pytest.mark.parametrize("method", ["forward", "backward", "adaptive", "brute_force"])
def test_run_selects_true_model(space, data, method) -> None:  # noqa: ANN001
    sel = _run(space, data, method=method)
    assert sel.best_id == "deg_y"
    assert sel.complete
    assert sel.path[-1] == "deg_y"
    best = sel.best
    assert best.n_parameters == 2
    assert best.best_pars["k1"] == pytest.approx(0.5, rel=0.05)
    assert best.best_pars["k2"] == pytest.approx(2.0, rel=0.05)
    assert sel.best_model.get_parameter_values()["k2"] == best.best_pars["k2"]
    if method == "forward":
        assert sel.path == ["base", "deg_y"]
    if method == "backward":
        assert sel.path == ["deg_y+deg_x", "deg_y"]
    if method == "brute_force":
        assert set(sel.candidates) == {"base", "deg_y", "deg_x", "deg_y+deg_x"}


def test_table(space, data) -> None:  # noqa: ANN001
    sel = _run(space, data, method="brute_force", criterion="bic")
    table = sel.table
    assert table.index[0] == "deg_y"
    assert table["bic"].is_monotonic_increasing
    assert float(table["delta"].iloc[0]) == 0.0
    assert table["weight"].to_numpy().sum() == pytest.approx(1.0)
    assert table.loc["deg_y+deg_x", "k"] == 3
    assert (table["n"] == data.size).all()
    assert bool(table.loc["deg_y", "deg_y"])
    assert not bool(table.loc["deg_y", "deg_x"])
    assert {"log_likelihood", "aic", "aicc", "converged", "n_failed", "parent"} <= set(
        table.columns
    )
    c = sel["deg_y"]
    assert c.log_likelihood == -c.loss
    assert c.aic == select.aic(c.log_likelihood, 2)
    assert len(c.objectives) == 3  # warm start + 2 random starts
    assert 0 < c.converged() <= 1


def test_free_parameter_warm_start_reproduces_parent(space, data) -> None:  # noqa: ANN001
    sel = _run(space, data, method="forward", n_starts=0)
    base, child = sel["base"], sel["deg_y"]
    assert child.parent == "base"
    # The warm start begins at the parent's optimum with k2 = 0, so the
    # nested model fits at least as well
    assert child.loss <= base.loss + 1e-6


def test_lrt_search(space, data) -> None:  # noqa: ANN001
    sel = _run(space, data, criterion="lrt", method="adaptive")
    assert sel.best_id == "deg_y"
    assert "delta" not in sel.table.columns
    test = sel.lrt("deg_y", "deg_y+deg_x")
    assert test.df == 1
    assert not test.rejects(sel.significance)
    assert sel.lrt("base", "deg_y").rejects()
    with pytest.raises(ValueError, match="brute_force"):
        _run(space, data, criterion="lrt", method="brute_force")


def test_max_candidates(space, data) -> None:  # noqa: ANN001
    with pytest.warns(UserWarning, match="max_candidates"):
        sel = _run(space, data, method="forward", max_candidates=2)
    assert not sel.complete
    assert sel.path == ["base"]
    assert set(sel.candidates) == {"base"}
    with pytest.raises(ValueError, match="max_candidates"):
        _run(space, data, method="brute_force", max_candidates=3)


def test_reuse(space, data) -> None:  # noqa: ANN001
    first = _run(space, data, method="forward")
    second = _run(space, data, method="backward", reuse=first, max_candidates=1)
    # Only deg_y+deg_x wasn't fitted by the forward search
    assert set(second.candidates) == set(first.candidates) | {"deg_y+deg_x"}
    assert second["deg_y"] is first["deg_y"]
    assert second.best_id == "deg_y"
    with pytest.raises(ValueError, match="same data"):
        _run(space, data.iloc[:5], reuse=first)


def test_seed_reproducible(space, data) -> None:  # noqa: ANN001
    a = _run(space, data, method="brute_force", seed=3)
    b = _run(space, data, method="brute_force", seed=3)
    for k in a.candidates:
        assert a[k].objectives == b[k].objectives


def test_fitted_initial_value_switch(data) -> None:  # noqa: ANN001
    space = select.ModelSpace(
        base=_decay(), switches={"x0": select.free_parameter("x", 1.5)}
    )
    sel = select.run(
        space,
        data,
        {"k1": 1.0, "k2": 1.0, "x": 2.0},
        loss_fn=NLL,
        method="brute_force",
        n_starts=0,
        parallel=False,
    )
    base, free = sel["base"], sel["x0"]
    assert base.fixed == {"x": 1.5}
    assert set(base.best_pars) == {"k1", "k2"}
    assert free.best_pars["x"] == pytest.approx(2.0, rel=0.02)
    assert free.model.get_initial_conditions()["x"] == free.best_pars["x"]
    assert sel.best_id == "x0"


def test_run_validation(space, data) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="likelihood"):
        select.run(space, data, PARS, loss_fn=fit.rmse)
    with pytest.raises(ValueError, match="criterion"):
        _run(space, data, criterion="foo")
    with pytest.raises(ValueError, match="method"):
        _run(space, data, method="foo")
    with pytest.raises(ValueError, match="n_starts"):
        _run(space, data, n_starts=-1)
    bad = select.ModelSpace(
        base=_decay(), switches={"s": select.free_parameter("nope")}
    )
    with pytest.raises(ValueError, match="unknown parameters"):
        select.run(bad, data, {"k1": 1.0}, loss_fn=NLL, parallel=False)


def test_parallel_matches_sequential(space, data) -> None:  # noqa: ANN001
    seq = _run(space, data, method="brute_force", n_starts=1)
    par = _run(space, data, method="brute_force", n_starts=1, parallel=True)
    assert par.best_id == seq.best_id
    for k in seq.candidates:
        assert par[k].loss == pytest.approx(seq[k].loss, rel=1e-6)


###############################################################################
# Compare
###############################################################################


def test_compare(space, data) -> None:  # noqa: ANN001
    models = {"two": _decay(), "three": space.build(["deg_y", "deg_x"])[0]}
    sel = select.compare(
        models, data, PARS, loss_fn=NLL, n_starts=1, bounds=BOUNDS, parallel=False
    )
    assert sel.best_id == "two"
    assert sel.switches == []
    assert sel["three"].n_parameters == 3
    assert sel["two"].parent is None
    assert not sel.lrt("two", "three").rejects()
    with pytest.raises(ValueError, match="compare needs"):
        select.compare(models, data, PARS, loss_fn=NLL, criterion="lrt")  # type: ignore[arg-type]


def test_compare_steady_state() -> None:
    def source(k_out: float) -> KineticModelBuilder:
        return (
            KineticModelBuilder()
            .add_parameters({"p": 3.0, "k": k_out})
            .add_variables({"x": 1.0})
            .add_reaction("vin", fns.constant, args=["p"], stoichiometry={"x": 1})
            .add_reaction(
                "vout", fns.mass_action_1s, args=["x", "k"], stoichiometry={"x": -1}
            )
        )

    sel = select.compare(
        {"fixed_p": source(1.0), "free": source(1.0)},
        pd.Series({"x": 2.0}),
        {"k": 1.0},
        loss_fn=fit.GaussianNll(0.1),
        n_starts=0,
        parallel=False,
    )
    assert sel["free"].best_pars["k"] == pytest.approx(1.5, rel=1e-3)
    assert sel["free"].n_data == 1


def test_plot_model_selection(space, data) -> None:  # noqa: ANN001
    sel = _run(space, data, method="brute_force")
    _, ax = plot.model_selection(sel, n_max=3)
    assert len(ax.patches) == 3
    assert ax.get_yticklabels()[0].get_text() == "deg_y"
    plot.model_selection(sel, criterion="bic", ax=ax)


###############################################################################
# Regressions
###############################################################################


def test_nothing_to_estimate_keeps_model_cache(data) -> None:  # noqa: ANN001
    model = _decay()
    sel = select.compare(
        {"fixed": model}, data, {}, loss_fn=NLL, n_starts=0, parallel=False
    )
    assert sel["fixed"].n_parameters == 0
    assert math.isfinite(sel["fixed"].loss)
    assert set(model.get_parameter_values()) == {"k1", "k2"}
    assert set(sel["fixed"].model.get_parameter_values()) == {"k1", "k2"}


def test_log_transform_with_zero_fixed_value(data) -> None:  # noqa: ANN001
    base = _decay().update_parameter("k2", 0.0)
    space = select.ModelSpace(
        base=base, switches={"deg_y": select.free_parameter("k2", 0.0)}
    )
    sel = select.run(
        space,
        data,
        {"k1": 1.0, "k2": 1.0},
        loss_fn=NLL,
        method="forward",
        n_starts=0,
        transform="log",
        parallel=False,
    )
    assert sel.best_id == "deg_y"
    assert sel.best.best_pars["k2"] == pytest.approx(2.0, rel=0.05)


def test_fixed_initial_value_set_on_model(data) -> None:  # noqa: ANN001
    space = select.ModelSpace(
        base=_decay(), switches={"x0": select.free_parameter("x", 1.5)}
    )
    sel = select.run(
        space,
        data,
        {"k1": 1.0, "k2": 1.0, "x": 2.0},
        loss_fn=NLL,
        method="brute_force",
        n_starts=0,
        parallel=False,
    )
    base = sel["base"]
    assert base.model.get_initial_conditions()["x"] == 1.5
    res = Simulator(base.model).simulate_time_course(data.index.to_numpy())
    sim = res.get_result().unwrap_or_err().get_combined().loc[data.index, data.columns]
    assert NLL(sim, data) == pytest.approx(base.loss, rel=1e-4)


def test_unknown_names_raise(space, data) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="neither parameters nor variables"):
        select.run(space, data, {**PARS, "k_typo": 1.0}, loss_fn=NLL, parallel=False)
    with pytest.raises(ValueError, match="not in p0"):
        select.run(space, data, PARS, loss_fn=NLL, bounds={"nope": (0, 1)})
    with pytest.raises(ValueError, match="neither parameters nor variables"):
        select.compare({"a": _decay()}, data, {"kx": 1.0}, loss_fn=NLL)


def test_negative_min_improvement_raises(space, data) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="min_improvement"):
        _run(space, data, min_improvement=-1.0)
