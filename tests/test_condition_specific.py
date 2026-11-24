# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import math

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, Simulator, fit, fns, plot, select
from mxlpy.select._conditions import _clusters, _Detector, _Layout

mpl.use("Agg")

SIGMA = 0.01
NLL = fit.GaussianNll(SIGMA)
P0 = {"k1": 1.0, "k2": 1.0, "k3": 1.0}
CANDIDATES = ["k1", "k2", "k3"]
TIMES = np.linspace(0.2, 6.0, 15)


def _chain() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0, "k3": 1.0})
        .add_variables({"x": 2.0, "y": 0.0, "z": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1, "z": 1}
        )
        .add_reaction(
            "v3", fns.mass_action_1s, args=["z", "k3"], stoichiometry={"z": -1}
        )
    )


def _data(seed: int, **parameters: float) -> pd.DataFrame:
    model = _chain().update_parameters(parameters)
    res = Simulator(model).simulate_time_course(TIMES).get_result().unwrap_or_err()
    df = res.get_combined().loc[TIMES, ["x", "y", "z"]]
    return df + np.random.default_rng(seed).normal(0, SIGMA, df.shape)


@pytest.fixture(scope="module")
def conditions() -> dict[str, fit.FitSettings]:
    # mut: k2 doubled; stress: k2 doubled and k3 halved
    return {
        "wt": fit.FitSettings(_chain(), _data(0)),
        "mut": fit.FitSettings(_chain(), _data(1, k2=4.0)),
        "stress": fit.FitSettings(_chain(), _data(2, k2=4.0, k3=0.5)),
    }


def _run(conditions, **kwargs):  # noqa: ANN001, ANN003, ANN202
    kwargs = {"lambdas": 8, "parallel": False, **kwargs}
    return select.condition_specific(
        conditions, P0, CANDIDATES, loss_fn=NLL, reference="wt", **kwargs
    )


@pytest.fixture(scope="module")
def result(conditions: dict[str, fit.FitSettings]) -> select.ConditionSpecific:
    return _run(conditions)


TRUE = {"mut": ["k2"], "stress": ["k2", "k3"]}


###############################################################################
# Selection
###############################################################################


def test_selects_true_structure(result: select.ConditionSpecific) -> None:
    assert result.selected == TRUE
    assert result.best_id == "k2[mut]+k2[stress]+k3[stress]"
    pars = result.best.parameters
    assert pars.loc["wt", "k2"] == pytest.approx(2.0, rel=0.02)
    assert pars.loc["mut", "k2"] == pytest.approx(4.0, rel=0.02)
    assert pars.loc["stress", "k3"] == pytest.approx(0.5, rel=0.02)
    # Shared values are the same in all conditions
    np.testing.assert_allclose(pars["k1"], pars.loc["wt", "k1"])


def test_path(result: select.ConditionSpecific) -> None:
    path = result.path
    assert list(path.columns.names) == ["parameter", "condition"]
    assert list(path.columns) == [(k, c) for k in CANDIDATES for c in ("mut", "stress")]
    assert len(path) == 8
    assert path.index.is_monotonic_decreasing
    assert list(result.path_fits.columns) == ["loss", "penalty", "structure"]
    # Weaker penalties fit better
    losses = result.path_fits["loss"].to_numpy()
    assert np.all(np.diff(losses) <= 1e-6 * losses[:-1])
    # The strongest penalty leaves (almost) everything shared, the weakest none
    assert result.path.iloc[0].abs().max() < 0.1
    assert result.path.iloc[-1][("k2", "mut")] == pytest.approx(math.log(2), abs=0.05)


def test_candidates_and_table(result: select.ConditionSpecific) -> None:
    assert result.conditions == ["wt", "mut", "stress"]
    assert result.reference == "wt"
    assert "shared" in result.fits
    full = result[result.full_id]
    assert full.n_parameters == 3 + 6
    assert result["shared"].n_parameters == 3
    table = result.table
    assert list(table.columns) == [
        "k",
        "n",
        "log_likelihood",
        "aic",
        "aicc",
        "bic",
        "p_value",
        "lambda",
    ]
    assert table["k"].is_monotonic_increasing
    assert math.isnan(table.loc[result.full_id, "p_value"])
    assert table.loc["shared", "p_value"] < 1e-6
    assert table.loc[result.best_id, "p_value"] > result.significance
    assert table["n"].unique().tolist() == [3 * 15 * 3]
    for fit_ in result.fits.values():
        assert fit_.loss >= full.loss - 1e-6 * abs(full.loss)


def test_best_names_and_differs(result: select.ConditionSpecific) -> None:
    best = result.best
    assert list(best.best_pars) == [
        "k1",
        "k2",
        "k3",
        "k2[mut]",
        "k2[stress]",
        "k3[stress]",
    ]
    assert best.structure == {
        "k1": (("wt", "mut", "stress"),),
        "k2": (("wt",), ("mut",), ("stress",)),
        "k3": (("wt", "mut"), ("stress",)),
    }
    assert best.differs == TRUE
    assert best.lambdas


def test_lrt(result: select.ConditionSpecific) -> None:
    test = result.lrt("shared")
    assert test.df == 6
    assert test.rejects()
    assert not result.lrt(result.best_id).rejects()
    assert result.lrt("shared", result.best_id).df == 3


def test_models(result: select.ConditionSpecific) -> None:
    models = result.models()
    assert list(models) == ["wt", "mut", "stress"]
    pars = result.best.parameters
    for c, model in models.items():
        values = model.get_parameter_values()
        for k in CANDIDATES:
            assert values[k] == pytest.approx(pars.loc[c, k])
    shared = result.models("shared")
    assert shared["wt"].get_parameter_values()["k2"] == pytest.approx(
        shared["mut"].get_parameter_values()["k2"]
    )


@pytest.mark.parametrize("criterion", ["bic", "aicc"])
def test_information_criteria(result: select.ConditionSpecific, criterion: str) -> None:
    fits = {k: v.criterion(criterion) for k, v in result.fits.items()}  # type: ignore[arg-type]
    best = min(fits, key=fits.__getitem__)
    assert best == result.best_id


def test_criterion_bic(conditions: dict[str, fit.FitSettings]) -> None:
    res = _run(conditions, criterion="bic", lambdas=[1e3, 3e2, 1e2, 30.0])
    assert res.criterion == "bic"
    np.testing.assert_allclose(res.path.index, [1e3, 3e2, 1e2, 30.0])
    bics = {k: v.bic for k, v in res.fits.items()}
    assert res.best_id == min(bics, key=bics.__getitem__)
    table = res.table
    assert table.index[0] == res.best_id
    assert table["delta"].iloc[0] == 0.0
    assert table["weight"].to_numpy(dtype=float).sum() == pytest.approx(1.0)
    assert table["bic"].is_monotonic_increasing


def test_default_grid_starts_shared(result: select.ConditionSpecific) -> None:
    assert result.path_fits["structure"].iloc[0] == "shared"
    assert result["shared"].lambdas
    assert result.path.index[0] == pytest.approx(1.2 * result.path.index[-1] * 1e3)


def test_multistart(conditions: dict[str, fit.FitSettings]) -> None:
    from mxlpy.distributions import LogUniform  # noqa: PLC0415

    p0 = {"k1": LogUniform(0.1, 10.0), "k2": 1.0, "k3": LogUniform(0.1, 10.0)}
    res = select.condition_specific(
        conditions,
        p0,
        CANDIDATES,
        loss_fn=NLL,
        lambdas=6,
        n_starts=3,
        bounds={"k1": (1e-3, 1e2)},
        parallel=False,
    )
    shared = res["shared"]
    assert len(shared.objectives) == 4
    assert shared.n_failed == 0
    assert shared.objective == pytest.approx(min(shared.objectives))
    assert res.selected == TRUE
    assert all(len(f.objectives) == 1 for k, f in res.fits.items() if k != "shared")


def test_explicit_lambdas_are_sorted(conditions: dict[str, fit.FitSettings]) -> None:
    res = _run(conditions, lambdas=[10.0, 1e3, 100.0])
    np.testing.assert_allclose(res.path.index, [1e3, 100.0, 10.0])


###############################################################################
# Regularisations
###############################################################################


def test_adaptive_lasso(conditions: dict[str, fit.FitSettings]) -> None:
    assert _run(conditions, regularisation="adaptive").selected == TRUE


def test_elastic_net(conditions: dict[str, fit.FitSettings]) -> None:
    # The ridge part spreads the fold-changes, so the path can miss the true
    # structure; the selection still contains the true differences
    res = _run(conditions, regularisation="elastic_net")
    for c, names in TRUE.items():
        assert set(names) <= set(res.selected[c])
    if res.best_id != res.full_id:
        assert not res.lrt(res.best_id).rejects()


def test_group_lasso(conditions: dict[str, fit.FitSettings]) -> None:
    res = _run(conditions, regularisation="group")
    # A parameter differs in all conditions or none
    for groups in res.best.structure.values():
        assert len(groups) in (1, 3)
    assert res.selected == {"mut": ["k2", "k3"], "stress": ["k2", "k3"]}


def test_fused_lasso_shares_values(conditions: dict[str, fit.FitSettings]) -> None:
    res = _run(conditions, regularisation="fused")
    assert res.best_id == "k2[mut,stress]+k3[stress]"
    assert res.best.structure["k2"] == (("wt",), ("mut", "stress"))
    pars = res.best.parameters
    assert pars.loc["mut", "k2"] == pytest.approx(pars.loc["stress", "k2"])
    assert res.selected == TRUE


###############################################################################
# Other settings
###############################################################################


def test_default_reference_and_parallel(
    conditions: dict[str, fit.FitSettings],
) -> None:
    reordered = {k: conditions[k] for k in ("mut", "wt", "stress")}
    res = select.condition_specific(
        reordered, P0, ["k2", "k3"], loss_fn=NLL, lambdas=6, parallel=True
    )
    assert res.reference == "mut"
    assert res.conditions == ["mut", "wt", "stress"]
    # Relative to mut, wt has a different k2 and stress a different k3
    assert res.selected == {"wt": ["k2"], "stress": ["k3"]}


def test_steady_state_conditions() -> None:
    def model() -> KineticModelBuilder:
        return (
            KineticModelBuilder()
            .add_parameters({"vin": 1.0, "k": 0.5, "kb": 2.0})
            .add_variables({"x": 1.0, "y": 0.0})
            .add_reaction("v0", fns.constant, args=["vin"], stoichiometry={"x": 1})
            .add_reaction(
                "v1",
                fns.mass_action_1s,
                args=["x", "k"],
                stoichiometry={"x": -1, "y": 1},
            )
            .add_reaction(
                "v2", fns.mass_action_1s, args=["y", "kb"], stoichiometry={"y": -1}
            )
        )

    # x = vin / k, y = vin / kb; only k differs in the mutant
    data = {
        "wt": pd.Series({"x": 2.0, "y": 0.5}),
        "mut": pd.Series({"x": 1.0, "y": 0.5}),
    }
    res = select.condition_specific(
        {c: fit.FitSettings(model(), d) for c, d in data.items()},
        {"k": 1.0, "kb": 1.0},
        ["k", "kb"],
        loss_fn=fit.GaussianNll(0.01),
        lambdas=6,
        parallel=False,
    )
    assert res.selected == {"mut": ["k"]}
    assert res.best.parameters.loc["mut", "k"] == pytest.approx(1.0, rel=1e-3)


def test_log_transform_and_bounds(conditions: dict[str, fit.FitSettings]) -> None:
    res = _run(
        conditions,
        transform="log",
        bounds={"k1": (1e-3, 1e2), "k2": (1e-3, 1e2), "k3": (1e-3, 1e2)},
    )
    assert res.selected == TRUE


def test_plot(result: select.ConditionSpecific) -> None:
    _, ax = plot.condition_specific_path(result)
    assert len(ax.get_lines()) >= 6
    assert ax.get_xscale() == "log"
    assert result.best_id in ax.get_title()


###############################################################################
# Checks
###############################################################################


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reference": "nope"}, "Reference"),
        ({"candidates": []}, "No candidate"),
        ({"candidates": ["k1", "k1"]}, "unique"),
        ({"candidates": ["kx"]}, "in p0"),
        ({"p0": {"k1": -1.0, "k2": 1.0, "k3": 1.0}}, "positive values"),
        ({"regularisation": "lasso"}, "Unknown regularisation"),
        ({"criterion": "aicx"}, "Unknown criterion"),
        ({"gamma": 0.0}, "gamma"),
        ({"n_starts": -1}, "n_starts"),
        ({"bounds": {"kx": (0.0, 1.0)}}, "bounds of parameters"),
        ({"transform": {"kx": "log"}}, "transform of parameters"),
        ({"loss_fn": fit.rmse}, "likelihood"),
        ({"lambdas": 0}, "at least one"),
        ({"lambdas": [1.0, -1.0]}, "positive"),
        ({"significance": 1.0}, "significance"),
        ({"l1_ratio": 0.0}, "l1_ratio"),
        ({"max_fold_change": 1.0}, "max_fold_change"),
        ({"p0": {"k1": 1.0, "k2": 1.0, "k3": 1.0, "nope": 1.0}}, "neither"),
    ],
)
def test_checks(conditions: dict[str, fit.FitSettings], kwargs, match: str) -> None:  # noqa: ANN001
    args = {
        "conditions": conditions,
        "p0": P0,
        "candidates": CANDIDATES,
        "loss_fn": NLL,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        select.condition_specific(**args)


def test_condition_loss_has_to_be_a_likelihood(
    conditions: dict[str, fit.FitSettings],
) -> None:
    wt = conditions["wt"]
    rmse = fit.FitSettings(wt.model, wt.data, loss_fn=fit.rmse)
    with pytest.raises(ValueError, match="condition 'mut'"):
        select.condition_specific({"wt": wt, "mut": rmse}, P0, CANDIDATES, loss_fn=NLL)


def test_needs_two_conditions(conditions: dict[str, fit.FitSettings]) -> None:
    with pytest.raises(ValueError, match="two conditions"):
        select.condition_specific({"wt": conditions["wt"]}, P0, CANDIDATES, loss_fn=NLL)


def test_models_need_the_candidates(conditions: dict[str, fit.FitSettings]) -> None:
    other = _chain().remove_reaction("v3").remove_parameter("k3")
    with pytest.raises(ValueError, match="doesn't have the candidates"):
        select.condition_specific(
            {
                "wt": conditions["wt"],
                "ko": fit.FitSettings(other, conditions["wt"].data),
            },
            P0,
            CANDIDATES,
            loss_fn=NLL,
        )


###############################################################################
# Internals
###############################################################################


def test_clusters() -> None:
    assert _clusters({"a": 0.0, "b": 0.001, "c": 1.0, "d": 1.0015}, 0.002) == [
        ["a", "b"],
        ["c", "d"],
    ]
    assert _clusters({"a": 0.0}, 0.1) == [["a"]]


def test_detect() -> None:
    def detector(regularisation: str) -> _Detector:
        return _Detector(
            regularisation=regularisation,  # type: ignore[arg-type]
            candidates=["k"],
            conditions=["r", "a", "b"],
            reference="r",
            smoothing=1e-3,
            l1_ratio=1.0,
        )

    values = {"k[a]": 0.5, "k[b]": 0.5005}
    assert detector("l1").detect(values) == ((("r",), ("a",), ("b",)),)
    assert detector("fused").detect(values) == ((("r",), ("a", "b")),)
    assert detector("l1").detect({"k[a]": 1e-4, "k[b]": 0.5}) == ((("r", "a"), ("b",)),)
    assert detector("group").detect({"k[a]": 1e-4, "k[b]": 1e-4}) == (
        (("r", "a", "b"),),
    )
    assert detector("group").detect({"k[a]": 1e-4, "k[b]": 0.5}) == (
        (("r",), ("a",), ("b",)),
    )


def test_layout_matrix_matches_finite_differences() -> None:
    layout = _Layout(
        structure={"k": (("r",), ("a", "b"))},
        theta=["k", "p"],
        local=[["k", "p"], ["k", "p"], ["k"]],
        conditions=["r", "a", "b"],
    )
    assert layout.names == ["k", "p", "k[a,b]"]
    values = {"k": 2.0, "p": 3.0, "k[a,b]": 0.4}
    for i in range(3):
        m = layout.matrix(i, values)
        for j, name in enumerate(layout.names):
            h = 1e-6
            up = layout.updates(i, values | {name: values[name] + h})
            down = layout.updates(i, values | {name: values[name] - h})
            fd = [(up[k] - down[k]) / (2 * h) for k in layout.local[i]]
            np.testing.assert_allclose(m[:, j], fd, rtol=1e-6, atol=1e-9)
    assert layout.updates(1, values)["k"] == pytest.approx(2.0 * math.exp(0.4))
    pars = layout.parameters(values)
    assert pars.loc["r", "k"] == 2.0
    assert pars.loc["b", "k"] == pytest.approx(2.0 * math.exp(0.4))
