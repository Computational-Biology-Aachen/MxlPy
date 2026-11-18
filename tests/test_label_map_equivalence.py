# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Cross-model invariants of LabelMapper and LinearLabelMapper.

All networks use mass-action kinetics (for which isotopomer rates are exact)
and non-self-inverse label maps, so that the direction in which a label map
is applied is observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mxlpy import LabelMapper, LinearLabelMapper, Model, Simulator, fns

if TYPE_CHECKING:
    from collections.abc import Callable

TIME_POINTS = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])


@dataclass
class Network:
    build: Callable[[dict[str, float]], Model]
    steady_state: dict[str, float]
    label_variables: dict[str, int]
    label_maps: dict[str, list[int]]
    initial_labels: dict[str, int | list[int]]
    closed: bool


def _reversible_isomerisation(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"k1": 1.0, "k2": 1.0}).add_variables(y0)
    m.add_reaction(
        "v1", fns.mass_action_1s, args=["A", "k1"], stoichiometry={"A": -1, "B": 1}
    )
    m.add_reaction(
        "v2", fns.mass_action_1s, args=["B", "k2"], stoichiometry={"B": -1, "A": 1}
    )
    return m


def _condensation_cleavage(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"kf": 1.0, "kr": 1.0}).add_variables(y0)
    m.add_reaction(
        "v1",
        fns.mass_action_2s,
        args=["A", "C", "kf"],
        stoichiometry={"A": -1, "C": -1, "D": 1},
    )
    m.add_reaction(
        "v2",
        fns.mass_action_1s,
        args=["D", "kr"],
        stoichiometry={"D": -1, "A": 1, "C": 1},
    )
    return m


def _rubisco_like(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"kf": 1.0, "kr": 1.0}).add_variables(y0)
    m.add_reaction(
        "v1",
        fns.mass_action_2s,
        args=["A", "C", "kf"],
        stoichiometry={"A": -1, "C": -1, "P": 2},
    )
    m.add_reaction(
        "v2",
        fns.mass_action_2s,
        args=["P", "P", "kr"],
        stoichiometry={"P": -2, "A": 1, "C": 1},
    )
    return m


def _open_chain(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"k_in": 1.0, "k1": 1.0, "k_out": 1.0}).add_variables(y0)
    m.add_reaction("v_in", fns.constant, args=["k_in"], stoichiometry={"A": 1})
    m.add_reaction(
        "v1", fns.mass_action_1s, args=["A", "k1"], stoichiometry={"A": -1, "B": 1}
    )
    m.add_reaction(
        "v_out", fns.mass_action_1s, args=["B", "k_out"], stoichiometry={"B": -1}
    )
    return m


def _unlabelled_cofactor(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"k_atp": 1.0, "k1": 1.0, "k2": 1.0}).add_variables(y0)
    m.add_reaction("v_atp", fns.constant, args=["k_atp"], stoichiometry={"ATP": 1})
    m.add_reaction(
        "v1",
        fns.mass_action_2s,
        args=["A", "ATP", "k1"],
        stoichiometry={"A": -1, "ATP": -1, "B": 1},
    )
    m.add_reaction(
        "v2", fns.mass_action_1s, args=["B", "k2"], stoichiometry={"B": -1, "A": 1}
    )
    return m


def _labelled_modifier(y0: dict[str, float]) -> Model:
    m = Model().add_parameters({"k1": 1.0, "k2": 1.0}).add_variables(y0)
    m.add_reaction(
        "v1",
        fns.mass_action_2s,
        args=["A", "C", "k1"],
        stoichiometry={"A": -1, "B": 1},
    )
    m.add_reaction(
        "v2", fns.mass_action_1s, args=["B", "k2"], stoichiometry={"B": -1, "A": 1}
    )
    return m


NETWORKS = {
    "isomerisation": Network(
        build=_reversible_isomerisation,
        steady_state={"A": 1.0, "B": 1.0},
        label_variables={"A": 3, "B": 3},
        label_maps={"v1": [1, 2, 0], "v2": [0, 1, 2]},
        initial_labels={"A": 0},
        closed=True,
    ),
    "condensation": Network(
        build=_condensation_cleavage,
        steady_state={"A": 1.0, "C": 1.0, "D": 1.0},
        label_variables={"A": 2, "C": 1, "D": 3},
        label_maps={"v1": [2, 0, 1], "v2": [2, 0, 1]},
        initial_labels={"A": 0},
        closed=True,
    ),
    "rubisco_like": Network(
        build=_rubisco_like,
        steady_state={"A": 1.0, "C": 1.0, "P": 1.0},
        label_variables={"A": 5, "C": 1, "P": 3},
        label_maps={"v1": [2, 1, 0, 5, 3, 4], "v2": [0, 1, 2, 3, 4, 5]},
        initial_labels={"C": 0},
        closed=True,
    ),
    "open_chain": Network(
        build=_open_chain,
        steady_state={"A": 1.0, "B": 1.0},
        label_variables={"A": 3, "B": 3},
        label_maps={"v_in": [0, 1, 2], "v1": [1, 2, 0], "v_out": [0, 1, 2]},
        initial_labels={"A": 0},
        closed=False,
    ),
    "unlabelled_cofactor": Network(
        build=_unlabelled_cofactor,
        steady_state={"A": 1.0, "ATP": 1.0, "B": 1.0},
        label_variables={"A": 3, "B": 3},
        label_maps={"v1": [1, 2, 0], "v2": [0, 1, 2]},
        initial_labels={"A": 0},
        closed=True,
    ),
    "labelled_modifier": Network(
        build=_labelled_modifier,
        steady_state={"A": 1.0, "B": 2.0, "C": 2.0},
        label_variables={"A": 3, "B": 3, "C": 1},
        label_maps={"v1": [1, 2, 0], "v2": [0, 1, 2]},
        initial_labels={"A": 0, "C": 0},
        closed=True,
    ),
}
CLOSED_NETWORKS = {k: v for k, v in NETWORKS.items() if v.closed}


def _simulate(model: Model, time_points: np.ndarray = TIME_POINTS) -> pd.DataFrame:
    return (
        Simulator(model)
        .simulate_time_course(time_points)
        .get_result()
        .unwrap_or_err()
        .variables
    )


def _isotopomer_model(net: Network, y0: dict[str, float] | None = None) -> Model:
    return LabelMapper(
        net.build(net.steady_state if y0 is None else y0),
        label_variables=net.label_variables,
        label_maps=net.label_maps,
    ).build_model(initial_labels=net.initial_labels)


def _linear_model(net: Network) -> Model:
    base = net.build(net.steady_state)
    concs, fluxes = Simulator(base).simulate(1.0).get_result().unwrap_or_err()
    return LinearLabelMapper(
        base,
        label_variables=net.label_variables,
        label_maps=net.label_maps,
    ).build_model(
        concs=concs.iloc[-1],
        fluxes=fluxes.iloc[-1],
        external_label=1.0,
        initial_labels=net.initial_labels,
    )


def _positional_enrichment(
    isotopomers: pd.DataFrame, name: str, n_labels: int
) -> pd.DataFrame:
    """Fraction of `name` labelled at each position, from isotopomer trajectories."""
    total = isotopomers[f"{name}__total"]
    columns = [c for c in isotopomers.columns if c.startswith(f"{name}__")]
    columns.remove(f"{name}__total")
    return pd.DataFrame(
        {
            f"{name}__{pos}": sum(
                isotopomers[c] for c in columns if c.split("__")[1][pos] == "1"
            )
            / total
            for pos in range(n_labels)
        }
    )


def _labelled_atoms(isotopomers: pd.DataFrame, net: Network) -> pd.Series:
    columns = [
        c
        for name in net.label_variables
        for c in isotopomers.columns
        if c.startswith(f"{name}__") and c != f"{name}__total"
    ]
    weights = pd.Series({c: c.split("__")[1].count("1") for c in columns})
    return isotopomers[columns].mul(weights).sum(axis=1)


@pytest.mark.parametrize("network", NETWORKS.values(), ids=NETWORKS.keys())
def test_linear_enrichment_matches_isotopomer_model(network: Network) -> None:
    isotopomers = _simulate(_isotopomer_model(network))
    linear = _simulate(_linear_model(network))

    for name, n_labels in network.label_variables.items():
        expected = _positional_enrichment(isotopomers, name, n_labels)
        pd.testing.assert_frame_equal(
            linear.loc[:, expected.columns], expected, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("network", NETWORKS.values(), ids=NETWORKS.keys())
def test_isotopomer_totals_match_unlabelled_model(network: Network) -> None:
    y0 = {
        k: v * (1.0 + 0.5 * i) for i, (k, v) in enumerate(network.steady_state.items())
    }
    unlabelled = _simulate(network.build(y0))
    isotopomers = _simulate(_isotopomer_model(network, y0))

    for name in network.steady_state:
        column = f"{name}__total" if name in network.label_variables else name
        np.testing.assert_allclose(
            isotopomers[column], unlabelled[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "network", CLOSED_NETWORKS.values(), ids=CLOSED_NETWORKS.keys()
)
def test_isotopomer_model_conserves_labelled_atoms(network: Network) -> None:
    atoms = _labelled_atoms(_simulate(_isotopomer_model(network)), network)
    np.testing.assert_allclose(atoms, atoms.iloc[0], rtol=1e-5)


@pytest.mark.parametrize(
    "network", CLOSED_NETWORKS.values(), ids=CLOSED_NETWORKS.keys()
)
def test_linear_uniform_enrichment_is_fixed_point(network: Network) -> None:
    model = _linear_model(network)
    model.update_variables(dict.fromkeys(model.get_initial_conditions(), 0.3))
    np.testing.assert_allclose(_simulate(model), 0.3, rtol=1e-6)


def test_linear_enrichment_stays_within_bounds() -> None:
    linear = _simulate(_linear_model(NETWORKS["open_chain"]), np.linspace(0, 20, 50))
    assert (linear >= -1e-10).all().all()
    assert (linear <= 1 + 1e-10).all().all()
    np.testing.assert_allclose(linear.iloc[-1], 1.0, atol=1e-4)
