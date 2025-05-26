from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mxlpy.plot import (
    add_grid,
    bars,
    grid_layout,
    heatmap,
    heatmap_from_2d_idx,
    heatmaps_from_2d_idx,
    line_autogrouped,
    line_mean_std,
    lines,
    lines_grouped,
    lines_mean_std_from_2d_idx,
    rotate_xlabels,
    two_axes,
    violins,
    violins_from_2d_idx,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 3.0, 4.0, 5.0, 6.0],
            "C": [3.0, 4.0, 5.0, 6.0, 7.0],
        },
        index=pd.Index([0.0, 1.0, 2.0, 3.0, 4.0], name="time"),
    )


@pytest.fixture
def sample_series() -> pd.Series:
    return pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.Index([0.0, 1.0, 2.0, 3.0, 4.0], name="time"),
        name="value",
    )


@pytest.fixture
def multiindex_dataframe() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            [0.1, 0.2, 0.3],  # First level
            [0.0, 1.0, 2.0, 3.0],  # Second level
        ],
        names=["parameter", "time"],
    )
    rng = np.random.default_rng()
    return pd.DataFrame(
        {
            "A": rng.random(12),
            "B": rng.random(12),
            "C": rng.random(12),
        },
        index=index,
    )


def test_add_grid() -> None:
    fig, ax = plt.subplots()
    result = add_grid(ax)
    assert isinstance(result, Axes)
    # Check if grid lines are visible instead of using internal attributes
    assert ax.xaxis.get_gridlines()[0].get_visible()
    plt.close(fig)


def test_rotate_xlabels() -> None:
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["A", "B", "C"])
    result = rotate_xlabels(ax, rotation=45, ha="right")
    assert isinstance(result, Axes)
    for label in ax.get_xticklabels():
        assert label.get_rotation() == 45
        assert label.get_horizontalalignment() == "right"
    plt.close(fig)


def test_two_axes() -> None:
    fig, axs = two_axes()
    assert len(axs) == 2
    for ax in axs:
        assert isinstance(ax, Axes)
    plt.close(fig)


def test_grid_layout() -> None:
    fig, axs = grid_layout(5)
    assert isinstance(fig, Figure)
    assert len(axs) >= 5
    for ax in axs:
        assert isinstance(ax, Axes)
    plt.close(fig)


def test_bars(sample_dataframe: pd.DataFrame) -> None:
    fig, ax = bars(sample_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    # SeabornBarPlot no longer creates one patch per data point, just check some patches exist
    assert len(ax.patches) > 0
    plt.close(fig)


def test_lines_with_dataframe(sample_dataframe: pd.DataFrame) -> None:
    fig, ax = lines(sample_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == len(sample_dataframe.columns)
    plt.close(fig)


def test_lines_with_series(sample_series: pd.Series) -> None:
    fig, ax = lines(sample_series)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1
    plt.close(fig)


def test_lines_grouped(sample_dataframe: pd.DataFrame) -> None:
    dfs = [sample_dataframe[col] for col in sample_dataframe.columns]
    fig, axs = lines_grouped(dfs)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_line_autogrouped(sample_dataframe: pd.DataFrame) -> None:
    fig, axs = line_autogrouped(sample_dataframe)
    assert isinstance(fig, Figure)
    for ax in axs:
        assert isinstance(ax, Axes)
    plt.close(fig)


def test_line_mean_std(sample_dataframe: pd.DataFrame) -> None:
    fig, ax = line_mean_std(sample_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1  # For the fill_between
    plt.close(fig)


def test_lines_mean_std_from_2d_idx(multiindex_dataframe: pd.DataFrame) -> None:
    fig, ax = lines_mean_std_from_2d_idx(multiindex_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == len(multiindex_dataframe.columns)
    assert len(ax.collections) == len(
        multiindex_dataframe.columns
    )  # For the fill_between
    plt.close(fig)


def test_heatmap(sample_dataframe: pd.DataFrame) -> None:
    fig, ax, hm = heatmap(sample_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.collections[0] == hm
    plt.close(fig)


def test_heatmap_annotated(sample_dataframe: pd.DataFrame) -> None:
    fig, ax, hm = heatmap(sample_dataframe, annotate=True)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.collections[0] == hm
    assert len(ax.texts) == len(sample_dataframe.index) * len(sample_dataframe.columns)
    plt.close(fig)


def test_heatmap_from_2d_idx(multiindex_dataframe: pd.DataFrame) -> None:
    fig, ax, qm = heatmap_from_2d_idx(multiindex_dataframe, "A")
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) >= 1  # For the heatmap
    plt.close(fig)


def test_heatmaps_from_2d_idx(multiindex_dataframe: pd.DataFrame) -> None:
    fig, axs = heatmaps_from_2d_idx(multiindex_dataframe)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_violins(sample_dataframe: pd.DataFrame) -> None:
    fig, ax = violins(sample_dataframe)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0  # For the violins
    plt.close(fig)


def test_violins_from_2d_idx(multiindex_dataframe: pd.DataFrame) -> None:
    fig, axs = violins_from_2d_idx(multiindex_dataframe)
    assert isinstance(fig, Figure)
    plt.close(fig)
