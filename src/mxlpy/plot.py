"""Plotting Utilities Module.

This module provides functions and classes for creating various plots and visualizations
for metabolic models. It includes functionality for plotting heatmaps, time courses,
and parameter scans.

Functions:
    plot_heatmap: Plot a heatmap of the given data.
    plot_time_course: Plot a time course of the given data.
    plot_parameter_scan: Plot a parameter scan of the given data.
    plot_3d_surface: Plot a 3D surface of the given data.
    plot_3d_scatter: Plot a 3D scatter plot of the given data.
    plot_label_distribution: Plot the distribution of labels in the given data.
    plot_linear_label_distribution: Plot the distribution of linear labels in the given
        data.
    plot_label_correlation: Plot the correlation between labels in the given data.
"""

from __future__ import annotations

import contextlib
import itertools as it
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import (
    LogNorm,
    Normalize,
    SymLogNorm,
    colorConverter,  # type: ignore
)
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from wadler_lindig import pformat

from mxlpy.label_map import LabelMapper
from mxlpy.types import Derived

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    from matplotlib.collections import QuadMesh
    from numpy.typing import NDArray

    from mxlpy.linear_label_map import LinearLabelMapper
    from mxlpy.model import Model
    from mxlpy.types import Array, ArrayLike


__all__ = [
    "Axs",
    "Color",
    "FigAx",
    "FigAxs",
    "Linestyle",
    "RGB",
    "RGBA",
    "add_grid",
    "bars",
    "bars_autogrouped",
    "bars_grouped",
    "context",
    "grid_labels",
    "grid_layout",
    "heatmap",
    "heatmap_from_2d_idx",
    "heatmaps_from_2d_idx",
    "line_autogrouped",
    "line_mean_std",
    "lines",
    "lines_grouped",
    "lines_mean_std_from_2d_idx",
    "network",
    "one_axes",
    "relative_label_distribution",
    "reset_prop_cycle",
    "rotate_xlabels",
    "shade_protocol",
    "show",
    "trajectories_2d",
    "two_axes",
    "violins",
    "violins_from_2d_idx",
]


@dataclass
class Axs:
    """Convenience container  axes."""

    axs: NDArray[np.object_]

    def __iter__(self) -> Iterator[Axes]:
        """Get flat axes."""
        yield from cast(list[Axes], self.axs.flatten())

    def __len__(self) -> int:
        """Length of axes."""
        return len(self.axs.flatten())

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @overload
    def __getitem__(self, row_col: int) -> Axes: ...

    @overload
    def __getitem__(self, row_col: slice) -> NDArray[np.object_]: ...

    @overload
    def __getitem__(self, row_col: tuple[int, int]) -> Axes: ...

    @overload
    def __getitem__(self, row_col: tuple[slice, int]) -> NDArray[np.object_]: ...

    @overload
    def __getitem__(self, row_col: tuple[int, slice]) -> NDArray[np.object_]: ...

    def __getitem__(
        self, row_col: int | slice | tuple[int | slice, int | slice]
    ) -> Axes | NDArray[np.object_]:
        """Get Axes or Array of Axes."""
        return cast(Axes, self.axs[row_col])


type FigAx = tuple[Figure, Axes]
type FigAxs = tuple[Figure, Axs]

type Linestyle = Literal[
    "solid",
    "dotted",
    "dashed",
    "dashdot",
]


type RGB = tuple[float, float, float]
type RGBA = tuple[float, float, float, float]
type Color = str | RGB | RGBA


##########################################################################
# Helpers
##########################################################################


def _relative_luminance(color: Array) -> float:
    """Calculate the relative luminance of a color."""
    rgb = colorConverter.to_rgba_array(color)[:, :3]

    # If RsRGB <= 0.03928 then R = RsRGB/12.92 else R = ((RsRGB+0.055)/1.055) ^ 2.4
    rsrgb = np.where(
        rgb <= 0.03928,  # noqa: PLR2004
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.matmul(rsrgb, [0.2126, 0.7152, 0.0722])[0]


def _get_norm(vmin: float, vmax: float) -> Normalize:
    """Get a suitable normalization object for the given data.

    Uses a logarithmic scale for values greater than 1000 or less than -1000,
    a symmetrical logarithmic scale for values less than or equal to 0,
    and a linear scale for all other values.

    Parameters
    ----------
    vmin
        Minimum value of the data.
    vmax
        Maximum value of the data.

    Returns
    -------
    Normalize
        A normalization object for the given data.

    """
    if vmax < 1000 and vmin > -1000:  # noqa: PLR2004
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif vmin <= 0:
        norm = SymLogNorm(linthresh=1, vmin=vmin, vmax=vmax, base=10)
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    return norm


def _norm(df: pd.DataFrame) -> Normalize:
    """Get a normalization object for the given data."""
    vmin = df.min().min()
    vmax = df.max().max()
    return _get_norm(vmin, vmax)


def _norm_with_zero_center(df: pd.DataFrame) -> Normalize:
    """Get a normalization object with zero-centered values for the given data."""
    v = max(abs(df.min().min()), abs(df.max().max()))
    return _get_norm(vmin=-v, vmax=v)


def _partition_by_order_of_magnitude(s: pd.Series) -> list[list[str]]:
    """Partition a series into groups based on the order of magnitude of the values."""
    return [
        i.to_list()
        for i in s.abs()
        .apply(np.log10)
        .apply(np.floor)
        .to_frame(name=0)
        .groupby(0)[0]
        .groups.values()  # type: ignore
    ]


def _combine_small_groups(
    groups: list[list[str]], min_group_size: int
) -> list[list[str]]:
    """Combine smaller groups."""
    result = []
    current_group = groups[0]

    for next_group in groups[1:]:
        if len(current_group) < min_group_size:
            current_group.extend(next_group)
        else:
            result.append(current_group)
            current_group = next_group

    # Last group
    if len(current_group) < min_group_size:
        result[-1].extend(current_group)
    else:
        result.append(current_group)
    return result


def _split_large_groups[T](groups: list[list[T]], max_size: int) -> list[list[T]]:
    """Split groups larger than the given size into smaller groups."""
    return list(
        it.chain(
            *(
                (
                    [group]
                    if len(group) < max_size
                    else [  # type: ignore
                        list(i)
                        for i in np.array_split(group, math.ceil(len(group) / max_size))  # type: ignore
                    ]
                )
                for group in groups
            )
        )
    )  # type: ignore


def _default_color(ax: Axes, color: Color | None) -> Color:
    """Get a default color for the given axis."""
    return f"C{len(ax.lines)}" if color is None else color


def _default_labels(
    ax: Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
) -> None:
    """Set default labels for the given axis.

    Parameters
    ----------
    ax
        matplotlib Axes
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    zlabel
        Label for the z-axis.

    """
    ax.set_xlabel("Add a label / unit" if xlabel is None else xlabel)
    ax.set_ylabel("Add a label / unit" if ylabel is None else ylabel)
    if isinstance(ax, Axes3D):
        ax.set_zlabel("Add a label / unit" if zlabel is None else zlabel)


def _annotate_colormap(
    df: pd.DataFrame,
    ax: Axes,
    sci_annotation_bounds: tuple[float, float],
    annotation_style: str,
    hm: QuadMesh,
) -> None:
    """Annotate a heatmap with the values of the data.

    Parameters
    ----------
    df
        Dataframe to annotate.
    ax
        Axes to annotate.
    sci_annotation_bounds
        Bounds for scientific notation.
    annotation_style
        Style for the annotations.
    hm
        QuadMesh object of the heatmap.

    """
    hm.update_scalarmappable()  # So that get_facecolor is an array
    xpos, ypos = np.meshgrid(
        np.arange(len(df.columns)),
        np.arange(len(df.index)),
    )
    for x, y, val, color in zip(
        xpos.flat,
        ypos.flat,
        hm.get_array().flat,  # type: ignore
        hm.get_facecolor(),
        strict=True,
    ):
        val_text = (
            f"{val:.{annotation_style}}"
            if sci_annotation_bounds[0] < abs(val) <= sci_annotation_bounds[1]
            else f"{val:.0e}"
        )
        ax.text(
            x + 0.5,
            y + 0.5,
            val_text,
            ha="center",
            va="center",
            color="black" if _relative_luminance(color) > 0.45 else "white",  # type: ignore  # noqa: PLR2004
        )


def add_grid(ax: Axes) -> Axes:
    """Add a grid to the given axis.

    Parameters
    ----------
    ax
        Axis to add the grid to.

    Returns
    -------
    Axes
        The axis with the grid added.

    """
    ax.grid(visible=True)
    ax.set_axisbelow(b=True)
    return ax


def grid_labels(
    axs: Axs,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply labels to left and bottom axes.

    Parameters
    ----------
    axs
        Grid of axes to label.
    xlabel
        Label for the x-axis on the bottom row.
    ylabel
        Label for the y-axis on the left column.

    """
    for ax in axs[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)


def rotate_xlabels(
    ax: Axes,
    rotation: float = 45,
    ha: Literal["left", "center", "right"] = "right",
) -> Axes:
    """Rotate the x-axis labels of the given axis.

    Parameters
    ----------
    ax
        Axis to rotate the labels of.
    rotation
        Rotation angle in degrees (default: 45).
    ha
        Horizontal alignment of the labels (default

    Returns
    -------
        Axes object for object chaining

    """
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment(ha)
    return ax


def show(fig: Figure | None = None) -> None:
    """Show the given figure or the current figure.

    Parameters
    ----------
    fig
        Figure to show.

    """
    if fig is None:
        plt.show()
    else:
        fig.show()


def reset_prop_cycle(ax: Axes) -> None:
    """Reset the property cycle of the given axis.

    Parameters
    ----------
    ax
        Axis to reset the property cycle of.

    """
    ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])


@contextlib.contextmanager
def context(
    colors: list[str] | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
    rc: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Context manager to set the defaults for plots.

    Parameters
    ----------
    colors
        colors to use for the plot.
    linewidth
        line width to use for the plot.
    linestyle
        line style to use for the plot.
    rc
        additional keyword arguments to pass to the rc context.

    """
    rc = {} if rc is None else rc

    if colors is not None:
        rc["axes.prop_cycle"] = cycler(color=colors)

    if linewidth is not None:
        rc["lines.linewidth"] = linewidth

    if linestyle is not None:
        rc["lines.linestyle"] = linestyle

    with plt.rc_context(rc):
        yield


##########################################################################
# General plot layout
##########################################################################


def _default_fig_ax(
    *,
    ax: Axes | None,
    grid: bool,
    figsize: tuple[float, float] | None = None,
) -> FigAx:
    """Create a figure and axes if none are provided.

    Parameters
    ----------
    ax
        Axis to use for the plot.
    grid
        Whether to add a grid to the plot.
    figsize
        Size of the figure (default: None).

    Returns
    -------
        Figure and Axes objects for the plot.

    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = cast(Figure, ax.get_figure())

    if grid:
        add_grid(ax)
    return fig, ax


def _default_fig_axs(
    *,
    ncols: int,
    nrows: int,
    figsize: tuple[float, float] | None,
    grid: bool,
    sharex: bool,
    sharey: bool,
) -> FigAxs:
    """Create a figure and multiple axes if none are provided.

    Parameters
    ----------
    axs
        Axes to use for the plot.
    ncols
        Number of columns for the plot.
    nrows
        Number of rows for the plot.
    figsize
        Size of the figure (default: None).
    grid
        Whether to add a grid to the plot.
    sharex
        Whether to share the x-axis between the axes.
    sharey
        Whether to share the y-axis between the axes.

    Returns
    -------
        Figure and Axes objects for the plot.

    """
    fig, axs_array = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )
    axs = Axs(axs_array)

    if grid:
        for ax in axs:
            add_grid(ax)
    return fig, axs


def one_axes(
    *,
    figsize: tuple[float, float] | None = None,
    grid: bool = False,
) -> FigAx:
    """Create a figure with one axis.

    Parameters
    ----------
    figsize
        Size of the figure.
    grid
        Whether to add a grid.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    return _default_fig_ax(
        ax=None,
        grid=grid,
        figsize=figsize,
    )


def two_axes(
    *,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = False,
) -> FigAxs:
    """Create a figure with two axes.

    Parameters
    ----------
    figsize
        Size of the figure.
    sharex
        Whether to share the x-axis between axes.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    return _default_fig_axs(
        ncols=2,
        nrows=1,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )


def grid_layout(
    n_groups: int,
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 2.5,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    """Create a grid layout for the given number of groups.

    Parameters
    ----------
    n_groups
        Number of subplot groups.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    sharex
        Whether to share the x-axis between axes.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid to each axis.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    n_cols = min(n_groups, n_cols)
    n_rows = math.ceil(n_groups / n_cols)
    figsize = (n_cols * col_width, n_rows * row_height)

    fig, axs = _default_fig_axs(
        ncols=n_cols,
        nrows=n_rows,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    # Disable unused plots by default
    axsl = list(axs)
    for i in range(n_groups, len(axs)):
        axsl[i].set_visible(False)
    return fig, axs


##########################################################################
# Plots
##########################################################################


def bars(
    x: pd.Series | pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigAx:
    """Plot bars from a Series or DataFrame.

    Parameters
    ----------
    x
        Data to plot.
    ax
        Axis to plot on. Created if None.
    grid
        Whether to add a grid.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    sns.barplot(data=cast(pd.DataFrame, x), ax=ax)

    if xlabel is None:
        xlabel = x.index.name if x.index.name is not None else ""  # type: ignore
    _default_labels(ax, xlabel=xlabel, ylabel=ylabel)
    if isinstance(x, pd.DataFrame):
        ax.legend(x.columns)
    return fig, ax


def bars_grouped(
    groups: list[pd.DataFrame] | list[pd.Series],
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharey: bool = False,
    grid: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigAxs:
    """Plot multiple groups of bars on separate axes.

    Parameters
    ----------
    groups
        List of DataFrames or Series, one per subplot.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    fig, axs = grid_layout(
        len(groups),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharex=False,
        sharey=sharey,
        grid=grid,
    )

    for group, ax in zip(
        groups,
        axs,
        strict=False,
    ):
        bars(
            group,
            ax=ax,
            grid=grid,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    return fig, axs


def bars_autogrouped(
    s: pd.Series | pd.DataFrame,
    *,
    n_cols: int = 2,
    col_width: float = 4,
    row_height: float = 3,
    min_group_size: int = 1,
    max_group_size: int = 6,
    grid: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigAxs:
    """Plot bars from a Series or DataFrame grouped by order of magnitude.

    Parameters
    ----------
    s
        Data to plot.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    min_group_size
        Minimum number of items per group.
    max_group_size
        Maximum number of items per group.
    grid
        Whether to add a grid.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    group_names = (
        _partition_by_order_of_magnitude(s)
        if isinstance(s, pd.Series)
        else _partition_by_order_of_magnitude(s.max())
    )
    group_names = _combine_small_groups(group_names, min_group_size=min_group_size)
    group_names = _split_large_groups(group_names, max_size=max_group_size)

    groups: list[pd.Series] | list[pd.DataFrame] = (
        [s.loc[group] for group in group_names]
        if isinstance(s, pd.Series)
        else [s.loc[:, group] for group in group_names]
    )

    return bars_grouped(
        groups,
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        grid=grid,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def lines(
    x: pd.DataFrame | pd.Series,
    *,
    ax: Axes | None = None,
    alpha: float = 1.0,
    color: Color | list[Color] | None = None,
    grid: bool = True,
    legend: bool = True,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> FigAx:
    """Plot multiple lines on the same axis.

    Parameters
    ----------
    x
        Data to plot.
    ax
        Axis to plot on. Created if None.
    alpha
        Opacity of the lines.
    color
        Color or list of colors for the lines.
    grid
        Whether to add a grid.
    legend
        Whether to show a legend.
    linewidth
        Width of the lines.
    linestyle
        Style of the lines.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    _lines = ax.plot(
        x.index,
        x,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
    )
    _default_labels(
        ax,
        xlabel=x.index.name if xlabel is None else xlabel,  # type: ignore
        ylabel=ylabel,
    )
    if legend:
        names = x.columns if isinstance(x, pd.DataFrame) else [str(x.name)]
        for line, name in zip(_lines, names, strict=True):
            line.set_label(name)
        ax.legend()
    return fig, ax


def _repeat_color_if_necessary(
    color: list[list[Color]] | Color | None, n: int
) -> Iterable[list[Color] | Color | None]:
    return [color] * n if not isinstance(color, list) else color


def lines_grouped(
    groups: list[pd.DataFrame] | list[pd.Series],
    *,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 4,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: Color | list[list[Color]] | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
) -> FigAxs:
    """Plot multiple groups of lines on separate axes.

    Parameters
    ----------
    groups
        List of DataFrames or Series, one per subplot.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    sharex
        Whether to share the x-axis between axes.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    color
        Color or nested list of colors per group.
    linewidth
        Width of the lines.
    linestyle
        Style of the lines.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    fig, axs = grid_layout(
        len(groups),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    for group, ax, color_ in zip(
        groups,
        axs,
        _repeat_color_if_necessary(color, n=len(groups)),
        strict=False,
    ):
        lines(
            group,
            ax=ax,
            grid=grid,
            color=color_,
            linewidth=linewidth,
            linestyle=linestyle,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    return fig, axs


def line_autogrouped(
    s: pd.Series | pd.DataFrame,
    *,
    n_cols: int = 2,
    col_width: float = 4,
    row_height: float = 3,
    min_group_size: int = 1,
    max_group_size: int = 6,
    grid: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: Color | list[list[Color]] | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
) -> FigAxs:
    """Plot lines from a Series or DataFrame grouped by order of magnitude.

    Parameters
    ----------
    s
        Data to plot.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    min_group_size
        Minimum number of items per group.
    max_group_size
        Maximum number of items per group.
    grid
        Whether to add a grid.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    color
        Color or nested list of colors per group.
    linewidth
        Width of the lines.
    linestyle
        Style of the lines.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    group_names = (
        _partition_by_order_of_magnitude(s)
        if isinstance(s, pd.Series)
        else _partition_by_order_of_magnitude(s.max())
    )
    group_names = _combine_small_groups(group_names, min_group_size=min_group_size)
    group_names = _split_large_groups(group_names, max_size=max_group_size)

    groups: list[pd.Series] | list[pd.DataFrame] = (
        [s.loc[group] for group in group_names]
        if isinstance(s, pd.Series)
        else [s.loc[:, group] for group in group_names]
    )

    return lines_grouped(
        groups,
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        grid=grid,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def line_mean_std(
    df: pd.DataFrame,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    color: Color | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
    alpha: float = 0.2,
    grid: bool = True,
) -> FigAx:
    """Plot the mean and standard deviation using a line and fill.

    Parameters
    ----------
    df
        DataFrame where rows are time points and columns are replicates.
    label
        Label for the line in the legend.
    ax
        Axis to plot on. Created if None.
    color
        Color of the line and fill.
    linewidth
        Width of the mean line.
    linestyle
        Style of the mean line.
    alpha
        Opacity of the standard deviation fill.
    grid
        Whether to add a grid.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    color = _default_color(ax=ax, color=color)

    mean = df.mean(axis=1)
    std = df.std(axis=1)
    ax.plot(
        mean.index,
        mean,
        color=color,
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.fill_between(
        df.index,
        mean - std,
        mean + std,
        color=color,
        alpha=alpha,
    )
    _default_labels(
        ax,
        xlabel=df.index.name,  # type: ignore
        ylabel=None,
    )
    return fig, ax


def lines_mean_std_from_2d_idx(
    df: pd.DataFrame,
    *,
    names: list[str] | None = None,
    ax: Axes | None = None,
    alpha: float = 0.2,
    grid: bool = True,
    color: Color | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
) -> FigAx:
    """Plot the mean and standard deviation of a 2D indexed dataframe.

    Parameters
    ----------
    df
        DataFrame with a two-level MultiIndex.
    names
        Column names to plot. Defaults to all columns.
    ax
        Axis to plot on. Created if None.
    alpha
        Opacity of the standard deviation fill.
    grid
        Whether to add a grid.
    color
        Color of the lines.
    linewidth
        Width of the lines.
    linestyle
        Style of the lines.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, ax = _default_fig_ax(ax=ax, grid=grid)

    for name in df.columns if names is None else names:
        line_mean_std(
            df[name].unstack().T,
            label=name,
            alpha=alpha,
            ax=ax,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    ax.legend()
    return fig, ax


def _create_heatmap(
    *,
    ax: Axes | None,
    df: pd.DataFrame,
    title: str | None = None,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    yticklabels: list[str],
    norm: Normalize,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = True,
    cmap: str = "RdBu_r",
    cax: Axes | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> tuple[Figure, Axes, QuadMesh]:
    fig, ax = _default_fig_ax(
        ax=ax,
        figsize=(
            max(4, 0.5 * len(df.columns)),
            max(4, 0.5 * len(df.index)),
        ),
        grid=False,
    )

    # Note: pcolormesh swaps index/columns
    hm = ax.pcolormesh(df, norm=norm, cmap=cmap)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(
        np.arange(0, len(df.columns), 1, dtype=float) + 0.5,  # type: ignore
        labels=xticklabels,
    )
    ax.set_yticks(
        np.arange(0, len(df.index), 1, dtype=float) + 0.5,  # type: ignore
        labels=yticklabels,
    )

    if annotate:
        _annotate_colormap(df, ax, sci_annotation_bounds, annotation_style, hm)

    if colorbar:
        # Add a colorbar
        cb = fig.colorbar(hm, cax, ax)
        cb.outline.set_linewidth(0)  # type: ignore

    if invert_yaxis:
        ax.invert_yaxis()
    rotate_xlabels(ax, rotation=45, ha="right")
    return fig, ax, hm


def heatmap(
    df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = True,
    cmap: str = "RdBu_r",
    norm: Normalize | None = None,
    cax: Axes | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot a heatmap of the given data.

    Parameters
    ----------
    df
        DataFrame to plot as a heatmap.
    ax
        Axis to plot on. Created if None.
    title
        Title of the heatmap.
    annotate
        Whether to annotate cells with values.
    colorbar
        Whether to add a colorbar.
    invert_yaxis
        Whether to invert the y-axis.
    cmap
        Colormap name.
    norm
        Normalization for the colormap.
    cax
        Axis for the colorbar.
    sci_annotation_bounds
        Bounds for switching to scientific notation in annotations.
    annotation_style
        Format string for cell annotations.

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
        Figure, Axes, and QuadMesh objects.

    """
    return _create_heatmap(
        ax=ax,
        df=df,
        title=title,
        xlabel=df.index.name,  # type: ignore
        ylabel=df.columns.name,  # type: ignore
        xticklabels=cast(list, df.columns),
        yticklabels=cast(list, df.index),
        annotate=annotate,
        colorbar=colorbar,
        invert_yaxis=invert_yaxis,
        cmap=cmap,
        norm=_norm_with_zero_center(df) if norm is None else norm,
        cax=cax,
        sci_annotation_bounds=sci_annotation_bounds,
        annotation_style=annotation_style,
    )


def heatmap_from_2d_idx(
    df: pd.DataFrame,
    variable: str,
    *,
    ax: Axes | None = None,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = False,
    cmap: str = "viridis",
    norm: Normalize | None = None,
    cax: Axes | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot a heatmap of a 2D indexed dataframe.

    Parameters
    ----------
    df
        DataFrame with a two-level MultiIndex.
    variable
        Column name to plot.
    ax
        Axis to plot on. Created if None.
    annotate
        Whether to annotate cells with values.
    colorbar
        Whether to add a colorbar.
    invert_yaxis
        Whether to invert the y-axis.
    cmap
        Colormap name.
    norm
        Normalization for the colormap.
    cax
        Axis for the colorbar.
    sci_annotation_bounds
        Bounds for switching to scientific notation in annotations.
    annotation_style
        Format string for cell annotations.

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
        Figure, Axes, and QuadMesh objects.

    """
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)
    df2d = df[variable].unstack().T

    return _create_heatmap(
        df=df2d,
        xlabel=df2d.index.name,  # type: ignore
        ylabel=df2d.columns.name,  # type: ignore
        xticklabels=[f"{i:.2f}" for i in df2d.columns],
        yticklabels=[f"{i:.2f}" for i in df2d.index],
        ax=ax,
        cax=cax,
        annotate=annotate,
        colorbar=colorbar,
        invert_yaxis=invert_yaxis,
        cmap=cmap,
        norm=_norm(df2d) if norm is None else norm,
        sci_annotation_bounds=sci_annotation_bounds,
        annotation_style=annotation_style,
    )


def heatmaps_from_2d_idx(
    df: pd.DataFrame,
    *,
    n_cols: int = 3,
    col_width_factor: float = 1,
    row_height_factor: float = 0.6,
    sharex: bool = True,
    sharey: bool = False,
    annotate: bool = False,
    colorbar: bool = True,
    invert_yaxis: bool = False,
    cmap: str = "viridis",
    norm: Normalize | None = None,
    sci_annotation_bounds: tuple[float, float] = (0.01, 100),
    annotation_style: str = "2g",
) -> FigAxs:
    """Plot multiple heatmaps of a 2D indexed dataframe.

    Parameters
    ----------
    df
        DataFrame with a two-level MultiIndex.
    n_cols
        Number of columns in the grid.
    col_width_factor
        Multiplier for column width based on index level size.
    row_height_factor
        Multiplier for row height based on index level size.
    sharex
        Whether to share the x-axis between axes.
    sharey
        Whether to share the y-axis between axes.
    annotate
        Whether to annotate cells with values.
    colorbar
        Whether to add a colorbar.
    invert_yaxis
        Whether to invert the y-axis.
    cmap
        Colormap name.
    norm
        Normalization for the colormap.
    sci_annotation_bounds
        Bounds for switching to scientific notation in annotations.
    annotation_style
        Format string for cell annotations.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    idx = cast(pd.MultiIndex, df.index)

    fig, axs = grid_layout(
        n_groups=len(df.columns),
        n_cols=min(n_cols, len(df)),
        col_width=len(idx.levels[0]) * col_width_factor,
        row_height=len(idx.levels[1]) * row_height_factor,
        sharex=sharex,
        sharey=sharey,
        grid=False,
    )
    for ax, var in zip(axs, df.columns, strict=False):
        heatmap_from_2d_idx(
            df,
            var,
            ax=ax,
            annotate=annotate,
            colorbar=colorbar,
            invert_yaxis=invert_yaxis,
            cmap=cmap,
            norm=norm,
            sci_annotation_bounds=sci_annotation_bounds,
            annotation_style=annotation_style,
        )
    return fig, axs


def violins(
    df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    grid: bool = True,
) -> FigAx:
    """Plot multiple violins on the same axis.

    Parameters
    ----------
    df
        DataFrame to plot.
    ax
        Axis to plot on. Created if None.
    grid
        Whether to add a grid.

    Returns
    -------
    FigAx
        Figure and Axes objects.

    """
    fig, ax = _default_fig_ax(ax=ax, grid=grid)
    sns.violinplot(df, ax=ax)
    _default_labels(ax=ax, xlabel="", ylabel=None)
    return fig, ax


def violins_from_2d_idx(
    df: pd.DataFrame,
    *,
    n_cols: int = 4,
    row_height: int = 2,
    sharex: bool = True,
    sharey: bool = False,
    grid: bool = True,
) -> FigAxs:
    """Plot multiple violins of a 2D indexed dataframe.

    Parameters
    ----------
    df
        DataFrame with a two-level MultiIndex.
    n_cols
        Number of columns in the grid.
    row_height
        Height of each row in inches.
    sharex
        Whether to share the x-axis between axes.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    if len(cast(pd.MultiIndex, df.index).levels) != 2:  # noqa: PLR2004
        msg = "MultiIndex must have exactly two levels"
        raise ValueError(msg)

    fig, axs = grid_layout(
        len(df.columns),
        n_cols=n_cols,
        row_height=row_height,
        sharex=sharex,
        sharey=sharey,
        grid=grid,
    )

    for ax, col in zip(axs[: len(df.columns)].flatten(), df.columns, strict=True):
        ax.set_title(col)
        violins(df[col].unstack(), ax=ax)

    for ax in axs[len(df.columns) :]:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(0)
        ax.yaxis.set_ticks([])

    for ax in axs:
        rotate_xlabels(ax)
    return fig, axs


def shade_protocol(
    protocol: pd.Series,
    *,
    ax: Axes,
    cmap_name: str = "Greys_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.5,
    add_legend: bool = True,
) -> None:
    """Shade the given protocol on the given axis.

    Parameters
    ----------
    protocol
        Series with time deltas as index and protocol values.
    ax
        Axis to shade on.
    cmap_name
        Name of the colormap to use.
    vmin
        Minimum value for colormap normalization.
    vmax
        Maximum value for colormap normalization.
    alpha
        Opacity of the shading.
    add_legend
        Whether to add a legend.

    """
    cmap = colormaps[cmap_name]
    norm = Normalize(
        vmin=protocol.min() if vmin is None else vmin,
        vmax=protocol.max() if vmax is None else vmax,
    )

    t0 = pd.Timedelta(seconds=0)
    for t_end, val in protocol.items():
        t_end = cast(pd.Timedelta, t_end)
        ax.axvspan(
            t0.total_seconds(),
            t_end.total_seconds(),
            facecolor=cmap(norm(val)),
            edgecolor=None,
            alpha=alpha,
        )
        t0 = t_end  # type: ignore

    if add_legend:
        ax.add_artist(
            Legend(
                ax,
                handles=[
                    Patch(
                        facecolor=cmap(norm(val)),
                        alpha=alpha,
                        label=val,
                    )  # type: ignore
                    for val in protocol
                ],
                labels=protocol,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.0),
                title="protocol" if protocol.name is None else cast(str, protocol.name),
            )
        )


##########################################################################
# Plots that actually require a model :/
##########################################################################


def trajectories_2d(
    model: Model,
    x1: tuple[str, ArrayLike],
    x2: tuple[str, ArrayLike],
    y0: dict[str, float] | None = None,
    ax: Axes | None = None,
) -> FigAx:
    """Plot trajectories of two variables in a 2D phase space.

    Examples
    --------
        >>> trajectories_2d(
        ...     model,
        ...     ("S", np.linspace(0, 1, 10)),
        ...     ("P", np.linspace(0, 1, 10)),
        ... )

    Parameters
    ----------
    model
        Model to use for the plot.
    x1
        Tuple of the first variable name and its values.
    x2
        Tuple of the second variable name and its values.
    y0
        Initial conditions for the model.
    ax
        Axes to use for the plot.

    """
    name1, values1 = x1
    name2, values2 = x2
    n1 = len(values1)
    n2 = len(values2)
    u = np.zeros((n1, n2))
    v = np.zeros((n1, n2))
    y0 = model.get_initial_conditions() if y0 is None else y0
    for i, ii in enumerate(values1):
        for j, jj in enumerate(values2):
            rhs = model.get_right_hand_side(y0 | {name1: ii, name2: jj})
            u[i, j] = rhs[name1]
            v[i, j] = rhs[name2]

    fig, ax = _default_fig_ax(ax=ax, grid=False)
    ax.quiver(values1, values2, u.T, v.T)
    return fig, ax


##########################################################################
# Label Plots
##########################################################################


def relative_label_distribution(
    mapper: LabelMapper | LinearLabelMapper,
    concs: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    n_cols: int = 2,
    col_width: float = 3,
    row_height: float = 3,
    sharey: bool = False,
    grid: bool = True,
    color: Color | None = None,
    linewidth: float | None = None,
    linestyle: Linestyle | None = None,
) -> FigAxs:
    """Plot the relative distribution of labels in the given data.

    Parameters
    ----------
    mapper
        Label mapper defining isotopomer structure.
    concs
        DataFrame of concentration time courses.
    subset
        Subset of label variables to plot. Defaults to all.
    n_cols
        Number of columns in the grid.
    col_width
        Width of each column in inches.
    row_height
        Height of each row in inches.
    sharey
        Whether to share the y-axis between axes.
    grid
        Whether to add a grid.
    color
        Color of the lines.
    linewidth
        Width of the lines.
    linestyle
        Style of the lines.

    Returns
    -------
    FigAxs
        Figure and Axs objects.

    """
    variables = list(mapper.label_variables) if subset is None else subset
    fig, axs = grid_layout(
        n_groups=len(variables),
        n_cols=n_cols,
        col_width=col_width,
        row_height=row_height,
        sharey=sharey,
        grid=grid,
    )
    # FIXME: rewrite as building a dict of dataframes
    # and passing it to lines_grouped
    if isinstance(mapper, LabelMapper):
        for ax, name in zip(axs, variables, strict=False):
            for i in range(mapper.label_variables[name]):
                isos = mapper.get_isotopomers_of_at_position(name, i)
                labels = cast(pd.DataFrame, concs.loc[:, isos])
                total = concs.loc[:, f"{name}__total"]
                ax.plot(
                    labels.index,
                    (labels.sum(axis=1) / total),
                    label=f"C{i + 1}",
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                )
            ax.set_title(name)
            ax.legend()
    else:
        for ax, (name, isos) in zip(
            axs,
            mapper.get_isotopomers(variables).items(),
            strict=False,
        ):
            ax.plot(
                concs.index,
                concs.loc[:, isos],
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
            )
            ax.set_title(name)
            ax.legend([f"C{i + 1}" for i in range(len(isos))])

    return fig, axs


def network(
    model: Model,
    *,
    node_size: float = 500,
    cofactors: list[str] | None = None,
    layout: str = "kamada_kawai",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot reaction network topology as a directed graph.

    Species are shown as circles, reactions as squares. Stoichiometric
    edges are solid arrows; modifier edges (args not in stoichiometry)
    are dashed. Cofactor species are rendered as per-reaction copies
    to avoid visual clutter from high-degree hub nodes.

    Parameters
    ----------
    model
        Model whose topology to visualise.
    cofactors
        Species to render as per-reaction copies rather than as a
        shared hub. Useful for ubiquitous metabolites like ATP or NADH.
    layout
        networkx layout algorithm name, e.g. ``"kamada_kawai"``,
        ``"spring"``, ``"circular"``.
    ax
        Axes to draw on. If None, a new figure is created.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes containing the network diagram.

    """
    cofactor_set = set(cofactors or [])
    args: dict[str, float] = {
        **model.get_parameter_values(),
        **model.get_initial_conditions(),
    }

    G: nx.DiGraph = nx.DiGraph()
    species_nodes: list[str] = []
    reaction_nodes: list[str] = []
    regular_edges: list[tuple[str, str]] = []
    modifier_edges: list[tuple[str, str]] = []
    labels: dict[str, str] = {}

    for name in model.get_variable_names():
        if name not in cofactor_set:
            G.add_node(name)
            species_nodes.append(name)
            labels[name] = name

    for rxn_name, rxn in model.get_raw_reactions().items():
        G.add_node(rxn_name)
        reaction_nodes.append(rxn_name)
        labels[rxn_name] = rxn_name

        for species, stoich in rxn.stoichiometry.items():
            try:
                coeff: float = (
                    stoich.calculate(args)
                    if isinstance(stoich, Derived)
                    else float(stoich)
                )
            except (KeyError, TypeError):
                coeff = 1.0

            if species in cofactor_set:
                node_id = f"{species}__{rxn_name}"
                if node_id not in G:
                    G.add_node(node_id)
                    species_nodes.append(node_id)
                    labels[node_id] = species
            else:
                node_id = species

            if coeff < 0:
                regular_edges.append((node_id, rxn_name))
            else:
                regular_edges.append((rxn_name, node_id))

        for mod in rxn.get_modifiers(model):
            if mod in cofactor_set:
                node_id = f"{mod}__{rxn_name}"
                if node_id not in G:
                    G.add_node(node_id)
                    species_nodes.append(node_id)
                    labels[node_id] = mod
            else:
                node_id = mod
            modifier_edges.append((node_id, rxn_name))

    G.add_edges_from(regular_edges)
    G.add_edges_from(modifier_edges)

    layout_fn = getattr(nx, f"{layout}_layout")
    pos = layout_fn(G)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = cast(Figure, ax.get_figure())

    ax.set_axis_off()

    if species_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=species_nodes,
            node_shape="o",
            node_color="white",
            edgecolors="black",
            ax=ax,
            node_size=node_size,
        )
    if reaction_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=reaction_nodes,
            node_shape="s",
            node_color="lightgray",
            edgecolors="black",
            ax=ax,
            node_size=node_size,
        )
    if regular_edges:
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax, arrows=True)
    if modifier_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=modifier_edges, style="dashed", ax=ax, arrows=True
        )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

    return fig, ax
