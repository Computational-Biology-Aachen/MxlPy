"""Docstring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pandas as pd
from wadler_lindig import pformat

from mxlpy import plot
from mxlpy.simulator import Simulator
from mxlpy.types import Result, unwrap

if TYPE_CHECKING:
    from mxlpy.model import Model
    from mxlpy.types import ArrayLike

__all__ = [
    "ProtocolComparison",
    "SteadyStateComparison",
    "TimeCourseComparison",
    "protocol_time_courses",
    "steady_states",
    "time_courses",
]


@dataclass
class SteadyStateComparison:
    """Compare two steady states."""

    res1: Result
    res2: Result

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def variables(self) -> pd.DataFrame:
        """Compare the steady state variables."""
        ss1 = self.res1.get_variables().iloc[-1]
        ss2 = self.res2.get_variables().iloc[-1]
        diff = ss2 - ss1
        return pd.DataFrame(
            {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Compare the steady state fluxes."""
        ss1 = self.res1.get_fluxes().iloc[-1]
        ss2 = self.res2.get_fluxes().iloc[-1]
        diff = ss2 - ss1
        return pd.DataFrame(
            {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
        )

    @property
    def all(self) -> pd.DataFrame:
        """Compare both steady-state variables and fluxes."""
        ss1 = self.res1.get_combined().iloc[-1]
        ss2 = self.res2.get_combined().iloc[-1]
        diff = ss2 - ss1
        return pd.DataFrame(
            {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
        )

    def plot_variables(self, title: str = "Variables") -> plot.FigAxs:
        """Plot the relative difference of steady-state variables."""
        fig, axs = plot.bars_autogrouped(self.variables["rel_diff"], ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle(title)
        return fig, axs

    def plot_fluxes(self, title: str = "Fluxes") -> plot.FigAxs:
        """Plot the relative difference of steady-state fluxes."""
        fig, axs = plot.bars_autogrouped(self.fluxes["rel_diff"], ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle(title)
        return fig, axs

    def plot_all(self, title: str = "Variables and Fluxes") -> plot.FigAxs:
        """Plot the relative difference of steady-state variables and fluxes."""
        combined = self.all

        fig, axs = plot.bars_autogrouped(combined["rel_diff"], ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle(title)
        return fig, axs


@dataclass
class TimeCourseComparison:
    """Compare two time courses."""

    res1: Result
    res2: Result

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    # @property
    # def variables(self) -> pd.DataFrame:
    #     """Compare the steady state variables."""
    #     ss1 = self.res1.get_variables()
    #     ss2 = self.res2.get_variables()
    #     diff = ss2 - ss1
    #     return pd.DataFrame(
    #         {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
    #     )

    # @property
    # def fluxes(self) -> pd.DataFrame:
    #     """Compare the steady state fluxes."""
    #     ss1 = self.res1.get_fluxes()
    #     ss2 = self.res2.get_fluxes()
    #     diff = ss2 - ss1
    #     return pd.DataFrame(
    #         {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
    #     )

    def plot_variables_relative_difference(self) -> plot.FigAxs:
        """Plot the relative difference of time course variables."""
        c1 = self.res1.variables
        c2 = self.res2.variables

        rel_diff = ((c2.loc[:, cast(list[str], c1.columns)] - c1) / c1).fillna(0)
        fig, axs = plot.line_autogrouped(rel_diff, ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle("Variables")
        return fig, axs

    def plot_fluxes_relative_difference(self) -> plot.FigAxs:
        """Plot the relative difference of time course fluxes."""
        v1 = self.res1.fluxes
        v2 = self.res2.fluxes

        rel_diff = ((v2.loc[:, cast(list[str], v1.columns)] - v1) / v1).fillna(0)
        fig, axs = plot.line_autogrouped(rel_diff, ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle("Fluxes")
        return fig, axs


@dataclass
class ProtocolComparison:
    """Compare two protocol time courses."""

    res1: Result
    res2: Result
    protocol: pd.DataFrame

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    # @property
    # def variables(self) -> pd.DataFrame:
    #     """Compare the steady state variables."""
    #     ss1 = self.res1.get_variables()
    #     ss2 = self.res2.get_variables()
    #     diff = ss2 - ss1
    #     return pd.DataFrame(
    #         {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
    #     )

    # @property
    # def fluxes(self) -> pd.DataFrame:
    #     """Compare the steady state fluxes."""
    #     ss1 = self.res1.get_fluxes()
    #     ss2 = self.res2.get_fluxes()
    #     diff = ss2 - ss1
    #     return pd.DataFrame(
    #         {"m1": ss1, "m2": ss2, "diff": diff, "rel_diff": diff / ss1}
    #     )

    def plot_variables_relative_difference(
        self,
        shade_protocol_variable: str | None = None,
    ) -> plot.FigAxs:
        """Plot the relative difference of time course variables."""
        c1 = self.res1.variables
        c2 = self.res2.variables

        rel_diff = ((c2.loc[:, cast(list[str], c1.columns)] - c1) / c1).fillna(0)
        fig, axs = plot.line_autogrouped(rel_diff, ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle("Variables")

        if shade_protocol_variable is not None:
            protocol = self.protocol[shade_protocol_variable]
            for ax in axs:
                plot.shade_protocol(protocol=protocol, ax=ax)
        return fig, axs

    def plot_fluxes_relative_difference(
        self,
        shade_protocol_variable: str | None = None,
    ) -> plot.FigAxs:
        """Plot the relative difference of time course fluxes."""
        v1 = self.res1.fluxes
        v2 = self.res2.fluxes

        rel_diff = ((v2.loc[:, cast(list[str], v1.columns)] - v1) / v1).fillna(0)
        fig, axs = plot.line_autogrouped(rel_diff, ylabel="")
        plot.grid_labels(axs, ylabel="Relative difference")
        fig.suptitle("Fluxes")

        if shade_protocol_variable is not None:
            protocol = self.protocol[shade_protocol_variable]
            for ax in axs:
                plot.shade_protocol(protocol=protocol, ax=ax)
        return fig, axs


def steady_states(m1: Model, m2: Model) -> SteadyStateComparison:
    """Compare the steady states of two models."""
    return SteadyStateComparison(
        res1=unwrap(Simulator(m1).simulate_to_steady_state().get_result()),
        res2=unwrap(Simulator(m2).simulate_to_steady_state().get_result()),
    )


def time_courses(m1: Model, m2: Model, time_points: ArrayLike) -> TimeCourseComparison:
    """Compare the time courses of two models."""
    return TimeCourseComparison(
        res1=unwrap(
            Simulator(m1).simulate_time_course(time_points=time_points).get_result()
        ),
        res2=unwrap(
            Simulator(m2).simulate_time_course(time_points=time_points).get_result()
        ),
    )


def protocol_time_courses(
    m1: Model,
    m2: Model,
    protocol: pd.DataFrame,
) -> ProtocolComparison:
    """Compare the time courses of two models."""
    return ProtocolComparison(
        res1=unwrap(Simulator(m1).simulate_protocol(protocol=protocol).get_result()),
        res2=unwrap(Simulator(m2).simulate_protocol(protocol=protocol).get_result()),
        protocol=protocol,
    )
