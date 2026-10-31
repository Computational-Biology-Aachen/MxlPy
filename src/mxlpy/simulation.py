# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self, overload

import numpy as np
import pandas as pd
from wadler_lindig import pformat

__all__ = ["Evaluation", "OdeSimulation", "Simulation"]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy._steady_state_builder import SteadyStateModelBuilder
    from mxlpy.types import Array, ArrayLike


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    """Normalize split results by a given factor or array.

    Parameters
    ----------
    results
        List of DataFrames containing the results to normalize.
    normalise
        Normalization factor or array.

    Returns
    -------
    list[pd.DataFrame]
        List of normalized DataFrames.

    """
    if isinstance(normalise, int | float):
        return [i / normalise for i in results]
    if len(normalise) == len(results):
        return [(i.T / j).T for i, j in zip(results, normalise, strict=True)]

    results = []
    start = 0
    end = 0
    for i in results:
        end += len(i)
        results.append(i / np.reshape(normalise[start:end], (len(i), 1)))  # type: ignore
        start += end
    return results


@dataclass(kw_only=True, slots=True)
class _BaseSimulation[M: (KineticModelBuilder, OdeModelBuilder)]:
    """Simulation results shared by all integrable model builders."""

    model: M
    raw_variables: list[pd.DataFrame]
    raw_parameters: list[dict[str, float]]
    raw_args: list[pd.DataFrame] = field(default_factory=list)
    # A faster way to compute outputs, offered by the integrator if it has one
    args_fn: Callable | None = field(default=None, repr=False)

    def _args_for(
        self,
        variables: pd.DataFrame,
        python: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        if (
            self.args_fn is not None
            and (args := self.args_fn(self.model, variables, python)) is not None
        ):
            return args
        return python(variables)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @classmethod
    def default(cls, model: M, time_points: Array) -> Self:
        """Get result filled with NaNs.

        Parameters
        ----------
        model
            Model to create default results for.
        time_points
            Time points for the simulation index.

        """
        return cls(
            model=model,
            raw_variables=[
                pd.DataFrame(
                    data=np.full(
                        shape=(len(time_points), len(model.get_variable_names())),
                        fill_value=np.nan,
                    ),
                    index=time_points,
                    columns=model.get_variable_names(),
                )
            ],
            raw_parameters=[model.get_parameter_values()],
        )

    def _adjust_data(
        self,
        data: list[pd.DataFrame],
        normalise: float | ArrayLike | None = None,
        *,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        if normalise is not None:
            data = _normalise_split_results(data, normalise=normalise)
        if concatenated:
            return pd.concat(data, axis=0)
        return data

    def get_new_y0(self) -> dict[str, float]:
        """Get the new initial conditions after the simulation.

        Examples
        --------
            >>> Simulator(model).simulate_to_steady_state().get_new_y0()
            {"ATP": 1.0, "NADPH": 1.0}

        """
        return dict(
            self._adjust_data(self.raw_variables, concatenated=True).iloc[-1]  # type: ignore
        )


@dataclass(kw_only=True, slots=True)
class Simulation(_BaseSimulation["KineticModelBuilder"]):
    """Simulation results."""

    @property
    def variables(self) -> pd.DataFrame:
        """Simulation variables."""
        return self.get_variables(
            include_derived_variables=True,
            include_surrogate_variables=True,
            include_readouts=True,
            concatenated=True,
            normalise=None,
        )

    @property
    def fluxes(self) -> pd.DataFrame:
        """Simulation fluxes."""
        return self.get_fluxes(
            include_surrogates=True,
        )

    def _python_args(self, variables: pd.DataFrame) -> pd.DataFrame:
        return self.model.get_args_time_course(
            variables=variables,
            include_variables=True,
            include_parameters=True,
            include_derived_parameters=True,
            include_derived_variables=True,
            include_reactions=True,
            include_surrogate_variables=True,
            include_surrogate_fluxes=True,
            include_readouts=True,
        )

    def _compute_args(self) -> list[pd.DataFrame]:
        # Already computed
        if len(self.raw_args) > 0:
            return self.raw_args

        # Compute new otherwise
        for res, p in zip(self.raw_variables, self.raw_parameters, strict=True):
            self.model.update_parameters(p)
            self.raw_args.append(self._args_for(res, self._python_args))
        return self.raw_args

    def _select_data(
        self,
        dependent: list[pd.DataFrame],
        *,
        include_variables: bool = False,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = False,
        include_reactions: bool = False,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
    ) -> list[pd.DataFrame]:
        names = self.model.get_arg_names(
            include_time=False,
            include_variables=include_variables,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_reactions=include_reactions,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_fluxes=include_surrogate_fluxes,
            include_readouts=include_readouts,
        )
        return [i.loc[:, names] for i in dependent]

    @overload
    def get_args(  # type: ignore
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_reactions: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_fluxes: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the variables over time.

        Parameters
        ----------
        include_variables
            Whether to include model variables.
        include_parameters
            Whether to include model parameters.
        include_derived_parameters
            Whether to include derived parameters.
        include_derived_variables
            Whether to include derived variables.
        include_reactions
            Whether to include reaction fluxes.
        include_surrogate_variables
            Whether to include surrogate variables.
        include_surrogate_fluxes
            Whether to include surrogate fluxes.
        include_readouts
            Whether to include readouts.
        concatenated
            Whether to concatenate results into a single DataFrame.
        normalise
            Normalization factor or array.

        Examples
        --------
            >>> Result().get_variables()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        """
        variables = self._select_data(
            self._compute_args(),
            include_variables=include_variables,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_reactions=include_reactions,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_fluxes=include_surrogate_fluxes,
            include_readouts=include_readouts,
        )
        return self._adjust_data(
            variables, normalise=normalise, concatenated=concatenated
        )

    @overload
    def get_variables(  # type: ignore
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the variables over time.

        Parameters
        ----------
        include_derived_variables
            Whether to include derived variables.
        include_readouts
            Whether to include readouts.
        include_surrogate_variables
            Whether to include surrogate variables.
        concatenated
            Whether to concatenate results into a single DataFrame.
        normalise
            Normalization factor or array.

        Examples
        --------
            >>> Result().get_variables()
            Time            ATP      NADPH
            0.000000   1.000000   1.000000
            0.000100   0.999900   0.999900
            0.000200   0.999800   0.999800

        """
        if not (
            include_derived_variables or include_readouts or include_surrogate_variables
        ):
            return self._adjust_data(
                self.raw_variables,
                normalise=normalise,
                concatenated=concatenated,
            )

        variables = self._select_data(
            self._compute_args(),
            include_variables=True,
            include_derived_variables=include_derived_variables,
            include_surrogate_variables=include_surrogate_variables,
            include_readouts=include_readouts,
        )
        return self._adjust_data(
            variables, normalise=normalise, concatenated=concatenated
        )

    @overload
    def get_fluxes(  # type: ignore
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_fluxes(
        self,
        *,
        include_surrogates: bool = True,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the flux results.

        Parameters
        ----------
        include_surrogates
            Whether to include surrogate fluxes.
        normalise
            Normalization factor or array.
        concatenated
            Whether to concatenate results into a single DataFrame.

        Examples
        --------
            >>> Result.get_fluxes()
            Time             v1         v2
            0.000000   1.000000   10.00000
            0.000100   0.999900   9.999000
            0.000200   0.999800   9.998000

        Returns
        -------
        pd.DataFrame
            DataFrame of fluxes.

        """
        fluxes = self._select_data(
            self._compute_args(),
            include_reactions=True,
            include_surrogate_fluxes=include_surrogates,
        )
        return self._adjust_data(
            fluxes,
            normalise=normalise,
            concatenated=concatenated,
        )

    def get_combined(self) -> pd.DataFrame:
        """Get the variables and fluxes as a single pandas.DataFrame.

        Examples
        --------
            >>> Result.get_combined()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   10.00000
            0.000100   0.999900   0.999900   0.999900   9.999000
            0.000200   0.999800   0.999800   0.999800   9.998000

        Returns
        -------
        pd.DataFrame
            DataFrame of fluxes.

        """
        return pd.concat((self.variables, self.fluxes), axis=1)

    @overload
    def get_right_hand_side(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get right hand side over time.

        Parameters
        ----------
        normalise
            Normalization factor or array.
        concatenated
            Whether to concatenate results into a single DataFrame.

        """
        args_by_simulation = self._compute_args()
        return self._adjust_data(
            [
                self.model.update_parameters(p).get_right_hand_side_time_course(
                    args=args
                )
                for args, p in zip(args_by_simulation, self.raw_parameters, strict=True)
            ],
            normalise=normalise,
            concatenated=concatenated,
        )

    @overload
    def get_producers(  # type: ignore
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_producers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get fluxes of variable with positive stoichiometry.

        Parameters
        ----------
        variable
            Name of the variable to get producers for.
        scaled
            Whether to scale fluxes by their stoichiometric coefficients.
        normalise
            Normalization factor or array.
        concatenated
            Whether to concatenate results into a single DataFrame.

        """
        self.model.update_parameters(self.raw_parameters[0])
        names = [
            k
            for k, v in self.model.get_stoichiometries_of_variable(variable).items()
            if v > 0
        ]

        fluxes: list[pd.DataFrame] = [
            i.loc[:, names]
            for i in self.get_fluxes(normalise=normalise, concatenated=False)
        ]

        if scaled:
            fluxes = [i.copy() for i in fluxes]
            for v, p in zip(fluxes, self.raw_parameters, strict=True):
                self.model.update_parameters(p)
                stoichs = self.model.get_stoichiometries_of_variable(variable)
                for k in names:
                    v.loc[:, k] *= stoichs[k]

        self.model.update_parameters(self.raw_parameters[-1])
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    @overload
    def get_consumers(  # type: ignore
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_consumers(
        self,
        variable: str,
        *,
        scaled: bool = False,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get fluxes of variable with negative stoichiometry.

        Parameters
        ----------
        variable
            Name of the variable to get consumers for.
        scaled
            Whether to scale fluxes by their stoichiometric coefficients.
        normalise
            Normalization factor or array.
        concatenated
            Whether to concatenate results into a single DataFrame.

        """
        self.model.update_parameters(self.raw_parameters[0])
        names = [
            k
            for k, v in self.model.get_stoichiometries_of_variable(variable).items()
            if v < 0
        ]

        fluxes: list[pd.DataFrame] = [
            i.loc[:, names]
            for i in self.get_fluxes(normalise=normalise, concatenated=False)
        ]

        if scaled:
            fluxes = [i.copy() for i in fluxes]
            for v, p in zip(fluxes, self.raw_parameters, strict=True):
                self.model.update_parameters(p)
                stoichs = self.model.get_stoichiometries_of_variable(variable)
                for k in names:
                    v.loc[:, k] *= -stoichs[k]

        self.model.update_parameters(self.raw_parameters[-1])
        if concatenated:
            return pd.concat(fluxes, axis=0)
        return fluxes

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))


@dataclass(kw_only=True, slots=True)
class OdeSimulation(_BaseSimulation["OdeModelBuilder"]):
    """Simulation results of an `OdeModelBuilder`.

    Unlike `Simulation` there are no fluxes, as the model is defined by its
    right hand side directly.
    """

    @property
    def variables(self) -> pd.DataFrame:
        """Simulation variables."""
        return self.get_variables(
            include_derived_variables=True,
            include_surrogate_variables=True,
            include_readouts=True,
            concatenated=True,
            normalise=None,
        )

    def _python_args(self, variables: pd.DataFrame) -> pd.DataFrame:
        dependent = self.model.get_args_time_course(
            variables=variables,
            include_parameters=True,
            include_derived_parameters=True,
            include_derived_variables=True,
            include_diff_eqs=False,
            include_surrogate_variables=True,
            include_surrogate_targets=True,
            include_readouts=True,
        )
        dependent = dependent.drop(columns=variables.columns, errors="ignore")
        return pd.concat((variables, dependent), axis=1)

    def _compute_args(self) -> list[pd.DataFrame]:
        # Already computed
        if len(self.raw_args) > 0:
            return self.raw_args

        # The builder's args hold dx/dt under each variable's name, so we
        # exclude them and use the integrated concentrations instead.
        # Variables are dropped explicitly in case the builder ever returns them.
        for res, p in zip(self.raw_variables, self.raw_parameters, strict=True):
            self.model.update_parameters(p)
            self.raw_args.append(self._args_for(res, self._python_args))
        return self.raw_args

    def _select_data(
        self,
        dependent: list[pd.DataFrame],
        *,
        include_variables: bool = False,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = False,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = False,
    ) -> list[pd.DataFrame]:
        names = self.model.get_variable_names() if include_variables else []
        names += self.model.get_arg_names(
            include_time=False,
            include_diff_eqs=False,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_targets=include_surrogate_targets,
            include_readouts=include_readouts,
        )
        return [i.loc[:, names] for i in dependent]

    @overload
    def get_args(  # type: ignore
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = False,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_args(
        self,
        *,
        include_variables: bool = True,
        include_parameters: bool = False,
        include_derived_parameters: bool = False,
        include_derived_variables: bool = True,
        include_surrogate_variables: bool = False,
        include_surrogate_targets: bool = False,
        include_readouts: bool = False,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the model arguments over time.

        Parameters
        ----------
        include_variables
            Whether to include model variables.
        include_parameters
            Whether to include model parameters.
        include_derived_parameters
            Whether to include derived parameters.
        include_derived_variables
            Whether to include derived variables.
        include_surrogate_variables
            Whether to include surrogate outputs without targets.
        include_surrogate_targets
            Whether to include surrogate outputs feeding a dx/dt.
        include_readouts
            Whether to include readouts.
        concatenated
            Whether to concatenate results into a single DataFrame.
        normalise
            Normalization factor or array.

        """
        args = self._select_data(
            self._compute_args(),
            include_variables=include_variables,
            include_parameters=include_parameters,
            include_derived_parameters=include_derived_parameters,
            include_derived_variables=include_derived_variables,
            include_surrogate_variables=include_surrogate_variables,
            include_surrogate_targets=include_surrogate_targets,
            include_readouts=include_readouts,
        )
        return self._adjust_data(args, normalise=normalise, concatenated=concatenated)

    @overload
    def get_variables(  # type: ignore
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[False],
        normalise: float | ArrayLike | None = None,
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: Literal[True],
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    @overload
    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame: ...

    def get_variables(
        self,
        *,
        include_derived_variables: bool = True,
        include_readouts: bool = True,
        include_surrogate_variables: bool = True,
        concatenated: bool = True,
        normalise: float | ArrayLike | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the variables over time.

        Parameters
        ----------
        include_derived_variables
            Whether to include derived variables.
        include_readouts
            Whether to include readouts.
        include_surrogate_variables
            Whether to include surrogate variables.
        concatenated
            Whether to concatenate results into a single DataFrame.
        normalise
            Normalization factor or array.

        """
        if not (
            include_derived_variables or include_readouts or include_surrogate_variables
        ):
            return self._adjust_data(
                self.raw_variables,
                normalise=normalise,
                concatenated=concatenated,
            )
        variables = self._select_data(
            self._compute_args(),
            include_variables=True,
            include_derived_variables=include_derived_variables,
            include_surrogate_variables=include_surrogate_variables,
            include_readouts=include_readouts,
        )
        return self._adjust_data(
            variables, normalise=normalise, concatenated=concatenated
        )

    def get_combined(self) -> pd.DataFrame:
        """Get the variables as a single pandas.DataFrame.

        The equivalent of `Simulation.get_combined`, which has no fluxes to add
        for an `OdeModelBuilder`.
        """
        return self.variables

    @overload
    def get_right_hand_side(  # type: ignore
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[False],
    ) -> list[pd.DataFrame]: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: Literal[True],
    ) -> pd.DataFrame: ...

    @overload
    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame: ...

    def get_right_hand_side(
        self,
        *,
        normalise: float | ArrayLike | None = None,
        concatenated: bool = True,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Get right hand side over time.

        Parameters
        ----------
        normalise
            Normalization factor or array.
        concatenated
            Whether to concatenate results into a single DataFrame.

        """
        return self._adjust_data(
            [
                self.model.update_parameters(p).get_args_time_course(
                    variables=res,
                    include_parameters=False,
                    include_derived_parameters=False,
                    include_derived_variables=False,
                    include_diff_eqs=True,
                    include_surrogate_variables=False,
                    include_surrogate_targets=False,
                    include_readouts=False,
                )
                for res, p in zip(self.raw_variables, self.raw_parameters, strict=True)
            ],
            normalise=normalise,
            concatenated=concatenated,
        )


@dataclass(kw_only=True, slots=True)
class Evaluation:
    """Evaluation results of a `SteadyStateModelBuilder`.

    Mirrors the accessors shared by `Simulation` and `OdeSimulation`, with a
    single row, as nothing is integrated over time.
    """

    model: SteadyStateModelBuilder
    raw_values: pd.Series

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @classmethod
    def default(cls, model: SteadyStateModelBuilder) -> Evaluation:
        """Get result filled with NaNs.

        Parameters
        ----------
        model
            Model to create default results for.

        """
        names = model.get_parameter_names() + list(model.get_raw_derived(as_copy=False))
        return Evaluation(
            model=model,
            raw_values=pd.Series(np.nan, index=names, dtype=float),
        )

    def get_args(self, *, include_parameters: bool = False) -> pd.DataFrame:
        """Get the evaluated values as a single row.

        Parameters
        ----------
        include_parameters
            Whether to include the parameters.

        """
        names = list(self.model.get_raw_derived(as_copy=False))
        if include_parameters:
            names = self.model.get_parameter_names() + names
        return self.raw_values.loc[names].to_frame().T

    def get_combined(self) -> pd.DataFrame:
        """Get the derived values as a single row."""
        return self.get_args()
