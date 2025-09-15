"""Simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
from wadler_lindig import pformat

__all__ = ["Simulation"]

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mxlpy.model import Model
    from mxlpy.types import Array, ArrayLike


def _normalise_split_results(
    results: list[pd.DataFrame],
    normalise: float | ArrayLike,
) -> list[pd.DataFrame]:
    """Normalize split results by a given factor or array.

    Args:
        results: List of DataFrames containing the results to normalize.
        normalise: Normalization factor or array.

    Returns:
        list[pd.DataFrame]: List of normalized DataFrames.

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
class Simulation:
    """Simulation results."""

    model: Model
    raw_variables: list[pd.DataFrame]
    raw_parameters: list[dict[str, float]]
    raw_args: list[pd.DataFrame] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @classmethod
    def default(cls, model: Model, time_points: Array) -> Simulation:
        """Get result filled with NaNs."""
        return Simulation(
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

    def _compute_args(self) -> list[pd.DataFrame]:
        # Already computed
        if len(self.raw_args) > 0:
            return self.raw_args

        # Compute new otherwise
        for res, p in zip(self.raw_variables, self.raw_parameters, strict=True):
            self.model.update_parameters(p)
            self.raw_args.append(
                self.model.get_args_time_course(
                    variables=res,
                    include_variables=True,
                    include_parameters=True,
                    include_derived_parameters=True,
                    include_derived_variables=True,
                    include_reactions=True,
                    include_surrogate_variables=True,
                    include_surrogate_fluxes=True,
                    include_readouts=True,
                )
            )
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

        Examples:
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

        Examples:
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

        Examples:
            >>> Result.get_fluxes()
            Time             v1         v2
            0.000000   1.000000   10.00000
            0.000100   0.999900   9.999000
            0.000200   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

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

        Examples:
            >>> Result.get_combined()
            Time            ATP      NADPH         v1         v2
            0.000000   1.000000   1.000000   1.000000   10.00000
            0.000100   0.999900   0.999900   0.999900   9.999000
            0.000200   0.999800   0.999800   0.999800   9.998000

        Returns:
            pd.DataFrame: DataFrame of fluxes.

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
        """Get right hand side over time."""
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
        """Get fluxes of variable with positive stoichiometry."""
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
        """Get fluxes of variable with negative stoichiometry."""
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

    def get_new_y0(self) -> dict[str, float]:
        """Get the new initial conditions after the simulation.

        Examples:
            >>> Simulator(model).simulate_to_steady_state().get_new_y0()
            {"ATP": 1.0, "NADPH": 1.0}

        """
        return dict(
            self.get_variables(
                include_derived_variables=False,
                include_readouts=False,
                include_surrogate_variables=False,
            ).iloc[-1]
        )

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate over the concentration and flux response coefficients."""
        return iter((self.variables, self.fluxes))
