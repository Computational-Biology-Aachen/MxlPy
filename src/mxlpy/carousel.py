"""Reaction carousel."""

from __future__ import annotations

import itertools as it
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
from wadler_lindig import pformat

import mxlpy.fit.abstract
from mxlpy import fit, parallel, scan
from mxlpy.fit import losses
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.integrators import IntegratorType
from mxlpy.simulator import _normalise_protocol_index

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mxlpy import Model
    from mxlpy.integrators import IntegratorType
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
    from mxlpy.simulation import Simulation
    from mxlpy.types import Array, RateFn

__all__ = [
    "Carousel",
    "CarouselSteadyState",
    "CarouselTimeCourse",
    "ReactionTemplate",
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
]


@dataclass
class ReactionTemplate:
    """Template for a reaction in a model."""

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    fn: RateFn
    args: list[str]
    additional_parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class CarouselSteadyState:
    """Time course of a carousel simulation."""

    carousel: list[Model]
    results: list[Simulation]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_variables_by_model(self) -> pd.DataFrame:
        """Get the variables of the time course results, indexed by model."""
        return pd.DataFrame(
            {i: r.variables.iloc[-1] for i, r in enumerate(self.results)}
        ).T


@dataclass
class CarouselTimeCourse:
    """Time course of a carousel simulation."""

    carousel: list[Model]
    results: list[Simulation]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def get_variables_by_model(self) -> pd.DataFrame:
        """Get the variables of the time course results, indexed by model."""
        return pd.concat({i: r.variables for i, r in enumerate(self.results)})


def _dict_product[T1, T2](d: Mapping[T1, Iterable[T2]]) -> Iterable[dict[T1, T2]]:
    yield from (dict(zip(d.keys(), x, strict=True)) for x in it.product(*d.values()))


def _make_reaction_carousel(
    model: Model, rxns: dict[str, list[ReactionTemplate]]
) -> Iterable[Model]:
    for d in _dict_product(rxns):
        new = deepcopy(model)
        for rxn, template in d.items():
            new.add_parameters(template.additional_parameters)
            new.update_reaction(name=rxn, fn=template.fn, args=template.args)
        yield new


class Carousel:
    """A carousel of models with different reaction templates."""

    variants: list[Model]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __init__(
        self,
        model: Model,
        variants: dict[str, list[ReactionTemplate]],
    ) -> None:
        """Initialize the carousel with a model and reaction templates."""
        self.variants = list(
            _make_reaction_carousel(
                model=model,
                rxns=variants,
            )
        )

    def time_course(
        self,
        time_points: Array,
        *,
        y0: dict[str, float] | None = None,
        integrator: IntegratorType | None = None,
    ) -> CarouselTimeCourse:
        """Simulate the carousel of models over a time course.

        Parameters
        ----------
        time_points
            Array of time points to simulate.
        y0
            Initial values for variables. If None, uses model defaults.
        integrator
            Integrator to use. If None, uses the default integrator.

        """
        results = [
            i[1]
            for i in parallel.parallelise(
                partial(
                    scan._time_course_worker,  # noqa: SLF001
                    time_points=time_points,
                    integrator=integrator,
                    y0=y0,
                ),
                list(enumerate(self.variants)),
            )
        ]

        return CarouselTimeCourse(
            carousel=self.variants,
            results=results,
        )

    def protocol(
        self,
        protocol: pd.DataFrame,
        *,
        y0: dict[str, float] | None = None,
        integrator: IntegratorType | None = None,
    ) -> CarouselTimeCourse:
        """Simulate the carousel of models over a protocol time course.

        Parameters
        ----------
        protocol
            DataFrame defining the protocol steps.
        y0
            Initial values for variables. If None, uses model defaults.
        integrator
            Integrator to use. If None, uses the default integrator.

        """
        results = [
            i[1]
            for i in parallel.parallelise(
                partial(
                    scan._protocol_worker,  # noqa: SLF001
                    protocol=protocol,
                    integrator=integrator,
                    y0=y0,
                ),
                list(enumerate(self.variants)),
            )
        ]

        return CarouselTimeCourse(
            carousel=self.variants,
            results=results,
        )

    def protocol_time_course(
        self,
        protocol: pd.DataFrame,
        time_points: Array,
        *,
        y0: dict[str, float] | None = None,
        integrator: IntegratorType | None = None,
    ) -> CarouselTimeCourse:
        """Simulate the carousel of models over a protocol time course.

        Parameters
        ----------
        protocol
            DataFrame defining the protocol steps.
        time_points
            Array of time points to simulate within each protocol step.
        y0
            Initial values for variables. If None, uses model defaults.
        integrator
            Integrator to use. If None, uses the default integrator.

        """
        results = [
            i[1]
            for i in parallel.parallelise(
                partial(
                    scan._protocol_time_course_worker,  # noqa: SLF001
                    protocol=protocol,
                    integrator=integrator,
                    time_points=time_points,
                    y0=y0,
                ),
                list(enumerate(self.variants)),
            )
        ]

        return CarouselTimeCourse(
            carousel=self.variants,
            results=results,
        )

    def steady_state(
        self,
        *,
        y0: dict[str, float] | None = None,
        integrator: IntegratorType | None = None,
        rel_norm: bool = False,
    ) -> CarouselSteadyState:
        """Simulate the carousel of models to steady state.

        Parameters
        ----------
        y0
            Initial values for variables. If None, uses model defaults.
        integrator
            Integrator to use. If None, uses the default integrator.
        rel_norm
            Whether to use relative norm for steady-state convergence.

        """
        results = [
            i[1]
            for i in parallel.parallelise(
                partial(
                    scan._steady_state_worker,  # noqa: SLF001
                    integrator=integrator,
                    rel_norm=rel_norm,
                    y0=y0,
                ),
                list(enumerate(self.variants)),
            )
        ]

        return CarouselSteadyState(
            carousel=self.variants,
            results=results,
        )


###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


def carousel_steady_state(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: mxlpy.fit.abstract.FitResidual = steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
) -> mxlpy.fit.abstract.EnsembleFit:
    """Fit model parameters to steady-state experimental data over a carousel.

    Examples
    --------
        >>> fit.carousel_steady_state(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Parameters
    ----------
    carousel
        Model carousel to fit
    p0
        Initial parameter guesses as {parameter_name: value}
    data
        Experimental time course data
    protocol
        Experimental protocol
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state

    Returns
    -------
        Ensemble fit object

    """
    return fit.ensemble_steady_state(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )


def carousel_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: mxlpy.fit.abstract.FitResidual = time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
) -> mxlpy.fit.abstract.EnsembleFit:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples
    --------
        >>> fit.carousel_time_course(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Parameters
    ----------
    carousel
        Model carousel to fit
    p0
        Initial parameter guesses as {parameter_name: value}
    data
        Experimental time course data
    protocol
        Experimental protocol
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state

    Returns
    -------
        Ensemble fit object

    """
    return fit.ensemble_time_course(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )


def carousel_protocol_time_course(
    carousel: Carousel,
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: mxlpy.fit.abstract.FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
) -> mxlpy.fit.abstract.EnsembleFit:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples
    --------
        >>> fit.carousel_protocol_time_course(
        ...     carousel,
        ...     p0={
        ...         "beta": 0.1,
        ...         "gamma": 0.1,
        ...     },
        ...     protocol=protocol,
        ...     data=data,
        ...     minimizer=fit.LocalScipyMinimizer(),
        ... )

    Parameters
    ----------
    carousel
        Model carousel to fit
    p0
        Initial parameter guesses as {parameter_name: value}
    data
        Experimental time course data
    protocol
        Experimental protocol
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state

    Returns
    -------
        Ensemble fit object

    """
    protocol = _normalise_protocol_index(protocol)
    return fit.ensemble_protocol_time_course(
        carousel.variants,
        p0=p0,
        data=data,
        minimizer=minimizer,
        protocol=protocol,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
    )
