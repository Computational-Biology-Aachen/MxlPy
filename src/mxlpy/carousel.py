"""Reaction carousel."""

from __future__ import annotations

import itertools as it
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
from wadler_lindig import pformat

from mxlpy import parallel, scan

__all__ = ["Carousel", "CarouselSteadyState", "CarouselTimeCourse", "ReactionTemplate"]

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mxlpy import Model
    from mxlpy.types import Array, IntegratorType, RateFn, Result


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
    results: list[Result]

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
    results: list[Result]

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
        """Simulate the carousel of models over a time course."""
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
        """Simulate the carousel of models over a protocol time course."""
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
        """Simulate the carousel of models over a protocol time course."""
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
        """Simulate the carousel of models over a time course."""
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
