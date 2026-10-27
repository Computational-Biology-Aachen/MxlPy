# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Runners: how a model is turned into results.

Routines like `scan.*`, `fit.*`, `mc.*` and `mca.*` don't simulate models
themselves, but delegate that to a runner. A runner holds all configuration
of *how* a model is run (e.g. the integrator or the steady-state tolerance),
while the routine only describes *what* to do (which parameters to scan,
which data to fit).

Each model builder has its own runner:

- `KineticRunner` for `KineticModelBuilder`, returning `Simulation`
- `OdeRunner` for `OdeModelBuilder`, returning `OdeSimulation`
- `EvaluationRunner` for `SteadyStateModelBuilder`, returning `Evaluation`

Routines pick the default runner for the builder if none is given.
Custom behaviour is implemented by subclassing a runner and overriding one
of its methods, which then applies to every routine.

Runners return a `Result`, the routine decides what a failure means for it
(e.g. a NaN row for scans, an infinite residual for fits).

Examples
--------
    >>> scan.steady_state(model, to_scan=df, runner=KineticRunner(tolerance=1e-8))
    >>> fit.time_course(model, p0=p0, data=data, minimizer=minimizer,
    ...                 runner=KineticRunner(integrator=Scipy))

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, overload

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mxlpy.integrators import IntegratorType
    from mxlpy.types import ArrayLike

__all__ = [
    "DynamicRunner",
    "EvaluationRunner",
    "KineticRunner",
    "OdeRunner",
    "SteadyStateRunner",
    "default_runner",
]


class SteadyStateRunner[M, S](Protocol):
    """Runner that can compute steady states."""

    def steady_state(self, model: M) -> Result[S]:
        """Compute the steady state of the model."""
        ...


class DynamicRunner[M, S](SteadyStateRunner[M, S], Protocol):
    """Runner that can also integrate a model over time."""

    def time_course(self, model: M, time_points: ArrayLike) -> Result[S]:
        """Simulate the model at the given time points."""
        ...

    def protocol(
        self,
        model: M,
        protocol: pd.DataFrame,
        *,
        time_points_per_step: int = 10,
    ) -> Result[S]:
        """Simulate the model over a protocol."""
        ...

    def protocol_time_course(
        self,
        model: M,
        protocol: pd.DataFrame,
        time_points: ArrayLike,
    ) -> Result[S]:
        """Simulate the model over a protocol at the given time points."""
        ...


@dataclass(kw_only=True, slots=True)
class _IntegratingRunner[
    Mi: KineticModelBuilder | OdeModelBuilder,
    Si: Simulation | OdeSimulation,
]:
    """Shared implementation of the runners integrating a model with `Simulator`.

    Parameters
    ----------
    integrator
        Integrator to use. Defaults to `mxlpy.integrators.DefaultIntegrator`.
    tolerance
        Tolerance of the steady-state calculation.
    rel_norm
        Whether to use the relative norm for the steady-state calculation.
    oscillation_detector
        Callable for oscillation detection during steady-state calculations,
        or ``None`` to disable.

    """

    integrator: IntegratorType | None = None
    tolerance: float = 1e-6
    rel_norm: bool = False
    oscillation_detector: OscillationDetector | None = detect_oscillations

    def _run(
        self, model: Mi, simulate: Callable[[Simulator[Mi]], Simulator[Mi]]
    ) -> Result[Si]:
        try:
            res = simulate(Simulator(model, integrator=self.integrator)).get_result()
        except ZeroDivisionError as e:
            return Result(e)
        return cast(Result[Si], res)

    def steady_state(self, model: Mi) -> Result[Si]:
        """Simulate the model to its steady state."""
        return self._run(
            model,
            lambda s: s.simulate_to_steady_state(
                tolerance=self.tolerance,
                rel_norm=self.rel_norm,
                oscillation_detector=self.oscillation_detector,
            ),
        )

    def time_course(self, model: Mi, time_points: ArrayLike) -> Result[Si]:
        """Simulate the model at the given time points."""
        return self._run(model, lambda s: s.simulate_time_course(time_points))

    def protocol(
        self,
        model: Mi,
        protocol: pd.DataFrame,
        *,
        time_points_per_step: int = 10,
    ) -> Result[Si]:
        """Simulate the model over a protocol."""
        return self._run(
            model,
            lambda s: s.simulate_protocol(
                protocol=protocol, time_points_per_step=time_points_per_step
            ),
        )

    def protocol_time_course(
        self,
        model: Mi,
        protocol: pd.DataFrame,
        time_points: ArrayLike,
    ) -> Result[Si]:
        """Simulate the model over a protocol at the given time points."""
        return self._run(
            model,
            lambda s: s.simulate_protocol_time_course(
                protocol=protocol, time_points=time_points
            ),
        )


@dataclass(kw_only=True, slots=True)
class KineticRunner(_IntegratingRunner[KineticModelBuilder, Simulation]):
    """Runner for `KineticModelBuilder` models, see `_IntegratingRunner`."""


@dataclass(kw_only=True, slots=True)
class OdeRunner(_IntegratingRunner[OdeModelBuilder, OdeSimulation]):
    """Runner for `OdeModelBuilder` models, see `_IntegratingRunner`."""


@dataclass(kw_only=True, slots=True)
class EvaluationRunner:
    """Runner for `SteadyStateModelBuilder` models.

    These have no dynamics, so they are evaluated instead of integrated, which
    leaves nothing to configure.
    """

    def steady_state(self, model: SteadyStateModelBuilder) -> Result[Evaluation]:
        """Evaluate the model."""
        try:
            values = model.evaluate(include_parameters=True)
        except (ArithmeticError, ValueError) as e:
            return Result(e)
        return Result(Evaluation(model=model, raw_values=values))


@overload
def default_runner(model: KineticModelBuilder) -> KineticRunner: ...


@overload
def default_runner(model: OdeModelBuilder) -> OdeRunner: ...


@overload
def default_runner(model: SteadyStateModelBuilder) -> EvaluationRunner: ...


def default_runner(
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
) -> KineticRunner | OdeRunner | EvaluationRunner:
    """Get the default runner for a model builder."""
    if isinstance(model, SteadyStateModelBuilder):
        return EvaluationRunner()
    if isinstance(model, OdeModelBuilder):
        return OdeRunner()
    return KineticRunner()
