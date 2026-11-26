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

Runners integrating a model (`SensitivityRunner`) also provide the
derivatives of their results w.r.t. parameters and initial values, which
gradient-based fits use: sensitivities (see `mxlpy.forward_sensitivities`)
and vector-Jacobian products, e.g. loss gradients (see
`mxlpy.adjoint_sensitivities`). `derivatives` chooses how they are computed.
Fits fall back to finite differences of the runner's results for runners
without them.

Examples
--------
    >>> scan.steady_state(model, to_scan=df, runner=KineticRunner(tolerance=1e-8))
    >>> fit.time_course(model, p0=p0, data=data, minimizer=minimizer,
    ...                 runner=KineticRunner(integrator=Scipy))

"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast, overload

from mxlpy import adjoint_sensitivities, forward_sensitivities
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.integrators.utils import OscillationDetector, no_oscillation_detection
from mxlpy.simulation import Evaluation, OdeSimulation, Simulation
from mxlpy.simulator import Simulator
from mxlpy.types import Result

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from mxlpy.adjoint_sensitivities import Pullback
    from mxlpy.forward_sensitivities import ForwardMethod, ForwardSensitivities
    from mxlpy.integrators import IntegratorType
    from mxlpy.types import ArrayLike, DerivativeMethod

__all__ = [
    "DynamicRunner",
    "EvaluationRunner",
    "KineticRunner",
    "OdeRunner",
    "SensitivityRunner",
    "SteadyStateRunner",
    "default_runner",
    "precompile",
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


class SensitivityRunner[M](Protocol):
    """Runner that can also differentiate its results."""

    def steady_state_sensitivities(
        self, model: M, wrt: list[str]
    ) -> Result[ForwardSensitivities]:
        """Derivatives of the steady state w.r.t. parameters / initial values."""
        ...

    def time_course_sensitivities(
        self, model: M, time_points: ArrayLike, wrt: list[str]
    ) -> Result[ForwardSensitivities]:
        """Derivatives of a time course w.r.t. parameters / initial values."""
        ...

    def steady_state_pullback(
        self, model: M, wrt: list[str], columns: list[str] | None = None
    ) -> Result[Pullback]:
        """The steady state and its vector-Jacobian products."""
        ...

    def time_course_pullback(
        self,
        model: M,
        time_points: ArrayLike,
        wrt: list[str],
        *,
        columns: list[str] | None = None,
    ) -> Result[Pullback]:
        """A time course and its vector-Jacobian products."""
        ...

    def protocol_time_course_pullback(
        self,
        model: M,
        protocol: pd.DataFrame,
        time_points: ArrayLike,
        wrt: list[str],
        *,
        columns: list[str] | None = None,
    ) -> Result[Pullback]:
        """A protocol time course and its vector-Jacobian products."""
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
        Callable for oscillation detection during steady-state calculations.
        Defaults to `no_oscillation_detection` (disabled).
    derivatives
        How derivatives are computed: `forward` integrates the sensitivity
        equations, `finite_difference` perturbs every parameter and simulates
        again. The adjoints give gradients of scalar losses (sensitivities,
        e.g. residual Jacobians, then come from `auto`): `discrete_adjoint`
        differentiates `Native`'s solver steps (exact for the computed
        solution), `continuous_adjoint` solves the adjoint equations
        backwards in time, `adjoint` takes the discrete one where possible.
        `auto` uses forward sensitivities unless the model can't be
        differentiated symbolically, and adjoints for the gradients of large
        problems and `Native` (see `adjoint_sensitivities.uses_adjoint`)
    displacement
        Relative step of the finite differences

    """

    integrator: IntegratorType | None = None
    tolerance: float = 1e-6
    rel_norm: bool = False
    oscillation_detector: OscillationDetector | None = no_oscillation_detection
    derivatives: DerivativeMethod = "auto"
    displacement: float = 1e-3

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

    def steady_state_sensitivities(
        self, model: Mi, wrt: list[str]
    ) -> Result[ForwardSensitivities]:
        """Derivatives of the steady state, see `forward_sensitivities`."""
        return forward_sensitivities.steady_state(
            model,
            wrt,
            integrator=self.integrator,
            method=self._forward_method,
            displacement=self.displacement,
            tolerance=self.tolerance,
            rel_norm=self.rel_norm,
        )

    def time_course_sensitivities(
        self, model: Mi, time_points: ArrayLike, wrt: list[str]
    ) -> Result[ForwardSensitivities]:
        """Derivatives of a time course, see `forward_sensitivities`."""
        return forward_sensitivities.time_course(
            model,
            time_points,
            wrt,
            integrator=self.integrator,
            method=self._forward_method,
            displacement=self.displacement,
        )

    def steady_state_pullback(
        self, model: Mi, wrt: list[str], *, columns: list[str] | None = None
    ) -> Result[Pullback]:
        """The steady state and its vector-Jacobian products, see `adjoint_sensitivities`."""
        return adjoint_sensitivities.steady_state_pullback(
            model,
            wrt,
            columns=columns,
            integrator=self.integrator,
            method=self.derivatives,
            displacement=self.displacement,
            tolerance=self.tolerance,
            rel_norm=self.rel_norm,
        )

    def time_course_pullback(
        self,
        model: Mi,
        time_points: ArrayLike,
        wrt: list[str],
        *,
        columns: list[str] | None = None,
    ) -> Result[Pullback]:
        """A time course and its vector-Jacobian products, see `adjoint_sensitivities`."""
        return adjoint_sensitivities.time_course_pullback(
            model,
            time_points,
            wrt,
            columns=columns,
            integrator=self.integrator,
            method=self.derivatives,
            displacement=self.displacement,
        )

    def protocol_time_course_pullback(
        self,
        model: Mi,
        protocol: pd.DataFrame,
        time_points: ArrayLike,
        wrt: list[str],
        *,
        columns: list[str] | None = None,
    ) -> Result[Pullback]:
        """A protocol time course and its vector-Jacobian products, see `adjoint_sensitivities`."""
        return adjoint_sensitivities.protocol_time_course_pullback(
            model,
            protocol,
            time_points,
            wrt,
            columns=columns,
            integrator=self.integrator,
            # Forward sensitivities don't support protocols
            method="finite_difference"
            if self.derivatives == "forward"
            else self.derivatives,
            displacement=self.displacement,
        )

    @property
    def _forward_method(self) -> ForwardMethod:
        """Method of sensitivities, which adjoints don't provide."""
        if self.derivatives in ("forward", "finite_difference"):
            return self.derivatives
        return "auto"


@dataclass(kw_only=True, slots=True)
class KineticRunner(_IntegratingRunner[KineticModelBuilder, Simulation]):
    """Runner for `KineticModelBuilder` models, see `_IntegratingRunner`."""


@dataclass(kw_only=True, slots=True)
class OdeRunner(_IntegratingRunner[OdeModelBuilder, OdeSimulation]):
    """Runner for `OdeModelBuilder` models, see `_IntegratingRunner`."""


@dataclass(kw_only=True, slots=True)
class EvaluationRunner:
    """Runner for `SteadyStateModelBuilder` models.

    These have no dynamics, so they are evaluated instead of integrated.

    Parameters
    ----------
    displacement
        Relative step of finite differences, e.g. of response coefficients.
        Smaller than the integrating runners' default, as evaluations have no
        integration error the step has to stay above

    """

    displacement: float = 1e-4

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


def precompile(runner: object, model: object) -> None:
    """Compile a model for a `Native` integrator in this process, if it uses one.

    Call this before running the model in parallel: `parallelise` sends the
    compiled code to its workers, so they don't each compile it themselves.
    """
    if (native := sys.modules.get("mxlpy.integrators.int_native")) is None:
        return
    if not isinstance(runner, _IntegratingRunner) or not isinstance(
        model, KineticModelBuilder | OdeModelBuilder
    ):
        return
    if (integrator := runner.integrator) is None:
        return
    base = integrator.func if isinstance(integrator, partial) else integrator
    if not (isinstance(base, type) and issubclass(base, native.Native)):
        return
    y0 = model.get_initial_conditions()
    # Errors are left to the workers, which report them like without precompiling.
    with contextlib.suppress(Exception):
        integrator(model, tuple(y0[k] for k in model.get_variable_names()), None)
