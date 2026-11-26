# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint sensitivities: gradients of weighted sums of simulated values.

`d/dθ Σ w(t)ᵀ y(t)` for given cotangents `w` (e.g. the derivative of a loss
w.r.t. the simulated values) and every requested parameter or variable
initial value `θ`, i.e. the vector-Jacobian product of a simulation.

`time_course_pullback` (and `protocol_time_course_pullback`,
`steady_state_pullback`) simulate once and return a `Pullback`: its `simulated` values can be turned
into a loss first, and calling it with the loss's derivative returns the
gradient, without simulating again (as `jax.vjp`). `time_course(model,
cotangents, wrt)` does both in one call.

Forward sensitivities (`mxlpy.forward_sensitivities`) integrate `n` extra
states per parameter. Adjoints cost about the same for any number of
parameters, which makes fits with hundreds of parameters practical (e.g.
neural-network weights in hybrid models). There are two kinds:

- *Discrete* adjoints (`Native` integrator, all its methods) differentiate the
  solver's own steps, recorded during the forward solve (step sizes frozen,
  Newton iterations by the implicit function theorem, as diffrax's
  `RecursiveCheckpointAdjoint`). The gradient is exactly that of the computed
  solution, so it agrees with the loss an optimiser sees ("discretise, then
  optimise"). The recording is bounded by `Native.max_tape_bytes`; beyond it
  `discrete_adjoint` raises `TapeTooLarge`.
- *Continuous* adjoints integrate `dλ/dt = -(∂f/∂x)ᵀ λ` backwards in time,
  jumping by the cotangents at every time point, and accumulate
  `∫ λᵀ ∂f/∂θ dt` ("optimise, then discretise"). They work with every
  integrator, but their gradient is that of the exact solution, which differs
  from the computed one's by O(tolerance). The forward trajectory is
  interpolated (cubic Hermite) on a grid refined until the interpolation error
  is within the integrator's tolerances, never integrated backwards, which is
  unstable for stiff models. Both products are generated symbolically in
  reverse mode, without forming Jacobians w.r.t. the parameters.

Steady states use the adjoint of the implicit function theorem.

Examples
--------
    >>> pullback = adjoint_sensitivities.time_course_pullback(model, [1.0, 2.0], ["k1", "k2"]).unwrap_or_err()
    >>> loss = (pullback.simulated["x"] ** 2).sum()
    >>> pullback(2 * pullback.simulated[["x"]]).unwrap_or_err().gradient

"""

from __future__ import annotations

from ._core import (
    ADJOINT_THRESHOLD,
    PROTOCOL_ADJOINT_THRESHOLD,
    AdjointSensitivities,
    Pullback,
    protocol_time_course,
    protocol_time_course_pullback,
    steady_state,
    steady_state_pullback,
    time_course,
    time_course_pullback,
    uses_adjoint,
)
from ._discrete import TapeTooLarge

__all__ = [
    "ADJOINT_THRESHOLD",
    "PROTOCOL_ADJOINT_THRESHOLD",
    "AdjointSensitivities",
    "Pullback",
    "TapeTooLarge",
    "protocol_time_course",
    "protocol_time_course_pullback",
    "steady_state",
    "steady_state_pullback",
    "time_course",
    "time_course_pullback",
    "uses_adjoint",
]
