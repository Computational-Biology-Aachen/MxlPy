# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Adjoint sensitivities: gradients of weighted sums of simulated values.

`d/dθ Σ w(t)ᵀ y(t)` for given cotangents `w` (e.g. the derivative of a loss
w.r.t. the simulated values) and every requested parameter or variable
initial value `θ`, i.e. the vector-Jacobian product of a simulation.

Forward sensitivities (`mxlpy.forward_sensitivities`) integrate `n` extra
states per parameter. Adjoints integrate `λ` backwards in time instead,
`dλ/dt = -(∂f/∂x)ᵀ λ`, jumping by the cotangents at every time point, and
accumulate `∫ λᵀ ∂f/∂θ dt`. Their cost doesn't grow with the number of
parameters, which makes fits with hundreds of parameters practical (e.g.
neural-network weights in hybrid models). Both products are generated
symbolically in reverse mode, without forming Jacobians w.r.t. the
parameters.

The backward solve needs the forward trajectory at arbitrary times. It is
interpolated (cubic Hermite) on a grid refined until the interpolation error
is within the integrator's tolerances, never integrated backwards, which is
unstable for stiff models.

Steady states use the adjoint of the implicit function theorem.

Examples
--------
    >>> w = pd.DataFrame({"x": [1.0, -1.0]}, index=[1.0, 2.0])
    >>> adjoint_sensitivities.time_course(model, w, ["k1", "k2"]).unwrap_or_err().gradient

"""

from __future__ import annotations

from ._core import (
    ADJOINT_THRESHOLD,
    PROTOCOL_ADJOINT_THRESHOLD,
    AdjointSensitivities,
    protocol_time_course,
    steady_state,
    time_course,
    uses_adjoint,
)

__all__ = [
    "ADJOINT_THRESHOLD",
    "PROTOCOL_ADJOINT_THRESHOLD",
    "AdjointSensitivities",
    "protocol_time_course",
    "steady_state",
    "time_course",
    "uses_adjoint",
]
