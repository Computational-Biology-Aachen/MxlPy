# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Forward sensitivities of simulations w.r.t. parameters and initial values.

`d y(t) / d theta` for every variable, derived quantity and reaction `y` and
every requested parameter or variable initial value `theta`.

The sensitivities `S = d x / d theta` of the variables solve the augmented
system `dS/dt = J S + df/dtheta` alongside the model, with both derivatives
generated symbolically. Steady-state sensitivities come from the implicit
function theorem instead. Models which can't be differentiated symbolically
(e.g. neural-network surrogates or functions that can't be parsed) fall back
to central finite differences.

For global sensitivity analysis (Morris, Sobol) see `mxlpy.sensitivity`.

Examples
--------
    >>> sens = forward_sensitivities.time_course(model, [1, 2, 5], ["k1", "k2"])
    >>> sens.unwrap_or_err().derivatives["k1"]

"""

from __future__ import annotations

from ._core import ForwardMethod, ForwardSensitivities, steady_state, time_course

__all__ = [
    "ForwardMethod",
    "ForwardSensitivities",
    "steady_state",
    "time_course",
]
