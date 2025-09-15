"""Fitting routines.

Single model, single data routines
----------------------------------
- `steady_state`
- `time_course`
- `protocol_time_course`

Multiple model, single data routines
------------------------------------
- `ensemble_steady_state`
- `ensemble_time_course`
- `ensemble_protocol_time_course`

A carousel is a special case of an ensemble, where the general
structure (e.g. stoichiometries) is the same, while the reactions kinetics
can vary
- `carousel_steady_state`
- `carousel_time_course`
- `carousel_protocol_time_course`

Multiple model, multiple data
-----------------------------
- `joint_steady_state`
- `joint_time_course`
- `joint_protocol_time_course`

Multiple model, multiple data, multiple methods
-----------------------------------------------
Here we also allow to run different methods (e.g. steady-state vs time courses)
for each combination of model:data.

- `joint_mixed`


Loss functions
--------------
- rmse

"""

# Convenience imports to enable fit.time_course(fit.LocalScipyMinimizer) pattern
from mxlpy.minimizers import (
    Bounds,
    GlobalScipyMinimizer,
    LocalScipyMinimizer,
    LossFn,
    OptimisationState,
    Residual,
)

from .abstract import (
    EnsembleFit,
    Fit,
    FitSettings,
    JointFit,
    MixedSettings,
)
from .losses import (
    cosine_similarity,
    mae,
    mean,
    mean_absolute_percentage,
    mean_squared,
    mean_squared_logarithmic,
    rmse,
)
from .routines import (
    carousel_protocol_time_course,
    carousel_steady_state,
    carousel_time_course,
    ensemble_protocol_time_course,
    ensemble_steady_state,
    ensemble_time_course,
    joint_mixed,
    joint_protocol_time_course,
    joint_steady_state,
    joint_time_course,
    protocol_time_course,
    protocol_time_course_residual,
    steady_state,
    steady_state_residual,
    time_course,
    time_course_residual,
)

__all__ = [
    "Bounds",
    "EnsembleFit",
    "Fit",
    "FitSettings",
    "GlobalScipyMinimizer",
    "JointFit",
    "LocalScipyMinimizer",
    "LossFn",
    "MixedSettings",
    "OptimisationState",
    "Residual",
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
    "cosine_similarity",
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
    "joint_mixed",
    "joint_protocol_time_course",
    "joint_steady_state",
    "joint_time_course",
    "mae",
    "mean",
    "mean_absolute_percentage",
    "mean_squared",
    "mean_squared_logarithmic",
    "protocol_time_course",
    "protocol_time_course_residual",
    "rmse",
    "steady_state",
    "steady_state_residual",
    "time_course",
    "time_course_residual",
]
