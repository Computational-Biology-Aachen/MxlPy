# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Fitting routines.

Single model, single data routines
----------------------------------
- `steady_state`
- `time_course`
- `protocol_time_course`

Single model, single data, multiple initial guesses (multi-start)
-----------------------------------------------------------------
Also known as **multi-start** optimisation: the same model is fitted to the
same data from many different initial parameter guesses (one per row of the
`p0` DataFrame, e.g. generated with `mxlpy.distributions.sample`). The starts
run in parallel and a `GroupFit` is returned, whose `get_best_fit()` yields
the best result, `get_losses()` the loss of every start and `failures` the
reason of every failed start.
- `group_steady_state`
- `group_time_course`
- `group_protocol_time_course`

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


Hidden-state blocks (augmented ODEs)
------------------------------------
Train a model's hidden-state blocks (`add_hidden_states`) starting from the
fitted mechanistic parameters: scaled, weights only with priors and a
penalty on the corrections, then optionally jointly.

- `augmented_time_course`
- `augmented_protocol_time_course`


Bayesian posterior sampling (MCMC)
----------------------------------
Instead of a point estimate, sample the posterior distribution of the
parameters given the data, a Gaussian measurement noise `sigma` and a prior
per parameter, e.g. for credible intervals.

- `mcmc_steady_state`
- `mcmc_time_course`
- `mcmc_protocol_time_course`

Samplers: `AdaptiveMetropolis`

Priors and penalties (MAP estimation, regularisation)
-----------------------------------------------------
All point-estimate fits take `prior={name: distribution}` (the same mapping
as `mcmc_*`) and further `penalty=` terms (`Penalty`, `PriorPenalty`), and
minimise `loss + penalty`. With the `GaussianNll` loss this is a
maximum-a-posteriori estimate. `Fit.loss` is the loss of the data alone,
`Fit.penalty` the penalty at the optimum. `L1Penalty` (lasso, elastic net,
fused lasso) and `GroupL1Penalty` pull values to zero (sparsity).

Gradients
---------
`loss_gradient` evaluates the gradient of a fit's objective, as the
gradient-based minimizers see it. The runner chooses how (see
`mxlpy.runners`): adjoints (`derivatives="adjoint"`, automatic for large
time courses) cost one forward and one backward solve for any number of
parameters.

Progress
--------
Every fit function takes `callback=`, called with a `FitEvent` after every
evaluation and when a fit finished or failed (`start` tags the start of
multi-start fits). `JsonlCallback` appends the events to a file, also from
parallel workers. Multi-start fits keep failed starts in
`GroupFit.failures`.

Loss functions
--------------
- rmse, mean_squared, mae, ... (heuristic, compared after standard scaling)
- GaussianNll: negative log-likelihood of Gaussian noise with known sigma,
  making fits maximum-likelihood (or, with a `prior`, MAP) estimates

"""

from mxlpy.carousel import (
    carousel_protocol_time_course,
    carousel_steady_state,
    carousel_time_course,
)
from mxlpy.minimizers import (
    Bounds,
    FixedNMinimizer,
    GlobalScipyMinimizer,
    LeastSquaresMinimizer,
    LocalScipyMinimizer,
    LossFn,
    OptimisationState,
    Residual,
)

from ._augmented import (
    AugmentedFit,
    augmented_protocol_time_course,
    augmented_time_course,
)
from ._ensemble import (
    ensemble_protocol_time_course,
    ensemble_steady_state,
    ensemble_time_course,
)
from ._group import group_protocol_time_course, group_steady_state, group_time_course
from ._joint import (
    joint_protocol_time_course,
    joint_steady_state,
    joint_time_course,
)
from ._joint_mixed import joint_mixed
from ._mcmc import (
    AdaptiveMetropolis,
    McmcChain,
    McmcResult,
    McmcSampler,
    mcmc_protocol_time_course,
    mcmc_steady_state,
    mcmc_time_course,
)
from ._single import loss_gradient, protocol_time_course, steady_state, time_course
from .abstract import (
    Condition,
    Data,
    EnsembleFit,
    Fit,
    FitSettings,
    GroupFit,
    JointFit,
    MixedSettings,
)
from .callbacks import Callback, FitEvent, JsonlCallback
from .losses import (
    GaussianNll,
    Sigma,
    cosine_similarity,
    mae,
    mean,
    mean_absolute_percentage,
    mean_squared,
    mean_squared_logarithmic,
    rmse,
)
from .penalties import GroupL1Penalty, L1Penalty, Penalty, PriorPenalty
from .residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)

__all__: list[str] = [
    "AdaptiveMetropolis",
    "AugmentedFit",
    "Bounds",
    "Callback",
    "Condition",
    "Data",
    "EnsembleFit",
    "Fit",
    "FitEvent",
    "FitSettings",
    "FixedNMinimizer",
    "GaussianNll",
    "GlobalScipyMinimizer",
    "GroupFit",
    "GroupL1Penalty",
    "JointFit",
    "JsonlCallback",
    "L1Penalty",
    "LeastSquaresMinimizer",
    "LocalScipyMinimizer",
    "LossFn",
    "McmcChain",
    "McmcResult",
    "McmcSampler",
    "MixedSettings",
    "OptimisationState",
    "Penalty",
    "PriorPenalty",
    "Residual",
    "Sigma",
    "augmented_protocol_time_course",
    "augmented_time_course",
    "carousel_protocol_time_course",
    "carousel_steady_state",
    "carousel_time_course",
    "cosine_similarity",
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
    "group_protocol_time_course",
    "group_steady_state",
    "group_time_course",
    "joint_mixed",
    "joint_protocol_time_course",
    "joint_steady_state",
    "joint_time_course",
    "loss_gradient",
    "mae",
    "mcmc_protocol_time_course",
    "mcmc_steady_state",
    "mcmc_time_course",
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
