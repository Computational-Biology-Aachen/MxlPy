# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Joint fitting
# This is multi-model, multi-data, multi-simulation fitting
# The models share some parameters here, everything else can be changed though
###############################################################################


from collections.abc import Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import partial
from typing import Any, cast

import numpy as np
from loky import ProcessPoolExecutor

from mxlpy.fit import losses
from mxlpy.fit._joint import _pool
from mxlpy.fit.abstract import (
    FitResidual,
    JointFit,
    MixedSettings,
    PostprocessFn,
    _joint_settings,
    _Settings,
)
from mxlpy.fit.residuals import steady_state_residual
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
)
from mxlpy.runners import SteadyStateRunner
from mxlpy.types import Result

__all__ = ["joint_mixed"]


def _execute(inp: tuple[dict[str, float], FitResidual, _Settings]) -> float:
    updates, residual_fn, settings = inp
    return residual_fn(updates, settings)


def _mixed_sum_of_residuals(
    updates: dict[str, float],
    fits: list[_Settings],
    pool: ProcessPoolExecutor | None,
) -> float:
    if pool is None:
        return sum(
            _execute((updates, cast(FitResidual, i.residual_fn), i)) for i in fits
        )
    futures = [pool.submit(_execute, (updates, i.residual_fn, i)) for i in fits]
    error = 0.0
    for future in futures:
        try:
            error += future.result()
        except FuturesTimeoutError:
            return np.inf
    return error


def joint_mixed(
    to_fit: Sequence[MixedSettings[Any]],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[JointFit]:
    """Multi-model, multi-data, multi-simulation fitting.

    Examples
    --------
        >>> fit.joint_mixed(
        ...     [
        ...         fit.MixedSettings(
        ...             model=model_fn(),
        ...             data=res.iloc[-1],
        ...             residual_fn=fit.steady_state_residual,
        ...         ),
        ...         fit.MixedSettings(
        ...             model=model_fn(),
        ...             data=res,
        ...             residual_fn=fit.time_course_residual,
        ...         ),
        ...     ],
        ...     p0={"k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    to_fit
        models, data and residual fn to fit
    p0
        Initial guesses as {name: value}
    y0
        Initial conditions as {species_name: value}
    minimizer
        Function to minimize fitting error
    residual_fn
        Function to calculate fitting error
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to the default runner of the model builder.
    loss_fn
        Loss function to use for residual calculation
    postprocess_fn
        Function applied to the combined simulation result before the loss is
        calculated, e.g. to compute quantities derived from the whole time
        series. Its output has to contain the data columns.
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    parallel
        Whether to evaluate the models in parallel (default: True).
    max_workers
        maximal amount of workers in parallel
    standard_scale
        Whether to apply standard scale to data and prediction. Skipped for
        fewer than two data points.

    Returns
    -------
        Result object containing JointFit object

    """
    full_settings = [
        _joint_settings(
            i,
            p0=p0,
            y0=y0,
            runner=runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            as_deepcopy=as_deepcopy,
            standard_scale=standard_scale,
            needs_dynamics=i.residual_fn is not steady_state_residual,
        )
        for i in to_fit
    ]

    with _pool(parallel=parallel, max_workers=max_workers) as pool:
        min_result = minimizer(
            partial(
                _mixed_sum_of_residuals,
                fits=full_settings,
                pool=pool,
            ),
            p0,
            {} if bounds is None else bounds,
        )

    match min_result.value:
        case OptimisationState(parameters, residual):
            return Result(JointFit(parameters, loss=residual))
        case _ as e:
            return Result(e)
