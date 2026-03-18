###############################################################################
# Joint fitting
# This is multi-model, multi-data, multi-simulation fitting
# The models share some parameters here, everything else can be changed though
###############################################################################


import multiprocessing
from concurrent.futures import TimeoutError as FuturesTimeoutError
from copy import deepcopy
from functools import partial

import numpy as np
from loky import ProcessPoolExecutor

from mxlpy.fit import losses
from mxlpy.fit.abstract import FitResidual, JointFit, MixedSettings, _Settings
from mxlpy.integrators import IntegratorType
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
)
from mxlpy.types import Result

__all__ = ["joint_mixed"]


def _execute(inp: tuple[dict[str, float], FitResidual, _Settings]) -> float:
    updates, residual_fn, settings = inp
    return residual_fn(updates, settings)


def _mixed_sum_of_residuals(
    updates: dict[str, float],
    fits: list[_Settings],
    pool: ProcessPoolExecutor,
) -> float:
    futures = [pool.submit(_execute, (updates, i.residual_fn, i)) for i in fits]
    error = 0.0
    for future in futures:
        try:
            error += future.result()
        except FuturesTimeoutError:
            return np.inf
    return error


def joint_mixed(
    to_fit: list[MixedSettings],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    max_workers
        maximal amount of workers in parallel
    standard_scale
        Whether to apply standard scale to data and prediction

    Returns
    -------
        Result object containing JointFit object

    """
    full_settings = []
    for i in to_fit:
        p_names = i.model.get_parameter_names()
        v_names = i.model.get_variable_names()
        full_settings.append(
            _Settings(
                model=deepcopy(i.model) if as_deepcopy else i.model,
                data=i.data,
                y0=i.y0 if i.y0 is not None else y0,
                integrator=i.integrator if i.integrator is not None else integrator,
                loss_fn=i.loss_fn if i.loss_fn is not None else loss_fn,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                protocol=i.protocol,
                residual_fn=i.residual_fn,
                standard_scale=standard_scale,
            )
        )

    with ProcessPoolExecutor(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
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
