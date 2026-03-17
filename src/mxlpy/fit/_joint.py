###############################################################################
# Joint fitting
# This is multi-model, multi-data fitting, where the models share some parameters
###############################################################################


import multiprocessing
from collections.abc import Callable
from copy import deepcopy
from functools import partial

import numpy as np
import pebble

from mxlpy.fit import losses
from mxlpy.fit.abstract import (
    FitResidual,
    FitSettings,
    JointFit,
    _Settings,
)
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.integrators import IntegratorType
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
)
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

__all__ = ["joint_protocol_time_course", "joint_steady_state", "joint_time_course"]


def _unpacked[T1, T2, Tout](inp: tuple[T1, T2], fn: Callable[[T1, T2], Tout]) -> Tout:
    return fn(*inp)


def _sum_of_residuals(
    updates: dict[str, float],
    residual_fn: FitResidual,
    fits: list[_Settings],
    pool: pebble.ProcessPool,
) -> float:
    future = pool.map(
        partial(_unpacked, fn=residual_fn),
        [(updates, i) for i in fits],
        timeout=None,
    )
    error = 0.0
    it = future.result()
    while True:
        try:
            error += next(it)
        except StopIteration:
            break
        except TimeoutError:
            return np.inf
    return error


def joint_steady_state(
    to_fit: list[FitSettings],
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
    """Multi-model, multi-data fitting.

    Examples
    --------
        >>> fit.joint_steady_state(
        ...     [
        ...         fit.FitSettings(model=model_fn(), data=res.iloc[-1]),
        ...         fit.FitSettings(model=model_fn(), data=res.iloc[-1]),
        ...     ],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    to_fit
        Models and data to fit
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
                standard_scale=standard_scale,
                oscillation_detector=i.oscillation_detector,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=steady_state_residual,
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


def joint_time_course(
    to_fit: list[FitSettings],
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
    """Multi-model, multi-data fitting.

    Examples
    --------
        >>> fit.joint_steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Parameters
    ----------
    to_fit
        Models and data to fit
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
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=time_course_residual,
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


def joint_protocol_time_course(
    to_fit: list[FitSettings],
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
    """Multi-model, multi-data fitting.

    Examples
    --------
        >>> fit.joint_steady_state(model, p0, data)
        {'k1': 0.1, 'k2': 0.2}

    Parameters
    ----------
    to_fit
        Models and data to fit
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
                protocol=_normalise_protocol_index(protocol)
                if (protocol := i.protocol) is not None
                else None,
                p_names=[j for j in p0 if j in p_names],
                v_names=[j for j in p0 if j in v_names],
                standard_scale=standard_scale,
            )
        )

    with pebble.ProcessPool(
        max_workers=(
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )
    ) as pool:
        min_result = minimizer(
            partial(
                _sum_of_residuals,
                residual_fn=protocol_time_course_residual,
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
