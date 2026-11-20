# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Joint fitting
# This is multi-model, multi-data fitting, where the models share some parameters
###############################################################################


import contextlib
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import partial
from typing import Any

import numpy as np
from loky import ProcessPoolExecutor

from mxlpy.distributions import Prior
from mxlpy.fit import losses
from mxlpy.fit.abstract import (
    FitResidual,
    FitSettings,
    JointFit,
    PostprocessFn,
    _joint_settings,
    _Settings,
)
from mxlpy.fit.penalties import Penalty, collect_penalties, penalised
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.minimizers.abstract import (
    Bounds,
    LossFn,
    MinimizerProtocol,
    OptimisationState,
    Residual,
)
from mxlpy.runners import DynamicRunner, SteadyStateRunner
from mxlpy.types import Result

__all__ = ["joint_protocol_time_course", "joint_steady_state", "joint_time_course"]


def _unpacked[T1, T2, Tout](inp: tuple[T1, T2], fn: Callable[[T1, T2], Tout]) -> Tout:
    return fn(*inp)


def _sum_of_residuals(
    updates: dict[str, float],
    residual_fn: FitResidual,
    fits: list[_Settings],
    pool: ProcessPoolExecutor | None,
) -> float:
    if pool is None:
        return sum(residual_fn(updates, i) for i in fits)
    futures = [
        pool.submit(partial(_unpacked, fn=residual_fn), (updates, i)) for i in fits
    ]
    error = 0.0
    for future in futures:
        try:
            error += future.result()
        except FuturesTimeoutError:
            return np.inf
    return error


def _minimize_joint(
    residual: Residual,
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    bounds: Bounds | None,
    prior: Mapping[str, Prior] | None,
    penalty: Penalty | Sequence[Penalty] | None,
) -> Result[JointFit]:
    """Minimize the summed residual of a joint fit plus its penalties."""
    with_penalty = None
    if penalties := collect_penalties(prior, penalty, list(p0)):
        residual = with_penalty = penalised(residual, penalties)
    match minimizer(residual, p0, {} if bounds is None else bounds).value:
        case OptimisationState(parameters, value):
            if with_penalty is None:
                return Result(JointFit(parameters, loss=value))
            # Evaluated separately, as `value - penalty` loses the loss's
            # precision if the penalty dominates
            return Result(
                JointFit(
                    parameters,
                    loss=with_penalty.inner(parameters),
                    penalty=with_penalty.penalty(parameters),
                )
            )
        case _ as e:
            return Result(e)


def _pool(
    *, parallel: bool, max_workers: int | None
) -> ProcessPoolExecutor | contextlib.nullcontext[None]:
    if not parallel:
        return contextlib.nullcontext()
    return ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count() if max_workers is None else max_workers
    )


def joint_steady_state(
    to_fit: Sequence[FitSettings[Any]],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[Any, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    parallel: bool = True,
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
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
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
            needs_dynamics=False,
        )
        for i in to_fit
    ]

    with _pool(parallel=parallel, max_workers=max_workers) as pool:
        return _minimize_joint(
            partial(
                _sum_of_residuals,
                residual_fn=steady_state_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0=p0,
            minimizer=minimizer,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
        )


def joint_time_course(
    to_fit: Sequence[FitSettings[Any]],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    parallel: bool = True,
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
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
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
            needs_dynamics=True,
        )
        for i in to_fit
    ]

    with _pool(parallel=parallel, max_workers=max_workers) as pool:
        return _minimize_joint(
            partial(
                _sum_of_residuals,
                residual_fn=time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0=p0,
            minimizer=minimizer,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
        )


def joint_protocol_time_course(
    to_fit: Sequence[FitSettings[Any]],
    *,
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    parallel: bool = True,
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
    prior
        Prior per fitted parameter (`mxlpy.distributions`), whose negative log
        densities are added to the loss (maximum a posteriori with a
        `GaussianNll` loss, regularisation with others). Parameters without
        one are unpenalised. Same mapping as for `mcmc_*`
    penalty
        Further penalties added to the loss, see `mxlpy.fit.Penalty`, e.g.
        `PriorPenalty(prior, weight=...)` to weight priors against a
        heuristic loss
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
            needs_dynamics=True,
        )
        for i in to_fit
    ]

    with _pool(parallel=parallel, max_workers=max_workers) as pool:
        return _minimize_joint(
            partial(
                _sum_of_residuals,
                residual_fn=protocol_time_course_residual,
                fits=full_settings,
                pool=pool,
            ),
            p0=p0,
            minimizer=minimizer,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
        )
