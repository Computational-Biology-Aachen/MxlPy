# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Group
# This is single-model, single data fitting, multiple initial parameter guesses
###############################################################################


from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, cast, overload

import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.distributions import Prior
from mxlpy.fit import losses
from mxlpy.fit._gradient import Transform
from mxlpy.fit._single import _steady_state, protocol_time_course, time_course
from mxlpy.fit.abstract import Fit, FitResidual, GroupFit, PostprocessFn
from mxlpy.fit.callbacks import Callback, failed, failure_message, tagged
from mxlpy.fit.penalties import Penalty
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
from mxlpy.parallel import parallelise
from mxlpy.runners import DynamicRunner, SteadyStateRunner
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

__all__ = ["group_protocol_time_course", "group_steady_state", "group_time_course"]


def _wrap_steady_state[M: _Model](
    task: tuple[Hashable, dict[str, float]],
    *,
    model: M,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    start, p0 = task
    return _steady_state(
        model=model,
        p0=p0,
        data=data,
        y0=y0,
        runner=runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
        transform=transform,
        callback=tagged(callback, start),
    )


def _wrap_time_course[M: KineticModelBuilder | OdeModelBuilder](
    task: tuple[Hashable, dict[str, float]],
    *,
    model: M,
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    start, p0 = task
    return time_course(
        model=model,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        runner=runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
        transform=transform,
        callback=tagged(callback, start),
    )


def _wrap_protocol_time_course[M: KineticModelBuilder | OdeModelBuilder](
    task: tuple[Hashable, dict[str, float]],
    *,
    model: M,
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> Result[Fit[M]]:
    start, p0 = task
    return protocol_time_course(
        model=model,
        p0=p0,
        data=data,
        protocol=protocol,
        y0=y0,
        runner=runner,
        loss_fn=loss_fn,
        postprocess_fn=postprocess_fn,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
        transform=transform,
        callback=tagged(callback, start),
    )


def _iterrows(df: pd.DataFrame) -> Iterable[tuple[Hashable, dict[str, float]]]:
    for name, val in df.iterrows():
        yield cast(Hashable, name), cast(dict, val.to_dict())


def _run_group[M: _Model](
    fn: Callable[[tuple[Hashable, dict[str, float]]], Result[Fit[M]]],
    p0: pd.DataFrame,
    *,
    callback: Callback | None,
    parallel: bool,
    timeout: float | None,
) -> GroupFit[M]:
    """Fit from every row of `p0`, keeping the fits and failures by start."""
    if not p0.index.is_unique:
        msg = "The starts (row labels of p0) have to be unique"
        raise ValueError(msg)
    inputs = [(k, (k, v)) for k, v in _iterrows(p0)]
    results = dict(parallelise(fn, inputs, parallel=parallel, timeout=timeout))
    fits: dict[Hashable, Fit[M]] = {}
    failures: dict[Hashable, str] = {}
    for k, _ in inputs:
        if k not in results:
            failures[k] = "timeout"
            failed(tagged(callback, k), None, start=str(k), message="timeout")
        elif isinstance(value := results[k].value, Exception):
            failures[k] = failure_message(value)
        else:
            fits[k] = value
    return GroupFit(fits, failures)


@overload
def group_steady_state[M: (KineticModelBuilder, OdeModelBuilder)](
    model: M,
    *,
    p0: pd.DataFrame,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> GroupFit[M]: ...


@overload
def group_steady_state(
    model: SteadyStateModelBuilder,
    *,
    p0: pd.DataFrame,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> GroupFit[SteadyStateModelBuilder]: ...


def group_steady_state[M: _Model](
    model: M,
    *,
    p0: pd.DataFrame,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> GroupFit[M]:
    """Fit model parameters to steady-state experimental data.

    Examples
    --------
        >>> fit.group_steady_state(
        ...     model_fn(),
        ...     data=res.iloc[-1],
        ...     p0=p0_guesses,
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    model
        Model to fit
    p0
        Initial parameter guesses
    data
        Experimental time course data
    protocol
        Experimental protocol
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
    time_points_per_step
        Number of time points per step in the protocol
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
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` (whose `start` is the row label of `p0`, as str)
        after every evaluation of every start and at the end of each, see
        `mxlpy.fit.callbacks`. Has to be picklable with `parallel=True`

    Returns
    -------
        Multi-start result: the fits and failures by start

    """
    return _run_group(
        partial(
            _wrap_steady_state,
            model=model,
            data=data,
            y0=y0,
            runner=runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            minimizer=minimizer,
            residual_fn=residual_fn,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
            as_deepcopy=as_deepcopy,
            transform=transform,
            callback=callback,
        ),
        p0,
        callback=callback,
        parallel=parallel,
        timeout=timeout,
    )


def group_time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    p0: pd.DataFrame,
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> GroupFit[M]:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples
    --------
        >>> fit.group_steady_state(
        ...     model_fn(),
        ...     data=res.iloc[-1],
        ...     p0=p0_guesses,
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    model
        Model to fit
    p0
        Initial parameter guesses
    data
        Experimental time course data
    protocol
        Experimental protocol
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
    time_points_per_step
        Number of time points per step in the protocol
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
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` (whose `start` is the row label of `p0`, as str)
        after every evaluation of every start and at the end of each, see
        `mxlpy.fit.callbacks`. Has to be picklable with `parallel=True`

    Returns
    -------
        Multi-start result: the fits and failures by start

    """
    return _run_group(
        partial(
            _wrap_time_course,
            model=model,
            data=data,
            y0=y0,
            runner=runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            minimizer=minimizer,
            residual_fn=residual_fn,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
            as_deepcopy=as_deepcopy,
            transform=transform,
            callback=callback,
        ),
        p0,
        callback=callback,
        parallel=parallel,
        timeout=timeout,
    )


def group_protocol_time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    p0: pd.DataFrame,
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
    transform: Transform | None = None,
    callback: Callback | None = None,
) -> GroupFit[M]:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples
    --------
        >>> fit.group_protocol_time_course(
        ...     model_fn(),
        ...     data=res_protocol,
        ...     protocol=protocol,
        ...     p0=p0_guesses,
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    model
        Model
    p0
        initial parameter guess
    data
        Experimental time course data
    protocol
        Experimental protocol
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
    time_points_per_step
        Number of time points per step in the protocol
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
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker
    transform
        Fit `log` (log10) or `linear` values, for all or per name. Defaults
        to linear. Log space suits values spanning orders of magnitude with
        `LeastSquaresMinimizer`, but the unit first steps of quasi-Newton
        methods like L-BFGS-B then change values tenfold
    callback
        Called with a `FitEvent` (whose `start` is the row label of `p0`, as str)
        after every evaluation of every start and at the end of each, see
        `mxlpy.fit.callbacks`. Has to be picklable with `parallel=True`

    Returns
    -------
        Multi-start result: the fits and failures by start

    """
    protocol = _normalise_protocol_index(protocol)
    return _run_group(
        partial(
            _wrap_protocol_time_course,
            model=model,
            data=data,
            protocol=protocol,
            y0=y0,
            runner=runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            minimizer=minimizer,
            residual_fn=residual_fn,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
            as_deepcopy=as_deepcopy,
            transform=transform,
            callback=callback,
        ),
        p0,
        callback=callback,
        parallel=parallel,
        timeout=timeout,
    )
