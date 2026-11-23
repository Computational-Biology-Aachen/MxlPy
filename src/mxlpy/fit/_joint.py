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
from typing import Any, cast

import numpy as np
from loky import ProcessPoolExecutor

from mxlpy.distributions import Prior
from mxlpy.fit import losses
from mxlpy.fit._gradient import GradientResidual
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
from mxlpy.types import Array, Result

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


class _JointGradient:
    """Summed residual of a joint fit, with the derivatives of its parts.

    The loss is the sum of the losses of all settings (`loss`, evaluated in
    parallel if requested), so its gradient and Hessian are the sums of
    theirs, each w.r.t. the parameters of its own model. The residual vector
    stacks the parts' vectors scaled by sqrt(c), c their `squares_factor`, so
    its sum of squares / 2 is the loss, which needs sum-of-squares losses.
    """

    def __init__(
        self, loss: Residual, parts: list[GradientResidual], p0: dict[str, float]
    ) -> None:
        self.loss = loss
        self.parts = parts
        names = list(p0)
        # Per part: the names it fits and their positions in the updates
        self.names = [
            [
                k
                for k in names
                if k in set(part.settings.p_names + part.settings.v_names)
            ]
            for part in parts
        ]
        self.columns = [[names.index(k) for k in n] for n in self.names]

    def __call__(self, updates: dict[str, float]) -> float:
        return self.loss(updates)

    def _sub(self, i: int, updates: dict[str, float]) -> dict[str, float]:
        return {k: updates[k] for k in self.names[i]}

    def gradient(self, updates: dict[str, float]) -> Array:
        out = np.zeros(len(updates))
        for i, part in enumerate(self.parts):
            out[self.columns[i]] += part.gradient(self._sub(i, updates))
        return out

    def _factors(self) -> list[float]:
        factors = [part.squares_factor() for part in self.parts]
        if any(c is None for c in factors):
            msg = "Residual vectors of joint fits need sum-of-squares losses (mean_squared or GaussianNll)"
            raise TypeError(msg)
        return cast(list[float], factors)

    def squares_factor(self) -> float | None:
        if any(part.squares_factor() is None for part in self.parts):
            return None
        return 1.0

    def residuals(self, updates: dict[str, float]) -> Array:
        return np.concatenate(
            [
                np.sqrt(c) * part.residuals(self._sub(i, updates))
                for i, (part, c) in enumerate(
                    zip(self.parts, self._factors(), strict=True)
                )
            ]
        )

    def jacobian(self, updates: dict[str, float]) -> Array:
        blocks = []
        for i, (part, c) in enumerate(zip(self.parts, self._factors(), strict=True)):
            jac = part.jacobian(self._sub(i, updates))
            block = np.zeros((jac.shape[0], len(updates)))
            block[:, self.columns[i]] = np.sqrt(c) * jac
            blocks.append(block)
        return np.vstack(blocks)

    def hessian(self, updates: dict[str, float]) -> Array | None:
        out = np.zeros((len(updates), len(updates)))
        for i, part in enumerate(self.parts):
            if (h := part.hessian(self._sub(i, updates))) is None:
                return None
            out[np.ix_(self.columns[i], self.columns[i])] += h
        return out


def _with_derivatives(
    loss: Residual, parts: list[tuple[FitResidual, _Settings]], p0: dict[str, float]
) -> Residual:
    """The joint loss with derivatives, if all its parts provide them."""
    if not all(GradientResidual.supports(fn) for fn, _ in parts):
        return loss
    return cast(
        "Residual",
        _JointGradient(loss, [GradientResidual(fn, s) for fn, s in parts], p0),
    )


def _minimize_joint(
    residual: Residual,
    *,
    parts: list[tuple[FitResidual, _Settings]],
    p0: dict[str, float],
    minimizer: MinimizerProtocol,
    bounds: Bounds | None,
    prior: Mapping[str, Prior] | None,
    penalty: Penalty | Sequence[Penalty] | None,
) -> Result[JointFit]:
    """Minimize the summed residual of a joint fit plus its penalties."""
    residual = _with_derivatives(residual, parts, p0)
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
            parts=[(steady_state_residual, i) for i in full_settings],
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
            parts=[(time_course_residual, i) for i in full_settings],
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
            parts=[(protocol_time_course_residual, i) for i in full_settings],
            p0=p0,
            minimizer=minimizer,
            bounds=bounds,
            prior=prior,
            penalty=penalty,
        )
