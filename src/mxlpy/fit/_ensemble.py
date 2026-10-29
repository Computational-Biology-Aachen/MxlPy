# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


from collections.abc import Sequence
from functools import partial
from typing import Any, overload

import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit import losses
from mxlpy.fit._single import _steady_state, protocol_time_course, time_course
from mxlpy.fit.abstract import EnsembleFit, FitResidual, PostprocessFn
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
from mxlpy.parallel import parallelise
from mxlpy.runners import DynamicRunner, SteadyStateRunner
from mxlpy.simulator import _normalise_protocol_index

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

__all__ = [
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
]


@overload
def ensemble_steady_state[M: (KineticModelBuilder, OdeModelBuilder)](
    ensemble: Sequence[M],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
) -> EnsembleFit[M]: ...


@overload
def ensemble_steady_state(
    ensemble: Sequence[SteadyStateModelBuilder],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
) -> EnsembleFit[SteadyStateModelBuilder]: ...


def ensemble_steady_state[M: _Model](
    ensemble: Sequence[M],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    runner: SteadyStateRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
) -> EnsembleFit[M]:
    """Fit model ensemble parameters to steady-state experimental data.

    Examples
    --------
        >>> fit.ensemble_steady_state(
        ...     [
        ...         model_fn(),
        ...         model_fn(),
        ...     ],
        ...     data=res.iloc[-1],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    ensemble
        Ensemble to fit
    p0
        Initial parameter guesses as {parameter_name: value}
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
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    parallel
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker

    Returns
    -------
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallelise(
                partial(
                    _steady_state,
                    p0=p0,
                    data=data,
                    y0=y0,
                    runner=runner,
                    loss_fn=loss_fn,
                    postprocess_fn=postprocess_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                parallel=parallel,
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def ensemble_time_course[M: KineticModelBuilder | OdeModelBuilder](
    ensemble: Sequence[M],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
) -> EnsembleFit[M]:
    """Fit model parameters to time course of experimental data over a carousel.

    Time points are taken from the data.

    Examples
    --------
        >>> fit.ensemble_steady_state(
        ...     [
        ...         model1,
        ...         model2,
        ...     ],
        ...     data=res.iloc[-1],
        ...     p0={"k1": 1.038, "k2": 1.87, "k3": 1.093},
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    ensemble
        Model ensemble to fit
    p0
        Initial parameter guesses as {parameter_name: value}
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
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    parallel
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker

    Returns
    -------
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallelise(
                partial(
                    time_course,
                    p0=p0,
                    data=data,
                    y0=y0,
                    runner=runner,
                    loss_fn=loss_fn,
                    postprocess_fn=postprocess_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                parallel=parallel,
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def ensemble_protocol_time_course[M: KineticModelBuilder | OdeModelBuilder](
    ensemble: Sequence[M],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    runner: DynamicRunner[M, Any] | None = None,
    loss_fn: LossFn = losses.rmse,
    postprocess_fn: PostprocessFn | None = None,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    parallel: bool = True,
    timeout: float | None = None,
) -> EnsembleFit[M]:
    """Fit model parameters to time course of experimental data over a protocol.

    Time points of protocol time course are taken from the data.

    Examples
    --------
        >>> fit.ensemble_protocol_time_course(
        ...     [
        ...         model_fn(),
        ...         model_fn(),
        ...     ],
        ...     data=res_protocol,
        ...     protocol=protocol,
        ...     p0={"k2": 1.87, "k3": 1.093},  # note that k1 is given by the protocol
        ...     minimizer=fit.LocalScipyMinimizer(tol=1e-6),
        ... )

    Parameters
    ----------
    ensemble
        Model ensemble: value}
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
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    parallel
        Whether to run the fits in parallel (default: True).
    timeout
        Timeout in seconds for each parallel worker

    Returns
    -------
        Ensemble fit object

    """
    protocol = _normalise_protocol_index(protocol)
    return EnsembleFit(
        [
            fit
            for i in parallelise(
                partial(
                    protocol_time_course,
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
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                parallel=parallel,
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )
