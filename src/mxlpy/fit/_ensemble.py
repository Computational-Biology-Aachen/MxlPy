###############################################################################
# Ensemble / carousel
# This is multi-model, single data fitting, where the models share parameters
###############################################################################


from functools import partial

import pandas as pd

from mxlpy import parallel
from mxlpy.fit import losses
from mxlpy.fit._single import protocol_time_course, steady_state, time_course
from mxlpy.fit.abstract import EnsembleFit, FitResidual
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.integrators import IntegratorType
from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
from mxlpy.model import Model
from mxlpy.simulator import _normalise_protocol_index

__all__ = [
    "ensemble_protocol_time_course",
    "ensemble_steady_state",
    "ensemble_time_course",
]


def ensemble_steady_state(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    timeout: float | None = None,
) -> EnsembleFit:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    timeout
        Timeout in seconds for each parallel worker

    Returns
    -------
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    steady_state,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def ensemble_time_course(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    timeout: float | None = None,
) -> EnsembleFit:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
    timeout
        Timeout in seconds for each parallel worker

    Returns
    -------
        Ensemble fit object

    """
    return EnsembleFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    time_course,
                    p0=p0,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def ensemble_protocol_time_course(
    ensemble: list[Model],
    *,
    p0: dict[str, float],
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    protocol: pd.DataFrame,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    timeout: float | None = None,
) -> EnsembleFit:
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
    integrator
        ODE integrator class
    loss_fn
        Loss function to use for residual calculation
    time_points_per_step
        Number of time points per step in the protocol
    bounds
        Mapping of bounds per parameter
    as_deepcopy
        Whether to copy the model to avoid overwriting the state
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
            for i in parallel.parallelise(
                partial(
                    protocol_time_course,
                    p0=p0,
                    data=data,
                    protocol=protocol,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(enumerate(ensemble)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )
