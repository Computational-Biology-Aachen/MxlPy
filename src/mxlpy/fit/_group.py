###############################################################################
# Group
# This is single-model, single data fitting, multiple initial parameter guesses
###############################################################################


from collections.abc import Iterable
from functools import partial
from typing import cast

import pandas as pd

from mxlpy import parallel
from mxlpy.fit import losses
from mxlpy.fit._single import protocol_time_course, steady_state, time_course
from mxlpy.fit.abstract import Fit, FitResidual, GroupFit
from mxlpy.fit.residuals import (
    protocol_time_course_residual,
    steady_state_residual,
    time_course_residual,
)
from mxlpy.integrators import IntegratorType
from mxlpy.integrators.utils import OscillationDetector, detect_oscillations
from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
from mxlpy.model import Model
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Result

__all__ = ["group_protocol_time_course", "group_steady_state", "group_time_course"]


def _wrap_steady_state(
    p0: dict[str, float],
    *,
    model: Model,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> Result[Fit]:
    return steady_state(
        model=model,
        p0=p0,
        data=data,
        y0=y0,
        integrator=integrator,
        loss_fn=loss_fn,
        minimizer=minimizer,
        residual_fn=residual_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
        oscillation_detector=oscillation_detector,
    )


def _wrap_time_course(
    p0: dict[str, float],
    *,
    model: Model,
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[Fit]:
    return time_course(
        model=model,
        p0=p0,
        data=data,
        minimizer=minimizer,
        y0=y0,
        residual_fn=residual_fn,
        integrator=integrator,
        loss_fn=loss_fn,
        bounds=bounds,
        as_deepcopy=as_deepcopy,
        standard_scale=standard_scale,
    )


def _wrap_protocol_time_course(
    p0: dict[str, float],
    *,
    model: Model,
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = protocol_time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    standard_scale: bool = True,
) -> Result[Fit]:
    return protocol_time_course(
        model=model,
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
        standard_scale=standard_scale,
    )


def _iterrows(df: pd.DataFrame) -> Iterable[tuple[str, dict[str, float]]]:
    for name, val in df.iterrows():
        yield cast(str, name), cast(dict, val.to_dict())


def group_steady_state(
    model: Model,
    *,
    p0: pd.DataFrame,
    data: pd.Series,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = steady_state_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    timeout: float | None = None,
    oscillation_detector: OscillationDetector | None = detect_oscillations,
) -> GroupFit:
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
        Group fit object

    """
    return GroupFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _wrap_steady_state,
                    model=model,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                    oscillation_detector=oscillation_detector,
                ),
                inputs=list(_iterrows(p0)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def group_time_course(
    model: Model,
    *,
    p0: pd.DataFrame,
    data: pd.DataFrame,
    minimizer: MinimizerProtocol,
    y0: dict[str, float] | None = None,
    residual_fn: FitResidual = time_course_residual,
    integrator: IntegratorType | None = None,
    loss_fn: LossFn = losses.rmse,
    bounds: Bounds | None = None,
    as_deepcopy: bool = True,
    timeout: float | None = None,
) -> GroupFit:
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
        Group fit object

    """
    return GroupFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _wrap_time_course,
                    model=model,
                    data=data,
                    y0=y0,
                    integrator=integrator,
                    loss_fn=loss_fn,
                    minimizer=minimizer,
                    residual_fn=residual_fn,
                    bounds=bounds,
                    as_deepcopy=as_deepcopy,
                ),
                inputs=list(_iterrows(p0)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )


def group_protocol_time_course(
    model: Model,
    *,
    p0: pd.DataFrame,
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
) -> GroupFit:
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
        Group fit object

    """
    protocol = _normalise_protocol_index(protocol)
    return GroupFit(
        [
            fit
            for i in parallel.parallelise(
                partial(
                    _wrap_protocol_time_course,
                    model=model,
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
                inputs=list(_iterrows(p0)),
                timeout=timeout,
            )
            if not isinstance(fit := i[1].value, Exception)
        ]
    )
