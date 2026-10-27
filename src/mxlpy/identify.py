# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numerical parameter identification estimations."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from mxlpy import fit
from mxlpy.distributions import LogNormal, sample
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy.runners import DynamicRunner
    from mxlpy.simulation import Simulation
    from mxlpy.types import Array

__all__ = [
    "profile_likelihood",
]


def _mc_fit_time_course_worker(
    p0: pd.Series,
    model: KineticModelBuilder,
    data: pd.DataFrame,
    loss_fn: fit.LossFn,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None,
) -> float:
    fit_result = fit.time_course(
        model=model,
        p0=cast(dict, p0.to_dict()),
        data=data,
        loss_fn=loss_fn,
        minimizer=fit.LocalScipyMinimizer(),
        runner=runner,
    )
    match fit_result.value:
        case fit.Fit(_, _, loss):
            return loss
        case _:
            return np.inf


def profile_likelihood(
    model: KineticModelBuilder,
    data: pd.DataFrame,
    parameter_name: str,
    parameter_values: Array,
    n_random: int = 10,
    loss_fn: fit.LossFn = fit.rmse,
    *,
    runner: DynamicRunner[KineticModelBuilder, Simulation] | None = None,
) -> pd.Series:
    """Estimate the profile likelihood of model parameters given data.

    Parameters
    ----------
    model
        The model to be fitted.
    data
        The data to fit the model to.
    parameter_name
        The name of the parameter to profile.
    parameter_values
        The values of the parameter to profile.
    n_random
        Number of Monte Carlo samples.
    loss_fn
        Loss function to use for fitting.
    runner
        How the model is simulated, see `mxlpy.runners`.
        Defaults to `KineticRunner()`.

    """
    parameter_distributions = sample(
        {
            k: LogNormal(np.log(v), sigma=1)
            for k, v in model.get_parameter_values().items()
        },
        n=n_random,
    )

    model = deepcopy(model)
    res = {}
    for value in tqdm(parameter_values, desc=parameter_name):
        model.update_parameter(parameter_name, value)
        res[value] = dict(
            parallelise(
                partial(
                    _mc_fit_time_course_worker,
                    model=model,
                    data=data,
                    loss_fn=loss_fn,
                    runner=runner,
                ),
                inputs=list(
                    parameter_distributions.drop(columns=parameter_name).iterrows()
                ),
                disable_tqdm=True,
            )
        )
    errors = pd.DataFrame(res, dtype=float).T.abs().mean(axis=1)
    errors.index.name = "fitting error"
    return errors
