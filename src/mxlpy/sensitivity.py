"""Global sensitivity analysis for mechanistic models.

This module provides screening and variance-based methods to identify which
parameters drive a model output, before committing to expensive analyses such
as profile likelihood, fitting, or full Sobol decomposition.

The output of a model is supplied as a callable that maps the model and a
DataFrame of parameter samples (one row per sample, one column per parameter)
to a one-dimensional array of scalar outputs.  This contract lets the reduction
re-use :mod:`mxlpy.scan`, which evaluates the samples in parallel, caches
results, and already returns ``NaN`` for samples whose integration failed.

Examples
--------
    >>> import numpy as np
    >>> from mxlpy import scan, sensitivity
    >>>
    >>> def steady_state_atp(model, samples):
    ...     return scan.steady_state(
    ...         model, to_scan=samples
    ...     ).variables["ATP"].to_numpy()
    >>>
    >>> sensitivity.morris(
    ...     model,
    ...     output=steady_state_atp,
    ...     param_bounds={"k1": (0.1, 10.0), "k2": (0.01, 1.0)},
    ...     n_trajectories=20,
    ...     seed=0,
    ... )

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from SALib.analyze.morris import analyze as _morris_analyze
from SALib.analyze.sobol import analyze as _sobol_analyze
from SALib.sample.morris import sample as _morris_sample
from SALib.sample.sobol import sample as _sobol_sample

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mxlpy.model import Model
    from mxlpy.types import Array

__all__ = ["OutputFn", "morris", "sobol"]

_LOGGER = logging.getLogger(__name__)

type OutputFn = Callable[[Model, pd.DataFrame], Array]


def _build_problem(param_bounds: Mapping[str, tuple[float, float]]) -> dict:
    """Build a SALib ``problem`` dict from parameter bounds.

    Parameters
    ----------
    param_bounds
        Mapping of parameter name to its ``(lower, upper)`` sampling bounds.

    Returns
    -------
    dict
        SALib problem definition with ``num_vars``, ``names`` and ``bounds``.

    """
    names = list(param_bounds)
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": [list(param_bounds[name]) for name in names],
    }


def _evaluate(
    model: Model,
    output: OutputFn,
    sample_matrix: Array,
    names: list[str],
) -> Array:
    """Evaluate ``output`` over a SALib sample matrix.

    Parameters
    ----------
    model
        Model instance to evaluate.
    output
        Callable mapping ``(model, samples)`` to a 1-D array of outputs.
    sample_matrix
        SALib sample matrix of shape ``(n_samples, n_params)``.
    names
        Parameter names, aligned to the columns of ``sample_matrix``.

    Returns
    -------
    Array
        One-dimensional array of outputs, one per sample.

    """
    samples = pd.DataFrame(sample_matrix, columns=names)
    y = np.asarray(output(model, samples), dtype=float).ravel()

    n_failed = int(np.count_nonzero(~np.isfinite(y)))
    if n_failed > 0:
        _LOGGER.warning(
            "%d of %d output evaluations were non-finite; "
            "sensitivity indices for affected parameters may be NaN.",
            n_failed,
            y.size,
        )
    return y


def morris(
    model: Model,
    *,
    output: OutputFn,
    param_bounds: Mapping[str, tuple[float, float]],
    n_trajectories: int = 10,
    num_levels: int = 4,
    seed: int | None = None,
) -> pd.DataFrame:
    """Screen parameters with the Morris elementary-effects method.

    The Morris method identifies which parameters matter using only
    ``n_trajectories * (k + 1)`` model evaluations (``k`` parameters).  It
    separates parameters that are important and nonlinear (high ``sigma``)
    from important and linear (high ``mu_star``, low ``sigma``) from
    unimportant (low ``mu_star``).

    Parameters
    ----------
    model
        Model instance to analyse.
    output
        Callable mapping ``(model, samples)`` to a one-dimensional array of
        scalar outputs, one per sample row.  Typically built on
        :mod:`mxlpy.scan`, e.g.
        ``lambda m, s: scan.steady_state(m, to_scan=s).variables["ATP"].to_numpy()``.
    param_bounds
        Mapping of parameter name to its ``(lower, upper)`` sampling bounds.
    n_trajectories
        Number of Morris trajectories (``N``).  Total evaluations are
        ``n_trajectories * (len(param_bounds) + 1)``.
    num_levels
        Number of grid levels in the Morris sampling design.
    seed
        Seed for the sampler and the analysis bootstrap, for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sensitivity indices indexed by parameter name, with columns
        ``mu`` (mean elementary effect), ``mu_star`` (mean absolute
        elementary effect), ``sigma`` (standard deviation of effects) and
        ``mu_star_conf`` (bootstrap confidence interval on ``mu_star``).

    Examples
    --------
    >>> from mxlpy import scan, sensitivity
    >>> def output(model, samples):
    ...     return scan.steady_state(
    ...         model, to_scan=samples
    ...     ).variables["v1"].to_numpy()
    >>> sensitivity.morris(
    ...     model,
    ...     output=output,
    ...     param_bounds={"p1": (0.1, 10.0)},
    ...     seed=0,
    ... )

    """
    problem = _build_problem(param_bounds)
    sample_matrix = _morris_sample(
        problem,
        N=n_trajectories,
        num_levels=num_levels,
        seed=seed,
    )
    y = _evaluate(model, output, sample_matrix, problem["names"])

    result: Any = _morris_analyze(
        problem,
        sample_matrix,
        y,
        num_levels=num_levels,
        seed=seed,
    )
    return cast("pd.DataFrame", result.to_df())


def sobol(
    model: Model,
    *,
    output: OutputFn,
    param_bounds: Mapping[str, tuple[float, float]],
    n_samples: int = 1024,
    seed: int | None = None,
) -> pd.DataFrame:
    """Quantify parameter influence with variance-based Sobol indices.

    The Sobol method decomposes the variance of a model output into the
    contributions of each parameter.  Unlike Morris screening, which only
    ranks parameters, Sobol quantifies *how much* of the output variance each
    parameter explains (``S1``, first-order) and how much it explains
    including all of its interactions (``ST``, total-order).  A large gap
    ``ST - S1`` flags a parameter that matters mostly through interactions.

    The cost is ``n_samples * (k + 2)`` model evaluations for ``k``
    parameters, so screen with :func:`morris` first and quantify only the
    survivors with :func:`sobol`.

    Parameters
    ----------
    model
        Model instance to analyse.
    output
        Callable mapping ``(model, samples)`` to a one-dimensional array of
        scalar outputs, one per sample row.  Typically built on
        :mod:`mxlpy.scan`, e.g.
        ``lambda m, s: scan.steady_state(m, to_scan=s).variables["ATP"].to_numpy()``.
    param_bounds
        Mapping of parameter name to its ``(lower, upper)`` sampling bounds.
    n_samples
        Base sample size of the Saltelli design.  Must be a power of two.
        Total evaluations are ``n_samples * (len(param_bounds) + 2)``.
    seed
        Seed for the sampler and the analysis bootstrap, for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sensitivity indices indexed by parameter name, with columns ``S1``
        (first-order index), ``ST`` (total-order index), ``S1_conf`` and
        ``ST_conf`` (bootstrap confidence intervals).

    Raises
    ------
    ValueError
        If ``n_samples`` is not a positive power of two.

    Examples
    --------
    >>> from mxlpy import scan, sensitivity
    >>> def output(model, samples):
    ...     return scan.steady_state(
    ...         model, to_scan=samples
    ...     ).variables["v1"].to_numpy()
    >>> sensitivity.sobol(
    ...     model,
    ...     output=output,
    ...     param_bounds={"p1": (0.1, 10.0)},
    ...     n_samples=512,
    ...     seed=0,
    ... )

    """
    if n_samples <= 0 or n_samples & (n_samples - 1) != 0:
        msg = f"n_samples must be a positive power of two, got {n_samples}."
        raise ValueError(msg)

    problem = _build_problem(param_bounds)
    sample_matrix = _sobol_sample(
        problem,
        N=n_samples,
        calc_second_order=False,
        seed=seed,
    )
    y = _evaluate(model, output, sample_matrix, problem["names"])

    result: Any = _sobol_analyze(
        problem,
        y,
        calc_second_order=False,
        seed=seed,
    )
    total_df, first_df = result.to_df()
    merged = pd.concat([first_df, total_df], axis=1)
    return cast("pd.DataFrame", merged[["S1", "ST", "S1_conf", "ST_conf"]])
