# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Staged training of hidden-state blocks (#229).

`augmented_time_course` starts at the fitted mechanistic optimum and trains a
model's hidden-state blocks (`KineticModelBuilder.add_hidden_states`) through
the regular joint fits:

1. scale the blocks along the mechanistic trajectory: inputs by their RMS,
   and the terms of additive blocks by the RMS of their target's mechanistic
   value (a rate for variable targets), so the weights are dimensionless
   (relative terms already are);
2. fit the weights only, with Normal(0, `weight_prior_scale`) priors and a
   smooth penalty on the corrections, so the mechanism keeps explaining as
   much as it can;
3. optionally refine the weights and the mechanistic parameters jointly.

The correction penalty is `correction_penalty * Σ (Δ / s)²` over the data
time points, Δ the contribution of a block to a target and s the RMS of the
target's mechanistic value: zero-valued pseudo-observations of the
contributions under the same Gaussian likelihood as the data
(`GaussianNll(sigma)`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy import fns
from mxlpy.distributions import Normal
from mxlpy.fit._joint import joint_protocol_time_course, joint_time_course
from mxlpy.fit.abstract import Condition, FitSettings, as_conditions
from mxlpy.fit.callbacks import Callback, tagged
from mxlpy.fit.losses import GaussianNll
from mxlpy.fit.penalties import PriorPenalty
from mxlpy.fit.residuals import predicted
from mxlpy.minimizers._scipy import LocalScipyMinimizer
from mxlpy.simulator import Simulator
from mxlpy.types import InitialAssignment, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.distributions import Prior
    from mxlpy.fit.abstract import JointFit
    from mxlpy.integrators import IntegratorType
    from mxlpy.minimizers.abstract import Bounds, MinimizerProtocol
    from mxlpy.runners import DynamicRunner

__all__ = [
    "AugmentedFit",
    "augmented_protocol_time_course",
    "augmented_time_course",
]

type _Model = KineticModelBuilder | OdeModelBuilder
type Sigma = float | pd.Series | pd.DataFrame


@dataclass(kw_only=True)
class AugmentedFit:
    """Result of `augmented_time_course`.

    Attributes
    ----------
    model
        The augmented model with the fitted values. With `fit_h0` and several
        data sets, its hidden states start at the blocks' original `h0`; the
        data sets' own models are in `models`
    best_pars
        Fitted values by name: weights, the values of `p0` (refined or not),
        and the initial values of the hidden states per data set if fitted
    loss
        Negative log-likelihood of the data at the optimum, without the
        correction penalty and the priors
    stages
        The joint fits of the stages, `"weights"` and, if run, `"joint"`
    models
        The fitted model of every data set
    predictions
        Simulated data columns of every data set at its time points
    residuals
        Data minus predictions of every data set
    correction
        Contribution of every block to each of its targets over time, by
        data set (first index level) and time
    scales
        Scale of each block term: the RMS of its target's mechanistic value

    """

    model: _Model
    best_pars: dict[str, float]
    loss: float
    stages: dict[str, JointFit]
    models: dict[str, _Model]
    predictions: dict[str, pd.DataFrame]
    residuals: dict[str, pd.DataFrame]
    correction: pd.DataFrame
    scales: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


def _rms(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    rms = math.sqrt(float(np.mean(arr**2))) if arr.size else 0.0
    return rms if rms > 0 else 1.0


def _simulate(
    model: _Model,
    times: pd.Index,
    protocol: pd.DataFrame | None,
    integrator: IntegratorType | None,
) -> Any:
    """Simulation at the (unique) data time points."""
    unique = pd.Index(np.unique(np.asarray(times, dtype=float)))
    sim = Simulator(model, integrator=integrator)
    if protocol is None:
        sim.simulate_time_course(unique)
    else:
        sim.simulate_protocol_time_course(protocol, unique)
    return sim.get_result().unwrap_or_err()


def _block_terms(model: _Model, blocks: Sequence[str]) -> list[str]:
    """Names of the target terms of `blocks`."""
    raw = model.get_raw_hidden_states(as_copy=False)
    return [f"{b}_{t}" for b in blocks for t in raw[b].targets]


def _condition_model(model: _Model, condition: Condition) -> _Model:
    """A copy of the model with the parameters and initial values of a condition."""
    part = deepcopy(model)
    if condition.parameters:
        part.update_addressable(dict(condition.parameters))
    if condition.y0:
        part.update_variables(dict(condition.y0))
    return part


def _scale_blocks(
    model: _Model,
    blocks: Sequence[str],
    conditions: Mapping[str, Condition],
    protocols: Mapping[str, pd.DataFrame],
    integrator: IntegratorType | None,
) -> dict[str, float]:
    """Scales of the block terms along the mechanistic trajectory.

    Sets the input scales of `blocks` and the output scales of their additive
    targets. Returns the RMS of every target's mechanistic value.
    """
    raw = model.get_raw_hidden_states(as_copy=False)
    readouts = {w for b in blocks for w in raw[b].readout_names(b)}
    weights = model.get_weight_values()
    model.update_addressable(dict.fromkeys(readouts, 0.0))
    try:
        # Results are computed lazily with the parameters of the simulation,
        # so everything is read before the readout is restored
        args, rhs, fluxes = [], [], []
        for name, cond in conditions.items():
            sim = _simulate(
                _condition_model(model, cond),
                cond.data.index,
                protocols.get(name),
                integrator,
            )
            a = cast(
                pd.DataFrame,
                sim.get_args(
                    include_parameters=True,
                    include_derived_parameters=True,
                    include_readouts=True,
                    include_terms=True,
                ),
            )
            a["time"] = a.index.to_numpy(dtype=float)
            args.append(a)
            rhs.append(cast(pd.DataFrame, sim.get_right_hand_side()))
            if hasattr(sim, "get_fluxes"):
                fluxes.append(cast(pd.DataFrame, sim.get_fluxes()))
    finally:
        model.update_addressable({k: weights[k] for k in readouts})
    all_args = pd.concat(args)
    all_rhs = pd.concat(rhs)
    all_fluxes = pd.concat(fluxes) if fluxes else pd.DataFrame()
    scales: dict[str, float] = {}
    for b in blocks:
        block = raw[b]
        target_scales = {
            t: _rms(all_fluxes[t]) if t in all_fluxes else _rms(all_rhs[t])
            for t in block.targets
        }
        model.update_hidden_states(
            b,
            input_scales={
                i: _rms(all_args[i]) if i in all_args else 1.0 for i in block.inputs
            },
            output_scales=target_scales if block.mechanism.name == "additive" else None,
        )
        scales |= {f"{b}_{t}": s for t, s in target_scales.items()}
    return scales


def _local_h0(model: _Model, blocks: Sequence[str], name: str) -> dict[str, float]:
    """Make h(0) of `blocks` a parameter per data set, returns their values."""
    out: dict[str, float] = {}
    raw = model.get_raw_hidden_states(as_copy=False)
    for b in blocks:
        for h in raw[b].state_names(b):
            par = f"{h}_0_{name}"
            out[par] = float(model.get_initial_conditions()[h])
            model.add_parameter(par, out[par])
            model.update_variable(h, InitialAssignment(fn=fns.constant, args=[par]))
    return out


def _sigma_of(
    sigma: Sigma | Mapping[str, Any], name: str, data_sets: Mapping[str, Any]
) -> Sigma:
    """Sigma of one data set: a mapping by data set name, or one for all."""
    if isinstance(sigma, Mapping) and not isinstance(sigma, pd.Series):
        if set(sigma) <= set(data_sets):
            return cast(Sigma, sigma[name])
        return pd.Series(sigma, dtype=float)
    return cast(Sigma, sigma)


def _noise(sigma: Sigma, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(sigma, pd.DataFrame):
        return sigma.reindex(index=data.index, columns=data.columns)
    if isinstance(sigma, pd.Series):
        return sigma.reindex(data.columns)
    return pd.Series(float(sigma), index=data.columns)


def _augmented(
    model: _Model,
    data: pd.DataFrame | Condition | Mapping[str, pd.DataFrame | Condition],
    protocol: pd.DataFrame | Mapping[str, pd.DataFrame] | None,
    *,
    p0: Mapping[str, float],
    sigma: Sigma | Mapping[str, Any],
    blocks: Sequence[str] | None,
    weight_prior_scale: float,
    correction_penalty: float,
    joint_refinement: bool,
    fit_h0: bool,
    rescale: bool,
    minimizer: MinimizerProtocol | None,
    runner: DynamicRunner[Any, Any] | None,
    integrator: IntegratorType | None,
    bounds: Bounds | None,
    prior: Mapping[str, Prior] | None,
    callback: Callback | None,
) -> Result[AugmentedFit]:
    conditions = as_conditions(data)
    if not all(isinstance(c.data, pd.DataFrame) for c in conditions.values()):
        msg = "Hidden-state blocks are trained on time courses (DataFrames)"
        raise TypeError(msg)
    data_sets = {k: cast(pd.DataFrame, c.data) for k, c in conditions.items()}
    protocols = {k: c.protocol for k, c in conditions.items() if c.protocol is not None}
    if isinstance(protocol, pd.DataFrame):
        protocols = dict.fromkeys(data_sets, protocol) | protocols
    elif protocol is not None:
        given = dict(protocol)
        if unknown := sorted(set(given) - set(data_sets)):
            msg = f"Protocols for unknown data sets {unknown}"
            raise KeyError(msg)
        protocols = given | protocols
        if missing := sorted(set(data_sets) - set(protocols)):
            msg = f"No protocol for the data sets {missing}"
            raise KeyError(msg)
    if protocols and (missing := sorted(set(data_sets) - set(protocols))):
        msg = f"Some data sets have a protocol, but not {missing}"
        raise KeyError(msg)
    if renamed := sorted({k for c in conditions.values() for k in (c.local or {})}):
        msg = f"Condition-specific names (local) aren't supported here, set {renamed} as parameters"
        raise ValueError(msg)
    if refined := sorted(
        {k for c in conditions.values() for k in c.overridden if k in p0}
    ):
        msg = f"Parameters set by a condition can't be refined: {refined}"
        raise ValueError(msg)
    model = deepcopy(model)
    raw = model.get_raw_hidden_states(as_copy=False)
    blocks = list(raw) if blocks is None else list(blocks)
    if not blocks:
        msg = "The model has no hidden-state blocks to train"
        raise ValueError(msg)
    if unknown := sorted(set(blocks) - set(raw)):
        msg = f"Unknown hidden-state blocks {unknown}"
        raise KeyError(msg)
    states = set(model.get_variable_names())
    if unknown := sorted(set(p0) - set(model.get_parameter_names()) - states):
        msg = f"p0 can only hold parameters and initial values, not {unknown}"
        raise KeyError(msg)
    model.update_addressable({k: v for k, v in p0.items() if k not in states})
    model.update_variables({k: v for k, v in p0.items() if k in states})
    minimizer = (
        LocalScipyMinimizer(method="L-BFGS-B") if minimizer is None else minimizer
    )

    terms = _block_terms(model, blocks)
    if rescale:
        scales = _scale_blocks(model, blocks, conditions, protocols, integrator)
    else:
        scales = {
            f"{b}_{t}": float(s)
            for b in blocks
            for t, s in zip(raw[b].targets, raw[b].output_scales, strict=True)
        }
    weights = [w for b in blocks for w in raw[b].weight_names(b)]

    settings: list[FitSettings[Any]] = []
    noises: dict[str, pd.Series | pd.DataFrame] = {}
    local: dict[str, float] = {}
    for name, df in data_sets.items():
        cond = conditions[name]
        part = _condition_model(model, cond)
        if fit_h0:
            local |= _local_h0(part, blocks, name)
        observed = df.copy()
        noise = noises[name] = _noise(
            _sigma_of(sigma, name, data_sets) if cond.sigma is None else cond.sigma,
            df,
        )
        if correction_penalty > 0:
            penalty_sigma = {
                t: scales[t] / math.sqrt(2 * correction_penalty) for t in terms
            }
            for t in terms:
                observed[t] = 0.0
            if isinstance(noise, pd.DataFrame):
                noise = noise.assign(**penalty_sigma)
            else:
                noise = pd.concat([noise, pd.Series(penalty_sigma)])
        settings.append(
            FitSettings(
                model=part,
                data=observed,
                loss_fn=GaussianNll(sigma=noise),
                protocol=protocols.get(name),
                runner=runner,
            )
        )

    priors: dict[str, Prior] = dict(prior or {}) | {
        w: Normal(0.0, weight_prior_scale) for w in weights
    }
    values = model.get_addressable_values() | model.get_initial_conditions()

    def stage(label: str, names: list[str]) -> Result[JointFit]:
        start = {k: float(values[k]) for k in names if k not in local} | {
            k: local[k] for k in names if k in local
        }
        kwargs: dict[str, Any] = {
            "p0": start,
            "minimizer": minimizer,
            "penalty": PriorPenalty({k: v for k, v in priors.items() if k in start}),
            "bounds": bounds,
            "parallel": False,
            "standard_scale": False,
            "callback": tagged(callback, label),
        }
        if protocols:
            return joint_protocol_time_course(settings, **kwargs)
        return joint_time_course(settings, **kwargs)

    stages: dict[str, JointFit] = {}
    if isinstance(fit := stage("weights", weights + list(local)).value, Exception):
        return Result(fit)
    stages["weights"] = fit
    values |= {k: v for k, v in fit.best_pars.items() if k not in local}
    local |= {k: v for k, v in fit.best_pars.items() if k in local}
    if joint_refinement and p0:
        if isinstance(fit := stage("joint", [*p0, *weights, *local]).value, Exception):
            return Result(fit)
        stages["joint"] = fit
        values |= {k: v for k, v in fit.best_pars.items() if k not in local}
        local |= {k: v for k, v in fit.best_pars.items() if k in local}

    fitted = {k: float(values[k]) for k in [*p0, *weights]}
    model.update_addressable({k: v for k, v in fitted.items() if k not in states})
    model.update_variables({k: v for k, v in fitted.items() if k in states})
    models: dict[str, _Model] = {}
    predictions: dict[str, pd.DataFrame] = {}
    residuals: dict[str, pd.DataFrame] = {}
    corrections = []
    loss = 0.0
    for (name, df), s in zip(data_sets.items(), settings, strict=True):
        part = s.model
        names = set(part.get_addressable_names())
        part.update_addressable(
            {k: v for k, v in (fitted | local).items() if k in names}
        )
        part.update_variables({k: v for k, v in fitted.items() if k in states})
        models[name] = part
        sim = _simulate(part, df.index, protocols.get(name), integrator)
        combined = predicted(sim, [*df.columns, *terms])
        prediction = combined.reindex(np.asarray(df.index, dtype=float)).loc[
            :, list(df.columns)
        ]
        prediction.index = df.index
        predictions[name] = prediction
        residuals[name] = df - prediction
        loss += GaussianNll(sigma=noises[name])(prediction, df)
        corrections.append(combined.loc[:, terms])
    correction = pd.concat(corrections, keys=list(data_sets), names=["data", "time"])
    if fit_h0 and len(data_sets) == 1:
        # one data set: its h(0) is the model's
        name = next(iter(data_sets))
        for b in blocks:
            block = model.get_raw_hidden_states(as_copy=False)[b]
            model.update_hidden_states(
                b, h0=[local[f"{h}_0_{name}"] for h in block.state_names(b)]
            )
    return Result(
        AugmentedFit(
            model=model,
            best_pars=fitted | local,
            loss=loss,
            stages=stages,
            models=models,
            predictions=predictions,
            residuals=residuals,
            correction=correction,
            scales=scales,
        )
    )


def augmented_time_course(
    model: _Model,
    *,
    data: pd.DataFrame | Condition | Mapping[str, pd.DataFrame | Condition],
    p0: Mapping[str, float],
    sigma: Sigma | Mapping[str, Any] = 1.0,
    blocks: Sequence[str] | None = None,
    weight_prior_scale: float = 0.1,
    correction_penalty: float = 1e-2,
    joint_refinement: bool = True,
    fit_h0: bool = False,
    rescale: bool = True,
    minimizer: MinimizerProtocol | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
    integrator: IntegratorType | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    callback: Callback | None = None,
) -> Result[AugmentedFit]:
    """Train the hidden-state blocks of a model on time courses.

    See the module docstring for the stages. The model isn't modified.

    Examples
    --------
        >>> model.add_hidden_states("aug", n_hidden=1, targets=["S"],
        ...                         mechanism="additive")
        >>> res = fit.augmented_time_course(
        ...     model, data=data, p0=mechanistic_fit.best_pars, sigma=0.05
        ... ).unwrap_or_err()
        >>> res.correction          # when and where dynamics were missing

    Parameters
    ----------
    model
        Model with hidden-state blocks
    data
        Time course, or time courses (or `fit.Condition`s with their own
        initial values, parameters, protocol and sigma) by data set name, all
        fitted jointly with the same weights. Parameters a condition sets
        can't be in `p0`, and conditions can't have condition-specific
        (`local`) names
    p0
        Fitted mechanistic parameters (and initial values), the start
    sigma
        Measurement noise of the data: scalar, per observable (Series or
        mapping) or per data point (DataFrame), or a mapping of these by
        data set name. The sigma of a `fit.Condition` takes precedence
    blocks
        Blocks to train, all by default; the weights of the others are fixed
    weight_prior_scale
        Standard deviation of the Normal(0, s) prior of every weight
    correction_penalty
        Weight of the penalty on the corrections, 0 for none
    joint_refinement
        Whether to refine weights and the values in `p0` jointly (skipped
        for an empty `p0`)
    fit_h0
        Whether to fit the initial values of the hidden states, one per data
        set, instead of starting at the blocks' `h0`
    rescale
        Whether to set the blocks' input and output scales from the
        mechanistic trajectory first
    minimizer
        Minimizer, L-BFGS-B by default
    runner
        Runner of the fits, e.g. to choose the integrator and derivatives
    integrator
        Integrator of the simulations for the scales and the results
    bounds
        Bounds of the fitted values; weights default to signed bounds
    prior
        Priors of the values in `p0`, used in the joint refinement
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every stage, whose `start` is the stage, see `mxlpy.fit.callbacks`

    Returns
    -------
    Result[AugmentedFit]
        The fitted model, its values, the stages, the corrections and the
        per-data-set models, predictions and residuals

    """
    return _augmented(
        model,
        data,
        None,
        p0=p0,
        sigma=sigma,
        blocks=blocks,
        weight_prior_scale=weight_prior_scale,
        correction_penalty=correction_penalty,
        joint_refinement=joint_refinement,
        fit_h0=fit_h0,
        rescale=rescale,
        minimizer=minimizer,
        runner=runner,
        integrator=integrator,
        bounds=bounds,
        prior=prior,
        callback=callback,
    )


def augmented_protocol_time_course(
    model: _Model,
    *,
    data: pd.DataFrame | Condition | Mapping[str, pd.DataFrame | Condition],
    protocol: pd.DataFrame | Mapping[str, pd.DataFrame],
    p0: Mapping[str, float],
    sigma: Sigma | Mapping[str, Any] = 1.0,
    blocks: Sequence[str] | None = None,
    weight_prior_scale: float = 0.1,
    correction_penalty: float = 1e-2,
    joint_refinement: bool = True,
    fit_h0: bool = False,
    rescale: bool = True,
    minimizer: MinimizerProtocol | None = None,
    runner: DynamicRunner[Any, Any] | None = None,
    integrator: IntegratorType | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    callback: Callback | None = None,
) -> Result[AugmentedFit]:
    """Train the hidden-state blocks of a model on protocol time courses.

    Like `augmented_time_course`, with a protocol (one for all data sets or
    one per data set), see `mxlpy.Simulator.simulate_protocol_time_course`.

    Parameters
    ----------
    model
        Model with hidden-state blocks
    data
        Time course, or time courses (or `fit.Condition`s) by data set name
    protocol
        Protocol, or protocols by data set name
    p0
        Fitted mechanistic parameters (and initial values), the start
    sigma
        Measurement noise of the data, see `augmented_time_course`
    blocks
        Blocks to train, all by default
    weight_prior_scale
        Standard deviation of the Normal(0, s) prior of every weight
    correction_penalty
        Weight of the penalty on the corrections, 0 for none
    joint_refinement
        Whether to refine weights and the values in `p0` jointly
    fit_h0
        Whether to fit the initial values of the hidden states per data set
    rescale
        Whether to set the blocks' input and output scales first
    minimizer
        Minimizer, L-BFGS-B by default
    runner
        Runner of the fits
    integrator
        Integrator of the simulations for the scales and the results
    bounds
        Bounds of the fitted values
    prior
        Priors of the values in `p0`, used in the joint refinement
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every stage, whose `start` is the stage, see `mxlpy.fit.callbacks`

    Returns
    -------
    Result[AugmentedFit]
        The fitted model, its values, the stages and the corrections

    """
    return _augmented(
        model,
        data,
        protocol,
        p0=p0,
        sigma=sigma,
        blocks=blocks,
        weight_prior_scale=weight_prior_scale,
        correction_penalty=correction_penalty,
        joint_refinement=joint_refinement,
        fit_h0=fit_h0,
        rescale=rescale,
        minimizer=minimizer,
        runner=runner,
        integrator=integrator,
        bounds=bounds,
        prior=prior,
        callback=callback,
    )
