"""Training loop utilities for JAX ODE and neural ODE models."""

import math
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange

from mxlpy.jax.models import JaxModel, Method, Ude

__all__ = [
    "IntegrationSettings",
    "grad_loss",
    "grad_loss_split",
    "make_step",
    "make_step_split",
    "train",
    "train_only_nde",
]


@dataclass(unsafe_hash=True)
class IntegrationSettings:
    """ODE solver settings shared across training steps.

    Parameters
    ----------
    atol : float
        Absolute tolerance for the adaptive step-size controller.
    rtol : float
        Relative tolerance for the adaptive step-size controller.
    method : Method
        Diffrax Runge-Kutta solver class (default: Tsit5).
    max_steps : int
        Maximum number of solver steps before aborting.
    """

    atol: float = 1e-6
    rtol: float = 1e-4
    method: Method = diffrax.Tsit5
    max_steps: int = 8192


@eqx.filter_value_and_grad
def grad_loss(
    model: JaxModel,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
) -> jax.Array:
    """Compute normalised MSE loss and its gradient w.r.t. model parameters.

    Parameters
    ----------
    model : JaxModel
        Model to differentiate.
    ts : jax.Array
        Time points, shape ``(T,)``.
    ys : jax.Array
        Observed trajectories, shape ``(T, n_obs)``.
    y_mean : jax.Array
        Mean used for normalisation.
    y_scale : jax.Array
        Standard deviation used for normalisation.
    ctx : IntegrationSettings
        ODE solver settings.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree.
    """
    y_pred = model.integrate(
        ts,
        ys[0],
        max_steps=ctx.max_steps,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


@eqx.filter_value_and_grad
def grad_loss_split(
    trainable: JaxModel,
    frozen: JaxModel,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
) -> jax.Array:
    """Compute normalised MSE loss and gradient for a partitioned model.

    Parameters
    ----------
    trainable : JaxModel
        Trainable partition; gradients are computed w.r.t. this.
    frozen : JaxModel
        Frozen partition; recombined with ``trainable`` before evaluation.
    ts : jax.Array
        Time points, shape ``(T,)``.
    ys : jax.Array
        Observed trajectories, shape ``(T, n_obs)``.
    y_mean : jax.Array
        Mean used for normalisation.
    y_scale : jax.Array
        Standard deviation used for normalisation.
    ctx : IntegrationSettings
        ODE solver settings.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree w.r.t. ``trainable``.
    """
    model = eqx.combine(trainable, frozen)
    y_pred = model.integrate(
        ts,
        ys[0],
        max_steps=ctx.max_steps,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


@eqx.filter_jit
def make_step[T: JaxModel](
    *,
    model: T,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
) -> tuple[jax.Array, T, optax.OptState]:
    """Perform one gradient update step on the full model.

    NaN gradients (from failed ODE solves) are zeroed before the update.
    Gradients are clipped by global norm to guard against exploding adjoints.

    Parameters
    ----------
    model : T
        Current model state.
    ts : jax.Array
        Time points.
    ys : jax.Array
        Observed trajectories.
    y_mean : jax.Array
        Mean for normalisation.
    y_scale : jax.Array
        Scale for normalisation.
    opt_state : optax.OptState
        Current optimiser state.
    optim : optax.GradientTransformationExtraArgs
        Optimiser.
    ctx : IntegrationSettings
        ODE solver settings.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState]
        Scalar loss, updated model, updated optimiser state.
    """
    loss, grads = grad_loss(
        model,  # needs to be by pos, is differentiated
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
    )
    # Failed solves can produce NaN in y_pred -> NaN gradients; zero them out
    # so the optimizer state isn't corrupted.
    grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)

    # Clip gradients to prevent sudden entry into stiff regions and exploding adjoints
    grad_norm = optax.global_norm(grads)
    clip_value = 1.0
    grads = jax.tree_util.tree_map(
        lambda g: g * (clip_value / (grad_norm + 1e-6)), grads
    )
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    return loss, eqx.apply_updates(model, updates), opt_state


@eqx.filter_jit
def make_step_split[T: JaxModel](
    *,
    trainable: T,
    frozen: T,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
) -> tuple[jax.Array, T, optax.OptState]:
    """Perform one gradient update step on the trainable model partition.

    Parameters
    ----------
    trainable : T
        Trainable model partition to update.
    frozen : T
        Frozen model partition (unchanged).
    ts : jax.Array
        Time points.
    ys : jax.Array
        Observed trajectories.
    y_mean : jax.Array
        Mean for normalisation.
    y_scale : jax.Array
        Scale for normalisation.
    opt_state : optax.OptState
        Current optimiser state.
    optim : optax.GradientTransformationExtraArgs
        Optimiser.
    ctx : IntegrationSettings
        ODE solver settings.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState]
        Scalar loss, updated trainable partition, updated optimiser state.
    """
    loss, grads = grad_loss_split(
        trainable,  # needs to be by pos, is differentiated
        frozen=frozen,
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
    )
    # Failed solves can produce NaN in y_pred -> NaN gradients; zero them out
    # so the optimizer state isn't corrupted.
    # grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)

    # Clip gradients to prevent sudden entry into stiff regions and exploding adjoints
    # grad_norm = optax.global_norm(grads)
    # clip_value = 1.0
    # grads = jax.tree_util.tree_map(
    #     lambda g: g * (clip_value / (grad_norm + 1e-6)), grads
    # )
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(trainable, eqx.is_array),
    )
    return loss, eqx.apply_updates(trainable, updates), opt_state


def train[T: JaxModel](
    model: T,
    *,
    ts: jax.Array,
    ys: jax.Array,
    training_steps: list[tuple[int, float]],
    target_loss: float = 1e-4,
    avg_every: int = 1000,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
) -> tuple[T, dict[int, float]]:
    """Train a JAX model through a sequence of curriculum steps.

    Each curriculum stage re-initialises the optimiser and uses a
    different fraction of the data.  Early stopping triggers when
    ``loss < target_loss`` during the **last** stage.

    Parameters
    ----------
    model : T
        Initial model state.
    ts : jax.Array
        Full time-point array.
    ys : jax.Array
        Full observed trajectory, shape ``(T, n_obs)``.
    training_steps : list[tuple[int, float]]
        Sequence of ``(n_steps, data_fraction)`` pairs.
    target_loss : float
        Early-stopping threshold, checked during the last stage only.
    avg_every : int
        Average and log loss every this many steps.
    optim : optax.GradientTransformationExtraArgs or None
        Optimiser; defaults to AdaBelief with ``lr=1e-4``.
    integration_settings : IntegrationSettings or None
        ODE solver settings; defaults to ``IntegrationSettings()``.

    Returns
    -------
    tuple[T, dict[int, float]]
        Best model encountered during training and a dict mapping
        global step index to average loss.
    """
    loss = 0
    acc_loss = 0
    acc_count = 0
    global_step = 0
    optim = optax.adabelief(learning_rate=1e-4) if optim is None else optim
    best_training_loss = jnp.inf
    best_model = model

    ctx = (
        IntegrationSettings() if integration_settings is None else integration_settings
    )

    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses = {}
    for i, (steps, frac) in enumerate(training_steps, start=1):
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        length = math.ceil(len(ts) * frac)
        _ts = ts[:length]
        _ys = ys[:length]

        with trange(steps, position=1, leave=True, dynamic_ncols=True) as pbar:
            for step in pbar:
                loss, model, opt_state = make_step(
                    model=model,
                    ts=_ts,
                    ys=_ys,
                    opt_state=opt_state,
                    optim=optim,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    ctx=ctx,
                )
                acc_loss += loss
                acc_count += 1

                if loss < best_training_loss:
                    best_model = model
                    best_training_loss = loss

                if i == len(training_steps) and loss < target_loss:
                    return (model, losses)

                if (step % avg_every) == 0 or step == steps - 1:
                    avg_loss = acc_loss / acc_count
                    pbar.set_postfix_str(
                        f"Avg. loss {(avg_loss):.2e} over last {acc_count} runs"
                    )
                    global_step += acc_count
                    acc_loss = 0
                    acc_count = 0
                    losses[global_step] = float(avg_loss)
    return best_model, losses


def train_only_nde[T: Ude](
    model: T,
    *,
    ts: jax.Array,
    ys: jax.Array,
    training_steps: list[tuple[int, float]],
    avg_every: int = 1000,
    target_loss: float = 1e-4,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
) -> tuple[T, dict[int, float]]:
    """Train only the neural-network part of a UDE, keeping the ODE frozen.

    Partitions ``model`` into trainable (neural network) and frozen (ODE)
    parts before the training loop.  The returned model recombines both.

    Parameters
    ----------
    model : T
        Initial UDE model.
    ts : jax.Array
        Full time-point array.
    ys : jax.Array
        Full observed trajectory, shape ``(T, n_obs)``.
    training_steps : list[tuple[int, float]]
        Sequence of ``(n_steps, data_fraction)`` pairs.
    avg_every : int
        Average and log loss every this many steps.
    target_loss : float
        Early-stopping threshold, checked during the last stage only.
    optim : optax.GradientTransformationExtraArgs or None
        Optimiser; defaults to AdaBelief with ``lr=1e-4``.
    integration_settings : IntegrationSettings or None
        ODE solver settings; defaults to ``IntegrationSettings()``.

    Returns
    -------
    tuple[T, dict[int, float]]
        Best model encountered during training and a dict mapping
        global step index to average loss.
    """
    loss = 0
    acc_loss = 0
    acc_count = 0
    global_step = 0
    optim = optax.adabelief(learning_rate=1e-4) if optim is None else optim
    best_training_loss = jnp.inf
    best_model = model

    ctx = (
        IntegrationSettings() if integration_settings is None else integration_settings
    )

    filter_spec = jax.tree.map(eqx.is_inexact_array, model)
    filter_spec = eqx.tree_at(
        lambda m: m.ode,
        filter_spec,
        replace_fn=lambda _: False,
    )
    trainable, frozen = eqx.partition(model, filter_spec)

    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses = {}
    for i, (steps, frac) in enumerate(training_steps):
        opt_state = optim.init(trainable)
        length = math.ceil(len(ts) * frac)
        _ts = ts[:length]
        _ys = ys[:length]

        with trange(steps, position=1, leave=True, dynamic_ncols=True) as pbar:
            for step in pbar:
                loss, trainable, opt_state = make_step_split(
                    trainable=trainable,
                    frozen=frozen,
                    ts=_ts,
                    ys=_ys,
                    opt_state=opt_state,
                    optim=optim,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    ctx=ctx,
                )
                acc_loss += loss
                acc_count += 1

                if i == len(training_steps) and loss < best_training_loss:
                    best_model = eqx.combine(trainable, frozen)
                    best_training_loss = loss

                if i == len(training_steps) and loss < target_loss:
                    return (eqx.combine(trainable, frozen), losses)

                if (step % avg_every) == 0 or step == steps - 1:
                    avg_loss = acc_loss / acc_count
                    pbar.set_postfix_str(
                        f"Avg. loss {(avg_loss):.2e} over last {acc_count} runs"
                    )
                    global_step += acc_count
                    acc_loss = 0
                    acc_count = 0
                    losses[global_step] = float(avg_loss)
    return best_model, losses
