"""Training loop utilities for JAX ODE and neural ODE models."""

import logging
import math
from dataclasses import dataclass

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from tqdm.auto import trange

from mxlpy.jax.models import JaxModel, Method, Ude

__all__ = [
    "GradNormsPerLesson",
    "IntegrationSettings",
    "LossesPerLesson",
    "grad_loss",
    "grad_loss_split",
    "make_step",
    "make_step_split",
    "squeeze_derived",
    "train",
    "train_only_nde",
]

logger = logging.getLogger(__name__)

type LossesPerLesson = list[pd.Series]
type GradNormsPerLesson = list[pd.Series]


def squeeze_derived(derived: jax.Array) -> jax.Array:
    """Collapse a trailing length-1 axis from ``JaxModel.derived()``'s output.

    ``derived()`` always returns shape ``(T, n_derived)``; for single-quantity
    models (``n_derived == 1``) that trailing axis would otherwise silently
    broadcast a ``(T, 1)`` prediction against a ``(T,)`` data series into a
    ``(T, T)`` pairwise loss instead of a per-timepoint one. For
    ``n_derived > 1`` there's no such axis to squeeze, so ``derived`` is
    returned unchanged.
    """
    if derived.shape[-1] == 1:
        return derived[:, 0]
    return derived


def _perturb_model[T: JaxModel](model: T, key: jax.Array, scale: float) -> T:
    """Nudge a model's trainable weights with small multiplicative noise.

    Used to recover from a solver failure (e.g. the model has drifted into
    a stiff numerical domain): perturbing the weights before moving on to
    the next training step gives the model a chance to land somewhere the
    solver can handle, instead of failing identically on every subsequent
    step too.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    noised = [
        leaf
        + scale * jax.random.normal(k, leaf.shape, leaf.dtype) * (jnp.abs(leaf) + 1e-12)
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, noised), static)


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
    args: jax.Array | None = None,
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
    args : jax.Array or None
        External arguments forwarded to ``model.integrate``, held constant
        across the trajectory (e.g. free/runtime parameters). Every current
        ``JaxModel``'s ``__call__`` unconditionally concatenates ``args``
        with its own state, so ``None`` here is only safe for models that
        need zero external arguments; it's translated to an empty array
        before being passed on, not left as ``None``.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree.
    """
    y_pred = model.integrate(
        ts,
        ys[0],
        max_steps=ctx.max_steps,
        args=jnp.zeros((0,)) if args is None else args,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


@eqx.filter_value_and_grad
def proto_grad_loss(
    model: JaxModel,
    ts: jax.Array,
    ys: jax.Array,
    protocol: jax.Array,
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
        Protocol transition times, shape ``(n_steps,)``.  ``ts[i]`` is the
        time at which ``protocol[i]`` stops being active; ``ys[i + 1]`` is
        the observation at that time.
    ys : jax.Array
        Observed trajectories, shape ``(n_steps + 1, n_obs)``.  ``ys[0]`` is
        the initial condition at ``t=0``.
    protocol : jax.Array
        Per-step external argument, shape ``(n_steps, n_args)``.
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
    y0 = ys[0]
    ts_segments = [ts[i : i + 1] for i in range(ts.shape[0])]
    y_pred = jnp.concatenate(
        (
            y0[None, :],
            model.integrate_protocol(
                ts=ts_segments,
                y0=y0,
                protocol=protocol,
                max_steps=ctx.max_steps,
                rtol=ctx.rtol,
                atol=ctx.atol,
                method=ctx.method,
            ),
        ),
        axis=0,
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
    args: jax.Array | None = None,
) -> tuple[jax.Array, T, optax.OptState, jax.Array]:
    """Perform one gradient update step on the full model.

    Non-finite gradients (from a failed ODE solve) and exploding adjoints
    are expected to be handled by ``optim`` itself -- see :func:`train`'s use
    of ``optax.apply_if_finite``/``optax.adaptive_grad_clip`` -- rather than
    here, so a bad step is skipped outright (including the optimiser's own
    internal state) instead of fed a zeroed-out gradient.

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
    args : jax.Array or None
        External arguments forwarded to ``model.integrate``; see
        :func:`grad_loss`.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState, jax.Array]
        Scalar loss, updated model, updated optimiser state, and the
        gradient norm (computed before ``optim`` does any clipping, for
        diagnostics).
    """
    loss, grads = grad_loss(
        model,  # needs to be by pos, is differentiated
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        args=args,
    )
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    return loss, eqx.apply_updates(model, updates), opt_state, grad_norm


@eqx.filter_jit
def proto_make_step[T: JaxModel](
    *,
    model: T,
    ts: jax.Array,
    ys: jax.Array,
    protocol: jax.Array,
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
    loss, grads = proto_grad_loss(
        model,  # needs to be by pos, is differentiated
        ts=ts,
        ys=ys,
        protocol=protocol,
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


def train[Model: JaxModel](
    model: Model,
    *,
    ts: jax.Array,
    ys: jax.Array,
    training_steps: list[tuple[int, float]],
    target_loss: float = 1e-4,
    avg_every: int = 1000,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
    clip_norm: float = 1.0,
    max_consecutive_nonfinite: int = 50,
    max_consecutive_solver_errors: int = 10,
    perturbation_scale: float = 1e-3,
    key: jax.Array | None = None,
    args: jax.Array | None = None,
) -> tuple[Model, LossesPerLesson, GradNormsPerLesson]:
    """Train a JAX model through a sequence of curriculum steps.

    Each curriculum stage re-initialises the optimiser and uses a
    different fraction of the data.  Early stopping triggers when
    ``loss < target_loss`` during the **last** stage.

    ``optim`` is wrapped in ``optax.apply_if_finite`` (skip the whole update,
    including the optimiser's own internal state, on a non-finite gradient)
    and ``optax.adaptive_grad_clip`` (per-parameter clipping by weight norm),
    so a bad step from a failed solve doesn't corrupt training. A solver
    failure that raises (``eqx.EquinoxRuntimeError``, e.g. from a diffrax
    integration hitting a stiff domain) is instead recovered from by
    perturbing the model's trainable weights with small multiplicative noise
    and moving on to the next step (not retrying the failed one), up to
    ``max_consecutive_solver_errors`` failures in a row before giving up on
    that curriculum stage and moving to the next one; the failure counter
    resets at the start of each stage. A ``KeyboardInterrupt`` during
    training returns the best model and losses/grad-norms accumulated so
    far instead of propagating.

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
        Optimiser; defaults to AdaBelief with ``lr=1e-4``.  Wrapped in
        ``apply_if_finite``/``adaptive_grad_clip`` before use; pass the raw
        optimiser here, not a pre-wrapped one.
    integration_settings : IntegrationSettings or None
        ODE solver settings; defaults to ``IntegrationSettings()``.
    clip_norm : float
        Per-parameter weight-norm clip threshold for
        ``optax.adaptive_grad_clip``.
    max_consecutive_nonfinite : int
        Consecutive non-finite updates ``optax.apply_if_finite`` skips
        before giving up and applying one anyway (it never raises).
    max_consecutive_solver_errors : int
        Consecutive solver failures (``eqx.EquinoxRuntimeError``) tolerated
        -- via perturb-and-retry -- before giving up on the current
        curriculum stage.
    perturbation_scale : float
        Relative magnitude of the weight perturbation applied after a
        solver failure; see :func:`_perturb_model`.
    key : jax.Array or None
        PRNG key seeding the weight perturbations; defaults to
        ``jax.random.PRNGKey(0)``.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate`` at every step;
        see :func:`grad_loss`.

    Returns
    -------
    tuple[Model, LossesPerLesson, GradNormsPerLesson]
        Best model encountered during training, losses per curriculum
        lesson, and per-step gradient norms per curriculum lesson.
    """
    loss = 0
    acc_loss = 0
    acc_count = 0
    perturb_key: jax.Array = jax.random.PRNGKey(0) if key is None else key

    raw_optim = optax.adabelief(learning_rate=1e-4) if optim is None else optim
    optim = optax.apply_if_finite(
        optax.chain(optax.adaptive_grad_clip(clip_norm), raw_optim),
        max_consecutive_errors=max_consecutive_nonfinite,
    )

    best_training_loss = jnp.inf
    best_model = model

    ctx = (
        IntegrationSettings() if integration_settings is None else integration_settings
    )

    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses_per_lesson: LossesPerLesson = []
    grad_norms_per_lesson: GradNormsPerLesson = []

    def _finish() -> tuple[Model, LossesPerLesson, GradNormsPerLesson]:
        return (
            best_model,
            [pd.Series(i) for i in losses_per_lesson],
            [pd.Series(i) for i in grad_norms_per_lesson],
        )

    try:
        for i, (steps, frac) in enumerate(training_steps, start=1):
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            # Fresh per stage, same as opt_state/losses/grad_norms below --
            # a stage's retry budget must not be affected by failures that
            # already happened in a previous stage.
            consecutive_solver_errors = 0
            length = math.ceil(len(ts) * frac)
            _ts = ts[:length]
            _ys = ys[:length]
            losses: dict[int, float] = {}
            grad_norms: dict[int, float] = {}
            losses_per_lesson.append(losses)
            grad_norms_per_lesson.append(grad_norms)

            with trange(steps, position=1, leave=True, dynamic_ncols=True) as pbar:
                for step in pbar:
                    try:
                        loss, model, opt_state, grad_norm = make_step(
                            model=model,
                            ts=_ts,
                            ys=_ys,
                            opt_state=opt_state,
                            optim=optim,
                            y_mean=y_mean,
                            y_scale=y_scale,
                            ctx=ctx,
                            args=args,
                        )
                    except eqx.EquinoxRuntimeError as e:
                        consecutive_solver_errors += 1
                        if consecutive_solver_errors > max_consecutive_solver_errors:
                            logger.warning(
                                "Lesson %d, step %d: solver failed %d times in a "
                                "row (%s); moving to the next curriculum stage.",
                                i,
                                step,
                                consecutive_solver_errors,
                                e,
                            )
                            break
                        perturb_key, subkey = jax.random.split(perturb_key)
                        model = _perturb_model(model, subkey, perturbation_scale)
                        logger.warning(
                            "Lesson %d, step %d: solver hit a stiff domain (%s); "
                            "nudging model parameters and retrying.",
                            i,
                            step,
                            e,
                        )
                        continue
                    consecutive_solver_errors = 0
                    grad_norms[step] = float(grad_norm)
                    acc_loss += loss
                    acc_count += 1

                    if loss < best_training_loss:
                        best_model = model
                        best_training_loss = loss

                    if i == len(training_steps) and loss < target_loss:
                        return _finish()

                    if (step % avg_every) == 0 or step == steps - 1:
                        avg_loss = acc_loss / acc_count
                        pbar.set_postfix_str(
                            f"Avg. loss {(avg_loss):.2e} over last {acc_count} runs"
                        )
                        acc_loss = 0
                        acc_count = 0
                        losses[step] = float(avg_loss)
    except KeyboardInterrupt:
        logger.warning("Training interrupted manually.")

    return _finish()


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
) -> tuple[T, LossesPerLesson]:
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
        Best model encountered during training and losses per curriculum lesson.
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

    losses_per_lesson = []
    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses = {}
    for i, (steps, frac) in enumerate(training_steps):
        opt_state = optim.init(trainable)
        length = math.ceil(len(ts) * frac)
        _ts = ts[:length]
        _ys = ys[:length]
        losses = {}
        losses_per_lesson.append(losses)

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
                    return (
                        eqx.combine(trainable, frozen),
                        [pd.Series(i) for i in losses_per_lesson],
                    )

                if (step % avg_every) == 0 or step == steps - 1:
                    avg_loss = acc_loss / acc_count
                    pbar.set_postfix_str(
                        f"Avg. loss {(avg_loss):.2e} over last {acc_count} runs"
                    )
                    global_step += acc_count
                    acc_loss = 0
                    acc_count = 0
                    losses[global_step] = float(avg_loss)
    return best_model, [pd.Series(i) for i in losses_per_lesson]


def train_protocol[Model: JaxModel](
    model: Model,
    *,
    ts: jax.Array,
    ys: jax.Array,
    protocol: jax.Array,
    training_steps: list[tuple[int, float]],
    target_loss: float = 1e-4,
    avg_every: int = 1000,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
) -> tuple[Model, LossesPerLesson]:
    """Train a JAX model through a sequence of curriculum steps.

    Each curriculum stage re-initialises the optimiser and uses a
    different fraction of the data.  Early stopping triggers when
    ``loss < target_loss`` during the **last** stage.

    Parameters
    ----------
    model : T
        Initial model state.
    ts : jax.Array
        Full protocol transition-time array, shape ``(n_steps,)``.
    ys : jax.Array
        Full observed trajectory, shape ``(n_steps + 1, n_obs)``.
    protocol : jax.Array
        Full per-step external argument, shape ``(n_steps, n_args)``.
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
        Best model encountered during training and losses per curriculum lesson.
    """
    loss = 0
    acc_loss = 0
    acc_count = 0
    optim = optax.adabelief(learning_rate=1e-4) if optim is None else optim
    best_training_loss = jnp.inf
    best_model = model

    ctx = (
        IntegrationSettings() if integration_settings is None else integration_settings
    )

    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses_per_lesson = []
    for i, (steps, frac) in enumerate(training_steps, start=1):
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        length = math.ceil(len(ts) * frac)
        _ts = ts[:length]
        _ys = ys[: length + 1]
        _protocol = protocol[:length]
        losses = {}
        losses_per_lesson.append(losses)

        with trange(steps, position=1, leave=True, dynamic_ncols=True) as pbar:
            for step in pbar:
                loss, model, opt_state = proto_make_step(
                    model=model,
                    ts=_ts,
                    ys=_ys,
                    protocol=_protocol,
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
                    return (model, [pd.Series(i) for i in losses_per_lesson])

                if (step % avg_every) == 0 or step == steps - 1:
                    avg_loss = acc_loss / acc_count
                    pbar.set_postfix_str(
                        f"Avg. loss {(avg_loss):.2e} over last {acc_count} runs"
                    )
                    acc_loss = 0
                    acc_count = 0
                    losses[step] = float(avg_loss)

    return best_model, [pd.Series(i) for i in losses_per_lesson]
