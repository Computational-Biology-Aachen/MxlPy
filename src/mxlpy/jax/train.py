"""Training loop utilities for JAX ODE and neural ODE models."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from jaxtyping import PyTree
from tqdm.auto import trange

from mxlpy.jax.models import JaxModel, Method, Ude

__all__ = [
    "FreezeFn",
    "GradLossFn",
    "GradLossSplitFn",
    "GradNormsPerLesson",
    "GradParsHistory",
    "GradParsPerLesson",
    "IntegrationSettings",
    "LossesPerLesson",
    "ProtoGradLossFn",
    "ProtoGradLossSplitFn",
    "ProtoSimulationFn",
    "TrainingSteps",
    "derived_simulation_fn",
    "grad_loss",
    "grad_loss_split",
    "integrate_protocol_states",
    "make_step",
    "make_step_split",
    "perturb_model",
    "proto_grad_loss",
    "proto_grad_loss_split",
    "proto_grad_loss_split_t0",
    "proto_grad_loss_t0",
    "proto_make_step",
    "proto_make_step_split",
    "sigmoid_schedule",
    "squeeze_derived",
    "train",
    "train_only_nde",
    "train_protocol",
]

logger = logging.getLogger(__name__)

type LossesPerLesson = list[pd.Series]
type GradNormsPerLesson = list[pd.Series]
type GradParsPerLesson = list[pd.DataFrame]
type TrainingSteps = list[tuple[int, float]]
type FreezeFn = Callable[[Any], Any]


def _curriculum_length(total: int, cutoff: float) -> int:
    """Resolve a training_steps cutoff to an absolute prefix length.

    A ``float`` cutoff is a fraction of ``total``, rounded up. An ``int``
    cutoff is an explicit element count (points or protocol windows,
    depending on caller), clipped to ``total`` if it overshoots.
    """
    if isinstance(cutoff, int):
        return min(cutoff, total)
    return math.ceil(total * cutoff)


class GradParsHistory:
    """Optional per-step, per-parameter gradient tracker for :func:`train`/:func:`train_protocol`.

    Records the raw gradient of ``model.pars`` (:class:`~mxlpy.jax.models.Ode`/
    :class:`~mxlpy.jax.models.FluxOde`'s flat trainable parameter vector) at
    every step, alongside the aggregate :data:`GradNormsPerLesson` already
    returned -- useful for diagnosing which specific parameters drive
    instability or fail to learn. Pass an instance via ``train``'s/
    ``train_protocol``'s ``grad_pars_history`` parameter; this doesn't
    change either function's return signature, so passing one in is fully
    opt-in and every other caller is unaffected. Only supported for models
    with a flat ``.pars`` array; :func:`train_only_nde` trains a neural
    network partition (no such flat vector), so it isn't supported there.

    Retrieve the recorded history afterwards via :meth:`to_frames`.
    """

    def __init__(self) -> None:
        self._by_lesson: list[dict[int, jax.Array]] = []

    def _start_lesson(self) -> None:
        self._by_lesson.append({})

    def _record(self, step: int, grad_pars: jax.Array) -> None:
        self._by_lesson[-1][step] = grad_pars

    def to_frames(self, par_names: list[str] | None = None) -> GradParsPerLesson:
        """Stack the recorded per-step gradients into one DataFrame per lesson.

        Parameters
        ----------
        par_names : list[str] or None
            Column names, in ``model.pars``'s order; defaults to
            positional indices ``0, 1, ...`` if not given.

        Returns
        -------
        GradParsPerLesson
            One DataFrame per lesson, indexed by step, columns named by
            ``par_names`` (or positional indices). A lesson with no
            recorded steps (e.g. every step failed and was skipped) yields
            an empty DataFrame.
        """
        frames: GradParsPerLesson = []
        for lesson in self._by_lesson:
            if not lesson:
                frames.append(pd.DataFrame(index=pd.Index([], name="step")))
                continue
            steps = list(lesson)
            values = jnp.stack([lesson[s] for s in steps])
            columns = (
                par_names if par_names is not None else list(range(values.shape[-1]))
            )
            frames.append(
                pd.DataFrame(
                    jax.device_get(values),
                    index=pd.Index(steps, name="step"),
                    columns=columns,
                )
            )
        return frames


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


def sigmoid_schedule(
    global_step: jax.Array,
    total_steps: jax.Array,
    steepness: float,
    midpoint: float,
) -> jax.Array:
    """Sigmoid annealing schedule over the whole curriculum.

    Returns ~1 early in training, smoothly decaying to ~0 as ``global_step``
    approaches ``total_steps``; not specific to any one use (e.g. blending
    :class:`~mxlpy.jax.models.MixedLatentMapper`/
    :class:`~mxlpy.jax.models.TrainableZ0LatentMapper`'s hard/learned
    decode) -- any custom ``grad_fn`` can use it against the ``global_step``/
    ``total_steps`` :func:`train`/:func:`train_protocol` already forward to
    it.

    Parameters
    ----------
    global_step : jax.Array
        Current step, monotonic across the whole curriculum (not reset per
        lesson); see :func:`grad_loss`.
    total_steps : jax.Array
        Total step count across every curriculum lesson.
    steepness : float
        How sharp the transition is; larger is sharper.
    midpoint : float
        Fraction of ``total_steps`` (in ``[0, 1]``) at which the schedule
        crosses ``0.5``.

    Returns
    -------
    jax.Array
        Scalar in ``(0, 1)``.
    """
    x = (global_step / total_steps - midpoint) * steepness
    return 1.0 / (1.0 + jnp.exp(x))


def perturb_model[T: JaxModel](model: T, key: jax.Array, scale: float) -> T:
    """Nudge a model's trainable weights with small multiplicative noise.

    Used by :func:`train`/:func:`train_protocol`/:func:`train_only_nde` to
    recover from a solver failure (e.g. the model has drifted into a stiff
    numerical domain): perturbing the weights before moving on to the next
    training step gives the model a chance to land somewhere the solver can
    handle, instead of failing identically on every subsequent step too.
    Exported so custom training loops can reuse this recovery primitive
    without reimplementing it.

    Parameters
    ----------
    model : T
        Model (or a partitioned subset of one, e.g. the trainable half of
        an :func:`eqx.partition`) to perturb.
    key : jax.Array
        PRNG key for the per-leaf noise.
    scale : float
        Relative magnitude of the perturbation: each inexact-array leaf is
        nudged by ``scale * N(0, 1) * (abs(leaf) + 1e-12)``, elementwise.

    Returns
    -------
    T
        A new model with the same non-array/static structure as ``model``
        and every inexact-array leaf perturbed.
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


type GradLossFn = Callable[
    ...,
    tuple[jax.Array, PyTree],
]


@eqx.filter_value_and_grad
def grad_loss(
    model: JaxModel,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
    args: jax.Array | None = None,
) -> jax.Array:
    """Compute normalised MSE loss and its gradient w.r.t. model parameters.

    Default ``grad_fn`` for :func:`train`/:func:`make_step`. Pass a
    different ``@eqx.filter_value_and_grad``-decorated function with this
    same ``(model, ts, ys, y_mean, y_scale, ctx, global_step, total_steps,
    args)`` call signature to train against a different loss (e.g. a
    residual against a frozen prior prediction, or a different error
    metric) without reimplementing the curriculum/robustness machinery in
    :func:`train`.

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
    global_step : jax.Array
        Monotonic step count spanning the whole curriculum (not reset per
        lesson); unused here, but available to a custom ``grad_fn`` for a
        training-progress-dependent computation (e.g. an annealed loss
        term or model parameter via :func:`sigmoid_schedule`). A traced
        array, not a plain Python int, so its changing value each step
        doesn't force a JIT recompile.
    total_steps : jax.Array
        Total step count across every lesson in ``training_steps``; see
        ``global_step``.
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


type ProtoSimulationFn = Callable[
    [JaxModel, list[jax.Array], jax.Array, jax.Array, IntegrationSettings], jax.Array
]


def integrate_protocol_states(
    model: JaxModel,
    ts: list[jax.Array],
    y0: jax.Array,
    protocol: jax.Array,
    ctx: IntegrationSettings,
) -> jax.Array:
    """Default ``simulation_fn`` for :func:`train_protocol`: raw integrated state.

    Compares predictions directly in state space (``ys`` must hold the raw
    ODE state, not a derived quantity). Models that need to fit a derived
    observable (e.g. fluorescence) instead of the raw state should pass
    :func:`derived_simulation_fn` instead.
    """
    return model.integrate_protocol(
        ts=ts,
        y0=y0,
        protocol=protocol,
        max_steps=ctx.max_steps,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )


def derived_simulation_fn(
    model: JaxModel,
    ts: list[jax.Array],
    y0: jax.Array,
    protocol: jax.Array,
    ctx: IntegrationSettings,
    *,
    t0_args: jax.Array | None = None,
) -> jax.Array:
    """Alternative ``simulation_fn`` for :func:`train_protocol`: a derived observable.

    Maps :meth:`JaxModel.integrate_protocol`'s raw state through
    ``model.derived(...)``, for fitting a derived quantity (e.g.
    fluorescence) instead of the raw state -- exactly the pattern
    :func:`integrate_protocol_states`'s own docstring tells callers to
    write themselves. Pass this directly as ``train_protocol``'s
    ``simulation_fn`` when ``ys`` holds a derived observable rather than
    raw state.

    Keeps ``model.derived``'s trailing quantity axis intact (shape
    ``(T, n_derived)``), matching ``ys``'s own shape convention -- the
    caller composes :func:`squeeze_derived` themselves (via a thin
    wrapper) for the less common case of a 1-D ``ys`` with that axis
    already dropped.

    Every saved time point is paired with the external argument active at
    that point (``protocol[i]``, held constant for the duration of
    ``ts[i]``) before calling ``model.derived``, since a derived quantity
    can itself depend on that argument (e.g. a light-dependent
    fluorescence yield).

    Parameters
    ----------
    t0_args : jax.Array or None
        When given, prepend a genuine ``model.derived(0.0, y0, t0_args)``
        row for ``t=0`` -- returning shape ``(T + 1, n_derived)`` instead
        of ``(T, n_derived)``. Needed because ``train_protocol``'s
        ``ts``/``protocol`` (see :func:`~mxlpy.jax.models.boundaries_to_ts`)
        describe only the protocol's steps, never ``t=0`` itself, yet a
        derived quantity's value at ``t=0`` generally isn't ``ys[0]`` for
        free the way the raw state trivially is (``y0`` *is* the state at
        ``t=0``, but ``model.derived(0, y0, t0_args)`` still has to be
        computed to know the derived quantity there). ``None`` (the
        default) leaves the return shape unchanged, so this stays a
        drop-in :data:`ProtoSimulationFn` -- pair a bound ``t0_args``
        (e.g. via ``functools.partial``) with :func:`proto_grad_loss_t0`/
        :func:`proto_grad_loss_split_t0` rather than the plain
        ``proto_grad_loss``/``proto_grad_loss_split``, which assume a
        ``(T, n_derived)``-shaped ``simulation_fn`` and free-pass ``ys[0]``
        as the ``t=0`` prediction instead.
    """
    states = model.integrate_protocol(
        ts=ts,
        y0=y0,
        protocol=protocol,
        max_steps=ctx.max_steps,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )
    ts_flat = jnp.concatenate(ts)
    step_lengths = [t.shape[0] for t in ts]
    args_flat = jnp.concatenate(
        [
            jnp.broadcast_to(protocol[i], (n, *protocol.shape[1:]))
            for i, n in enumerate(step_lengths)
        ],
        axis=0,
    )
    derived = jax.vmap(model.derived)(ts_flat, states, args_flat)
    if t0_args is None:
        return derived
    t0_derived = model.derived(jnp.zeros(()), y0, t0_args)
    return jnp.concatenate((t0_derived[None], derived), axis=0)


type ProtoGradLossFn = Callable[
    ...,
    tuple[jax.Array, PyTree],
]


@eqx.filter_value_and_grad
def proto_grad_loss(
    model: JaxModel,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
) -> jax.Array:
    """Compute normalised MSE loss and its gradient w.r.t. model parameters.

    Default ``grad_fn`` for :func:`train_protocol`/:func:`proto_make_step`.
    Pass a different ``@eqx.filter_value_and_grad``-decorated function with
    this same call signature to train against a different loss.

    Parameters
    ----------
    model : JaxModel
        Model to differentiate.
    y0 : jax.Array
        Physiological state at ``t=0``, used to seed integration. Kept
        separate from ``ys[0]`` since ``ys`` lives in observation space
        (e.g. fluorescence) which need not coincide with the raw state
        ``simulation_fn`` integrates from.
    ts : list[jax.Array]
        Per-step, ascending, absolute time points at which to save the
        solution -- one array per protocol step, mirroring
        :meth:`JaxModel.integrate_protocol`. The last entry of each array
        must be that step's transition time (where ``protocol`` switches to
        its next row).
    ys : jax.Array
        Observed trajectories, shape ``(sum(len(t) for t in ts) + 1, n_obs)``.
        ``ys[0]`` is the observation at ``t=0``; the rest align in order with
        the concatenated ``ts``.
    protocol : jax.Array
        Per-step external argument, shape ``(len(ts), n_args)``.
    y_mean : jax.Array
        Mean used for normalisation.
    y_scale : jax.Array
        Standard deviation used for normalisation.
    ctx : IntegrationSettings
        ODE solver settings.
    simulation_fn : ProtoSimulationFn
        Maps ``(model, ts, y0, protocol, ctx)`` to predictions aligned with
        ``ys[1:]``; see :func:`integrate_protocol_states` for the raw-state
        default.
    global_step : jax.Array
        Monotonic step count spanning the whole curriculum; see
        :func:`grad_loss`.
    total_steps : jax.Array
        Total step count across every lesson; see :func:`grad_loss`.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree.
    """
    pred = simulation_fn(model, ts, y0, protocol, ctx)
    y_pred = jnp.concatenate((ys[0][None], pred), axis=0)
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


@eqx.filter_value_and_grad
def proto_grad_loss_t0(
    model: JaxModel,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
) -> jax.Array:
    """:func:`proto_grad_loss` variant for a ``simulation_fn`` that predicts ``t=0`` itself.

    ``proto_grad_loss`` free-passes ``ys[0]`` as the ``t=0`` "prediction",
    which is exact for the raw-state default (``y0`` *is* the state at
    ``t=0``) but wrong for a derived observable, silently zeroing out any
    loss contribution at ``t=0`` regardless of whether the model actually
    reproduces it there. Use this instead when ``simulation_fn`` already
    returns a genuine, full-length prediction spanning ``ys`` (e.g.
    :func:`derived_simulation_fn` with ``t0_args`` bound via
    ``functools.partial``) -- compares ``simulation_fn``'s output against
    ``ys`` directly, with no substitution.

    Parameters
    ----------
    simulation_fn : ProtoSimulationFn
        Maps ``(model, ts, y0, protocol, ctx)`` to predictions aligned
        with the *full* ``ys`` (including ``t=0``), unlike
        :func:`proto_grad_loss`'s ``simulation_fn``.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree.
    """
    y_pred = simulation_fn(model, ts, y0, protocol, ctx)
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


type ProtoGradLossSplitFn = Callable[
    ...,
    tuple[jax.Array, PyTree],
]


@eqx.filter_value_and_grad
def proto_grad_loss_split(
    trainable: JaxModel,
    frozen: JaxModel,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
) -> jax.Array:
    """Compute normalised MSE loss and gradient for a partitioned model.

    Default ``split_grad_fn`` for :func:`train_protocol` when its ``freeze``
    argument is given, and for :func:`proto_make_step_split`. Pass a
    different ``@eqx.filter_value_and_grad``-decorated function with this
    same ``(trainable, frozen, y0, ts, ys, protocol, y_mean, y_scale, ctx,
    simulation_fn, global_step, total_steps)`` call signature to train
    against a different loss; see :func:`proto_grad_loss`, which this
    mirrors for a partitioned model.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree w.r.t. ``trainable``.
    """
    model = eqx.combine(trainable, frozen)
    pred = simulation_fn(model, ts, y0, protocol, ctx)
    y_pred = jnp.concatenate((ys[0][None], pred), axis=0)
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


@eqx.filter_value_and_grad
def proto_grad_loss_split_t0(
    trainable: JaxModel,
    frozen: JaxModel,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
) -> jax.Array:
    """:func:`proto_grad_loss_split` variant for a ``t=0``-predicting ``simulation_fn``.

    Mirrors :func:`proto_grad_loss_t0` for a partitioned model; see that
    function for why this exists instead of :func:`proto_grad_loss_split`.
    Pass as ``train_protocol``'s ``split_grad_fn`` (alongside
    ``grad_fn=proto_grad_loss_t0``) when ``freeze`` is also given.

    Returns
    -------
    tuple[jax.Array, PyTree]
        Scalar loss value and gradient PyTree w.r.t. ``trainable``.
    """
    model = eqx.combine(trainable, frozen)
    y_pred = simulation_fn(model, ts, y0, protocol, ctx)
    return jnp.mean(((ys - y_mean) / y_scale - (y_pred - y_mean) / y_scale) ** 2)


type GradLossSplitFn = Callable[
    ...,
    tuple[jax.Array, PyTree],
]


@eqx.filter_value_and_grad
def grad_loss_split(
    trainable: JaxModel,
    frozen: JaxModel,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    ctx: IntegrationSettings,
    global_step: jax.Array,  # noqa: ARG001
    total_steps: jax.Array,  # noqa: ARG001
    args: jax.Array | None = None,
) -> jax.Array:
    """Compute normalised MSE loss and gradient for a partitioned model.

    Default ``grad_fn``/``split_grad_fn`` for :func:`train_only_nde`,
    :func:`train`'s ``freeze``, and :func:`make_step_split`. Pass a
    different ``@eqx.filter_value_and_grad``-decorated function with this
    same ``(trainable, frozen, ts, ys, y_mean, y_scale, ctx, global_step,
    total_steps, args)`` call signature to train against a different loss.

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
    global_step : jax.Array
        Monotonic step count spanning the whole curriculum; unused here,
        but available to a custom ``grad_fn``/``split_grad_fn`` the same
        way :func:`grad_loss`'s is (e.g. for :func:`sigmoid_schedule`).
    total_steps : jax.Array
        Total step count across every lesson; see ``global_step``.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate``; see
        :func:`grad_loss`.

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
        args=jnp.zeros((0,)) if args is None else args,
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
    global_step: jax.Array,
    total_steps: jax.Array,
    args: jax.Array | None = None,
    grad_fn: GradLossFn = grad_loss,
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
    global_step : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    total_steps : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate``; see
        :func:`grad_loss`.
    grad_fn : GradLossFn
        Computes the (loss, grads) pair to apply; defaults to
        :func:`grad_loss`. Pass a different ``@eqx.filter_value_and_grad``
        function with the same call signature to train against a different
        loss.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState, jax.Array]
        Scalar loss, updated model, updated optimiser state, and the
        gradient norm (computed before ``optim`` does any clipping, for
        diagnostics).
    """
    loss, grads = grad_fn(
        model,  # needs to be by pos, is differentiated
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        global_step=global_step,
        total_steps=total_steps,
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
def _make_step_with_grad_pars[T: JaxModel](
    *,
    model: T,
    ts: jax.Array,
    ys: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
    global_step: jax.Array,
    total_steps: jax.Array,
    args: jax.Array | None = None,
    grad_fn: GradLossFn = grad_loss,
) -> tuple[jax.Array, T, optax.OptState, jax.Array, jax.Array]:
    """Same as :func:`make_step`, additionally returning ``grads.pars``.

    Used internally by :func:`train` when a :class:`GradParsHistory` is
    passed; a separate function (rather than an always-present return
    value on :func:`make_step` itself) so :func:`make_step`'s existing
    signature/callers are unaffected, and so this doesn't assume every
    model has a flat ``.pars`` attribute (:func:`make_step` works for any
    :class:`~mxlpy.jax.models.JaxModel`, but ``model.pars`` only exists on
    :class:`~mxlpy.jax.models.Ode`/:class:`~mxlpy.jax.models.FluxOde`).
    """
    loss, grads = grad_fn(
        model,  # needs to be by pos, is differentiated
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        global_step=global_step,
        total_steps=total_steps,
        args=args,
    )
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    return loss, eqx.apply_updates(model, updates), opt_state, grad_norm, grads.pars


@eqx.filter_jit
def proto_make_step[T: JaxModel](
    *,
    model: T,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,
    total_steps: jax.Array,
    grad_fn: ProtoGradLossFn = proto_grad_loss,
) -> tuple[jax.Array, T, optax.OptState, jax.Array]:
    """Perform one gradient update step on the full model.

    NaN gradients (from failed ODE solves) are zeroed before the update.
    Gradients are clipped by global norm to guard against exploding adjoints.

    Parameters
    ----------
    model : T
        Current model state.
    y0 : jax.Array
        Physiological state at ``t=0``; see :func:`proto_grad_loss`.
    ts : list[jax.Array]
        Per-step time points.
    ys : jax.Array
        Observed trajectories.
    protocol : jax.Array
        Per-step external argument.
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
    simulation_fn : ProtoSimulationFn
        See :func:`proto_grad_loss`.
    global_step : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    total_steps : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    grad_fn : ProtoGradLossFn
        Computes the (loss, grads) pair to apply; defaults to
        :func:`proto_grad_loss`. Pass a different
        ``@eqx.filter_value_and_grad`` function with the same call
        signature to train against a different loss.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState, jax.Array]
        Scalar loss, updated model, updated optimiser state, and the
        gradient norm (computed before clipping, for diagnostics).
    """
    loss, grads = grad_fn(
        model,  # needs to be by pos, is differentiated
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        simulation_fn=simulation_fn,
        global_step=global_step,
        total_steps=total_steps,
    )
    # Failed solves can produce NaN in y_pred -> NaN gradients; zero them out
    # so the optimizer state isn't corrupted.
    grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)

    # Clip gradients by global norm to guard against exploding adjoints;
    # only scales down (never up) so small, well-behaved gradients are left
    # alone.
    grad_norm = optax.global_norm(grads)
    clip_value = 1.0
    scale = jnp.minimum(1.0, clip_value / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    return loss, eqx.apply_updates(model, updates), opt_state, grad_norm


@eqx.filter_jit
def proto_make_step_split[T: JaxModel](
    *,
    trainable: T,
    frozen: T,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,
    total_steps: jax.Array,
    grad_fn: ProtoGradLossSplitFn = proto_grad_loss_split,
) -> tuple[jax.Array, T, optax.OptState, jax.Array]:
    """Perform one gradient update step on the trainable model partition.

    Mirrors :func:`proto_make_step` (NaN-safe gradients, global-norm
    clipping) for a partitioned model; see :func:`make_step_split` for the
    non-protocol equivalent. Used by :func:`train_protocol` when its
    ``freeze`` argument is given.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState, jax.Array]
        Scalar loss, updated trainable partition, updated optimiser state,
        and the gradient norm (computed before clipping, for diagnostics).
    """
    loss, grads = grad_fn(
        trainable,  # needs to be by pos, is differentiated
        frozen=frozen,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        simulation_fn=simulation_fn,
        global_step=global_step,
        total_steps=total_steps,
    )
    # Failed solves can produce NaN in y_pred -> NaN gradients; zero them out
    # so the optimizer state isn't corrupted.
    grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)

    # Clip gradients by global norm to guard against exploding adjoints;
    # only scales down (never up) so small, well-behaved gradients are left
    # alone.
    grad_norm = optax.global_norm(grads)
    clip_value = 1.0
    scale = jnp.minimum(1.0, clip_value / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(trainable, eqx.is_array),
    )
    return loss, eqx.apply_updates(trainable, updates), opt_state, grad_norm


@eqx.filter_jit
def _proto_make_step_with_grad_pars[T: JaxModel](
    *,
    model: T,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    y_mean: jax.Array,
    y_scale: jax.Array,
    opt_state: optax.OptState,
    optim: optax.GradientTransformationExtraArgs,
    ctx: IntegrationSettings,
    simulation_fn: ProtoSimulationFn,
    global_step: jax.Array,
    total_steps: jax.Array,
    grad_fn: ProtoGradLossFn = proto_grad_loss,
) -> tuple[jax.Array, T, optax.OptState, jax.Array, jax.Array]:
    """Same as :func:`proto_make_step`, additionally returning ``grads.pars``.

    Used internally by :func:`train_protocol` when a
    :class:`GradParsHistory` is passed; see
    :func:`_make_step_with_grad_pars` for why this is a separate function
    rather than an always-present return value on :func:`proto_make_step`.
    """
    loss, grads = grad_fn(
        model,  # needs to be by pos, is differentiated
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        simulation_fn=simulation_fn,
        global_step=global_step,
        total_steps=total_steps,
    )
    grads = jax.tree.map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    grad_norm = optax.global_norm(grads)
    clip_value = 1.0
    scale = jnp.minimum(1.0, clip_value / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    return loss, eqx.apply_updates(model, updates), opt_state, grad_norm, grads.pars


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
    global_step: jax.Array,
    total_steps: jax.Array,
    args: jax.Array | None = None,
    grad_fn: GradLossSplitFn = grad_loss_split,
) -> tuple[jax.Array, T, optax.OptState, jax.Array]:
    """Perform one gradient update step on the trainable model partition.

    Non-finite gradients (from a failed ODE solve) and exploding adjoints
    are expected to be handled by ``optim`` itself -- see :func:`train_only_nde`'s
    use of ``optax.apply_if_finite``/``optax.adaptive_grad_clip`` -- rather
    than here, mirroring :func:`make_step`.

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
    global_step : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    total_steps : jax.Array
        Forwarded to ``grad_fn``; see :func:`grad_loss`.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate``; see
        :func:`grad_loss`.
    grad_fn : GradLossSplitFn
        Computes the (loss, grads) pair to apply; defaults to
        :func:`grad_loss_split`. Pass a different
        ``@eqx.filter_value_and_grad`` function with the same call
        signature to train against a different loss.

    Returns
    -------
    tuple[jax.Array, T, optax.OptState, jax.Array]
        Scalar loss, updated trainable partition, updated optimiser state,
        and the gradient norm (computed before ``optim`` does any clipping,
        for diagnostics).
    """
    loss, grads = grad_fn(
        trainable,  # needs to be by pos, is differentiated
        frozen=frozen,
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        global_step=global_step,
        total_steps=total_steps,
        args=args,
    )
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optim.update(
        grads,
        opt_state,
        eqx.filter(trainable, eqx.is_array),
    )
    return loss, eqx.apply_updates(trainable, updates), opt_state, grad_norm


def train[Model: JaxModel](
    model: Model,
    *,
    ts: jax.Array,
    ys: jax.Array,
    training_steps: TrainingSteps,
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
    grad_fn: GradLossFn = grad_loss,
    stop_exceptions: tuple[type[BaseException], ...] = (KeyboardInterrupt,),
    prior_losses: LossesPerLesson | None = None,
    prior_grad_norms: GradNormsPerLesson | None = None,
    grad_pars_history: GradParsHistory | None = None,
    normalize_axis: int | None = None,
    disable_tqdm: bool = False,
    freeze: FreezeFn | None = None,
    split_grad_fn: GradLossSplitFn = grad_loss_split,
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
    resets at the start of each stage. A ``KeyboardInterrupt`` (or any other
    exception listed in ``stop_exceptions``) during training returns the
    best model and losses/grad-norms accumulated so far instead of
    propagating -- e.g. a scheduler-issued shutdown signal translated into a
    custom exception (SLURM sends SIGTERM shortly before killing a job that
    hit its hard time limit; a signal handler can raise a caller-defined
    exception from it, which this can then wind down gracefully from the
    same as a manual Ctrl-C, by passing it in ``stop_exceptions``).

    Parameters
    ----------
    model : T
        Initial model state.
    ts : jax.Array
        Full time-point array.
    ys : jax.Array
        Full observed trajectory, shape ``(T, n_obs)``.
    training_steps : TrainingSteps
        Sequence of ``(n_steps, cutoff)`` pairs. ``cutoff`` selects a
        prefix of ``ts``/``ys``: a ``float`` is a fraction of the full
        length (rounded up); an ``int`` is an explicit point count,
        clipped to the full length if it overshoots.
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
        solver failure; see :func:`perturb_model`.
    key : jax.Array or None
        PRNG key seeding the weight perturbations; defaults to
        ``jax.random.PRNGKey(0)``.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate`` at every step;
        see :func:`grad_loss`.
    grad_fn : GradLossFn
        Computes the (loss, grads) pair to apply at every step; defaults to
        :func:`grad_loss` (normalised MSE against ``ys``). Pass a different
        ``@eqx.filter_value_and_grad`` function with the same call
        signature to train against a different loss (e.g. a residual
        against a frozen prior prediction) without reimplementing this
        curriculum/robustness loop.
    stop_exceptions : tuple[type[BaseException], ...]
        Exception types that trigger a graceful stop (return the best
        model and history accumulated so far) instead of propagating;
        defaults to just ``KeyboardInterrupt``. Pass a tuple including a
        caller-defined exception (e.g. one raised from a SIGTERM handler)
        to also wind down gracefully from a scheduler-issued shutdown
        signal.
    prior_losses : LossesPerLesson or None
        Loss history from an earlier call to resume from (e.g. after
        checkpointing ``model``/losses/grad-norms to disk and reloading in
        a later process) -- prepended to this call's own per-lesson
        history in the returned value, so the result reads as one
        continuous record across the resume boundary instead of restarting
        from lesson 0.
    prior_grad_norms : GradNormsPerLesson or None
        Grad-norm history to resume from; see ``prior_losses``.
    grad_pars_history : GradParsHistory or None
        If given, records the raw gradient of ``model.pars`` at every
        successful step into it (requires ``model`` to have a flat
        ``.pars`` array, as :class:`~mxlpy.jax.models.Ode`/
        :class:`~mxlpy.jax.models.FluxOde` do); retrieve the recorded
        history afterwards via its ``to_frames()``. Off by default (no
        behaviour change, no extra cost) since not every model has a
        flat ``.pars`` to track.
    normalize_axis : int or None
        Axis passed to ``jnp.mean``/``jnp.std`` when computing ``y_mean``/
        ``y_scale`` from ``ys``. ``None`` (the default) normalises over the
        flattened array, i.e. one scalar shared across every observed
        quantity. Pass ``0`` to normalise each column of ``ys`` (each
        observed quantity, for a ``(T, n_obs)`` array with ``n_obs > 1``)
        independently instead -- needed when the fitted quantities have
        different scales (e.g. jointly fitting ``Fluo`` and ``NPQ``), since
        a shared scalar otherwise lets the higher-magnitude quantity
        dominate the loss.
    disable_tqdm : bool
        Suppress the per-step progress bar. Useful when running under a
        scheduler that captures stdout to a log file (a live progress bar's
        carriage-return updates otherwise bloat the file with escape codes).
    freeze : FreezeFn or None
        If given, an ``eqx.tree_at``-style selector (e.g. ``lambda m:
        m.spline``) picking the subtree of ``model`` to hold fixed;
        everything else is trained. Generalises :func:`train_only_nde`'s
        hardcoded ``lambda m: m.ode`` freeze to any model/subtree, so a
        model with some frozen, non-trainable component (e.g. a driver
        spline) doesn't need its own bespoke training loop to get this
        curriculum/robustness machinery. Solver-failure perturbation is
        applied to the trainable partition only, so a frozen subtree is
        never nudged. Not supported together with ``grad_pars_history``
        (its per-parameter tracking assumes a flat ``.pars`` on the whole
        model). ``None`` (the default) trains every leaf, identical to
        never having had this parameter.
    split_grad_fn : GradLossSplitFn
        Computes the (loss, grads) pair to apply at every step when
        ``freeze`` is given; defaults to :func:`grad_loss_split`. Ignored
        when ``freeze`` is ``None`` (use ``grad_fn`` instead).

    Returns
    -------
    tuple[Model, LossesPerLesson, GradNormsPerLesson]
        Best model encountered during training, losses per curriculum
        lesson, and per-step gradient norms per curriculum lesson --
        including any ``prior_losses``/``prior_grad_norms`` prepended.
    """
    if freeze is not None and grad_pars_history is not None:
        msg = "grad_pars_history is not supported together with freeze"
        raise ValueError(msg)

    loss = 0
    acc_loss = 0
    acc_count = 0
    global_step = 0
    total_steps = sum(s for s, _ in training_steps)
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

    y_mean = jnp.mean(ys, axis=normalize_axis)
    y_scale = jnp.std(ys, axis=normalize_axis)

    filter_spec = None
    if freeze is not None:
        filter_spec = jax.tree.map(eqx.is_inexact_array, model)
        filter_spec = eqx.tree_at(freeze, filter_spec, replace_fn=lambda _: False)

    def _init_opt_state(model: Model) -> optax.OptState:
        if filter_spec is None:
            return optim.init(eqx.filter(model, eqx.is_inexact_array))
        trainable, _frozen = eqx.partition(model, filter_spec)
        return optim.init(trainable)

    def _step(
        model: Model, opt_state: optax.OptState
    ) -> tuple[jax.Array, Model, optax.OptState, jax.Array]:
        if filter_spec is None:
            return make_step(
                model=model,
                ts=_ts,
                ys=_ys,
                opt_state=opt_state,
                optim=optim,
                y_mean=y_mean,
                y_scale=y_scale,
                ctx=ctx,
                global_step=jnp.asarray(global_step),
                total_steps=jnp.asarray(total_steps),
                args=args,
                grad_fn=grad_fn,
            )
        trainable, frozen = eqx.partition(model, filter_spec)
        loss, trainable, opt_state, grad_norm = make_step_split(
            trainable=trainable,
            frozen=frozen,
            ts=_ts,
            ys=_ys,
            opt_state=opt_state,
            optim=optim,
            y_mean=y_mean,
            y_scale=y_scale,
            ctx=ctx,
            global_step=jnp.asarray(global_step),
            total_steps=jnp.asarray(total_steps),
            args=args,
            grad_fn=split_grad_fn,
        )
        return loss, eqx.combine(trainable, frozen), opt_state, grad_norm

    def _perturb(model: Model, key: jax.Array, scale: float) -> Model:
        if filter_spec is None:
            return perturb_model(model, key, scale)
        trainable, frozen = eqx.partition(model, filter_spec)
        return eqx.combine(perturb_model(trainable, key, scale), frozen)

    losses_per_lesson: LossesPerLesson = (
        [] if prior_losses is None else list(prior_losses)
    )
    grad_norms_per_lesson: GradNormsPerLesson = (
        [] if prior_grad_norms is None else list(prior_grad_norms)
    )

    def _finish() -> tuple[Model, LossesPerLesson, GradNormsPerLesson]:
        return (
            best_model,
            [pd.Series(i) for i in losses_per_lesson],
            [pd.Series(i) for i in grad_norms_per_lesson],
        )

    try:
        for i, (steps, cutoff) in enumerate(training_steps, start=1):
            opt_state = _init_opt_state(model)
            # Fresh per stage, same as opt_state/losses/grad_norms below --
            # a stage's retry budget must not be affected by failures that
            # already happened in a previous stage.
            consecutive_solver_errors = 0
            length = _curriculum_length(len(ts), cutoff)
            _ts = ts[:length]
            _ys = ys[:length]
            losses: dict[int, float] = {}
            grad_norms: dict[int, float] = {}
            losses_per_lesson.append(losses)
            grad_norms_per_lesson.append(grad_norms)
            if grad_pars_history is not None:
                grad_pars_history._start_lesson()  # noqa: SLF001

            with trange(
                steps, position=1, leave=True, dynamic_ncols=True, disable=disable_tqdm
            ) as pbar:
                for step in pbar:
                    try:
                        # make_step's returned loss describes the model as
                        # it was BEFORE this step's update (it's computed,
                        # then the update is applied) -- so pre_step_model
                        # (not the post-update model make_step returns) is
                        # what pairs with it for best-model tracking below.
                        pre_step_model = model
                        if grad_pars_history is None:
                            loss, model, opt_state, grad_norm = _step(model, opt_state)
                        else:
                            loss, model, opt_state, grad_norm, grad_pars = (
                                _make_step_with_grad_pars(
                                    model=model,
                                    ts=_ts,
                                    ys=_ys,
                                    opt_state=opt_state,
                                    optim=optim,
                                    y_mean=y_mean,
                                    y_scale=y_scale,
                                    ctx=ctx,
                                    global_step=jnp.asarray(global_step),
                                    total_steps=jnp.asarray(total_steps),
                                    args=args,
                                    grad_fn=grad_fn,
                                )
                            )
                            grad_pars_history._record(step, grad_pars)  # noqa: SLF001
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
                        model = _perturb(model, subkey, perturbation_scale)
                        logger.warning(
                            "Lesson %d, step %d: solver hit a stiff domain (%s); "
                            "nudging model parameters and retrying.",
                            i,
                            step,
                            e,
                        )
                        continue
                    consecutive_solver_errors = 0
                    global_step += 1
                    grad_norms[step] = float(grad_norm)
                    acc_loss += loss
                    acc_count += 1

                    if loss < best_training_loss:
                        best_model = pre_step_model
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
    except stop_exceptions as e:
        logger.warning("Training stopped (%s).", type(e).__name__)

    return _finish()


def train_only_nde[T: Ude](
    model: T,
    *,
    ts: jax.Array,
    ys: jax.Array,
    training_steps: TrainingSteps,
    avg_every: int = 1000,
    target_loss: float = 1e-4,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
    clip_norm: float = 1.0,
    max_consecutive_nonfinite: int = 50,
    max_consecutive_solver_errors: int = 10,
    perturbation_scale: float = 1e-3,
    key: jax.Array | None = None,
    args: jax.Array | None = None,
    grad_fn: GradLossSplitFn = grad_loss_split,
    stop_exceptions: tuple[type[BaseException], ...] = (KeyboardInterrupt,),
    prior_losses: LossesPerLesson | None = None,
    prior_grad_norms: GradNormsPerLesson | None = None,
) -> tuple[T, LossesPerLesson, GradNormsPerLesson]:
    """Train only the neural-network part of a UDE, keeping the ODE frozen.

    Partitions ``model`` into trainable (neural network) and frozen (ODE)
    parts before the training loop.  The returned model recombines both.
    Otherwise mirrors :func:`train`: same curriculum/early-stopping/
    best-model semantics, the same ``apply_if_finite``/``adaptive_grad_clip``
    optimiser wrapping, the same perturb-and-retry recovery from a solver
    failure that raises ``eqx.EquinoxRuntimeError``, and the same graceful
    stop on ``stop_exceptions``.

    Parameters
    ----------
    model : T
        Initial UDE model.
    ts : jax.Array
        Full time-point array.
    ys : jax.Array
        Full observed trajectory, shape ``(T, n_obs)``.
    training_steps : TrainingSteps
        Sequence of ``(n_steps, cutoff)`` pairs; see :func:`train`.
    avg_every : int
        Average and log loss every this many steps.
    target_loss : float
        Early-stopping threshold, checked during the last stage only.
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
        solver failure; see :func:`perturb_model`.
    key : jax.Array or None
        PRNG key seeding the weight perturbations; defaults to
        ``jax.random.PRNGKey(0)``.
    args : jax.Array or None
        External arguments forwarded to ``model.integrate`` at every step;
        see :func:`grad_loss`.
    grad_fn : GradLossSplitFn
        Computes the (loss, grads) pair to apply at every step; defaults
        to :func:`grad_loss_split` (normalised MSE against ``ys``). Pass a
        different ``@eqx.filter_value_and_grad`` function with the same
        call signature to train against a different loss without
        reimplementing this curriculum/robustness loop.
    stop_exceptions : tuple[type[BaseException], ...]
        Exception types that trigger a graceful stop (return the best
        model and history accumulated so far) instead of propagating;
        defaults to just ``KeyboardInterrupt``. See :func:`train`.
    prior_losses : LossesPerLesson or None
        Loss history to resume from; see :func:`train`.
    prior_grad_norms : GradNormsPerLesson or None
        Grad-norm history to resume from; see :func:`train`.

    Returns
    -------
    tuple[T, LossesPerLesson, GradNormsPerLesson]
        Best model encountered during training, losses per curriculum
        lesson, and per-step gradient norms per curriculum lesson --
        including any ``prior_losses``/``prior_grad_norms`` prepended.
    """
    loss = 0
    acc_loss = 0
    acc_count = 0
    global_step = 0
    total_steps = sum(s for s, _ in training_steps)
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

    filter_spec = jax.tree.map(eqx.is_inexact_array, model)
    filter_spec = eqx.tree_at(
        lambda m: m.ode,
        filter_spec,
        replace_fn=lambda _: False,
    )
    trainable, frozen = eqx.partition(model, filter_spec)

    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)
    losses_per_lesson: LossesPerLesson = (
        [] if prior_losses is None else list(prior_losses)
    )
    grad_norms_per_lesson: GradNormsPerLesson = (
        [] if prior_grad_norms is None else list(prior_grad_norms)
    )

    def _finish() -> tuple[T, LossesPerLesson, GradNormsPerLesson]:
        return (
            best_model,
            [pd.Series(i) for i in losses_per_lesson],
            [pd.Series(i) for i in grad_norms_per_lesson],
        )

    try:
        for i, (steps, cutoff) in enumerate(training_steps, start=1):
            opt_state = optim.init(trainable)
            # Fresh per stage, same as opt_state/losses/grad_norms below --
            # a stage's retry budget must not be affected by failures that
            # already happened in a previous stage.
            consecutive_solver_errors = 0
            length = _curriculum_length(len(ts), cutoff)
            _ts = ts[:length]
            _ys = ys[:length]
            losses: dict[int, float] = {}
            grad_norms: dict[int, float] = {}
            losses_per_lesson.append(losses)
            grad_norms_per_lesson.append(grad_norms)

            with trange(steps, position=1, leave=True, dynamic_ncols=True) as pbar:
                for step in pbar:
                    # make_step_split's returned loss describes trainable as
                    # it was BEFORE this step's update; see train()'s
                    # equivalent comment.
                    pre_step_trainable = trainable
                    try:
                        loss, trainable, opt_state, grad_norm = make_step_split(
                            trainable=trainable,
                            frozen=frozen,
                            ts=_ts,
                            ys=_ys,
                            opt_state=opt_state,
                            optim=optim,
                            y_mean=y_mean,
                            y_scale=y_scale,
                            ctx=ctx,
                            global_step=jnp.asarray(global_step),
                            total_steps=jnp.asarray(total_steps),
                            args=args,
                            grad_fn=grad_fn,
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
                        trainable = perturb_model(trainable, subkey, perturbation_scale)
                        logger.warning(
                            "Lesson %d, step %d: solver hit a stiff domain (%s); "
                            "nudging model parameters and retrying.",
                            i,
                            step,
                            e,
                        )
                        continue
                    consecutive_solver_errors = 0
                    global_step += 1
                    grad_norms[step] = float(grad_norm)
                    acc_loss += loss
                    acc_count += 1

                    if loss < best_training_loss:
                        best_model = eqx.combine(pre_step_trainable, frozen)
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
    except stop_exceptions as e:
        logger.warning("Training stopped (%s).", type(e).__name__)

    return _finish()


def train_protocol[Model: JaxModel](
    model: Model,
    *,
    y0: jax.Array,
    ts: list[jax.Array],
    ys: jax.Array,
    protocol: jax.Array,
    training_steps: TrainingSteps,
    target_loss: float = 1e-4,
    avg_every: int = 1000,
    optim: optax.GradientTransformationExtraArgs | None = None,
    integration_settings: IntegrationSettings | None = None,
    simulation_fn: ProtoSimulationFn = integrate_protocol_states,
    clip_norm: float = 1.0,
    max_consecutive_nonfinite: int = 50,
    max_consecutive_solver_errors: int = 10,
    perturbation_scale: float = 1e-3,
    key: jax.Array | None = None,
    grad_fn: ProtoGradLossFn = proto_grad_loss,
    stop_exceptions: tuple[type[BaseException], ...] = (KeyboardInterrupt,),
    prior_losses: LossesPerLesson | None = None,
    prior_grad_norms: GradNormsPerLesson | None = None,
    grad_pars_history: GradParsHistory | None = None,
    normalize_axis: int | None = None,
    disable_tqdm: bool = False,
    freeze: FreezeFn | None = None,
    split_grad_fn: ProtoGradLossSplitFn = proto_grad_loss_split,
) -> tuple[Model, LossesPerLesson, GradNormsPerLesson]:
    """Train a JAX model through a sequence of curriculum steps.

    Each curriculum stage re-initialises the optimiser and uses a
    different fraction of the data.  Early stopping triggers when
    ``loss < target_loss`` during the **last** stage.

    ``optim`` is wrapped in ``optax.apply_if_finite``/``optax.adaptive_grad_clip``
    the same way :func:`train` wraps it -- see its docstring. ``proto_make_step``/
    ``proto_make_step_split`` additionally zero non-finite gradient entries and
    clip by global norm before this wrapping ever sees them, so in practice
    ``apply_if_finite`` rarely has a non-finite update left to skip; the two
    mechanisms are complementary defence in depth, not a replacement for
    each other.

    A solver failure that raises (``eqx.EquinoxRuntimeError``, e.g. from a
    diffrax/lineax integration hitting a stiff or singular domain) is
    recovered from by perturbing the model's trainable weights with small
    multiplicative noise and retrying the same step, up to
    ``max_consecutive_solver_errors`` failures in a row before giving up on
    that curriculum stage and moving to the next one; the failure counter
    resets at the start of each stage. Without this, such a failure would
    propagate and abort the whole run -- and since nothing else in this
    function is stochastic, simply retrying the run from scratch would hit
    the identical failure again. A ``KeyboardInterrupt`` (or any other
    exception listed in ``stop_exceptions``) returns the best model and
    history accumulated so far instead of propagating; see :func:`train`.

    Parameters
    ----------
    model : T
        Initial model state.
    y0 : jax.Array
        Physiological state at ``t=0``, used to seed integration; see
        :func:`proto_grad_loss`.
    ts : list[jax.Array]
        Full per-step, ascending, absolute time points, one array per
        protocol step; see :meth:`JaxModel.integrate_protocol`.
    ys : jax.Array
        Full observed trajectory, shape
        ``(sum(len(t) for t in ts) + 1, n_obs)``. ``ys[0]`` is the
        observation at ``t=0``.
    protocol : jax.Array
        Full per-step external argument, shape ``(len(ts), n_args)``.
    training_steps : TrainingSteps
        Sequence of ``(n_steps, cutoff)`` pairs; ``cutoff`` selects a
        prefix of ``ts``/``protocol`` (by step count) and the matching
        prefix of ``ys`` (by observation count). A ``float`` is a fraction
        of the full step count (rounded up); an ``int`` is an explicit
        step count, clipped to the full step count if it overshoots.
    target_loss : float
        Early-stopping threshold, checked during the last stage only.
    avg_every : int
        Average and log loss every this many steps.
    optim : optax.GradientTransformationExtraArgs or None
        Optimiser; defaults to AdaBelief with ``lr=1e-4``.  Wrapped in
        ``apply_if_finite``/``adaptive_grad_clip`` before use; pass the raw
        optimiser here, not a pre-wrapped one -- see :func:`train`.
    integration_settings : IntegrationSettings or None
        ODE solver settings; defaults to ``IntegrationSettings()``.
    simulation_fn : ProtoSimulationFn
        Maps ``(model, ts, y0, protocol, ctx)`` to predictions aligned with
        ``ys[1:]``; defaults to raw-state integration
        (:func:`integrate_protocol_states`). Pass
        :func:`derived_simulation_fn` (or a custom function with the same
        signature) for models that fit a derived observable (e.g.
        fluorescence) instead of raw state.
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
        solver failure; see :func:`perturb_model`.
    key : jax.Array or None
        PRNG key seeding the weight perturbations; defaults to
        ``jax.random.PRNGKey(0)``.
    grad_fn : ProtoGradLossFn
        Computes the (loss, grads) pair to apply at every step; defaults
        to :func:`proto_grad_loss` (normalised MSE against ``ys``). Pass a
        different ``@eqx.filter_value_and_grad`` function with the same
        call signature to train against a different loss without
        reimplementing this curriculum/robustness loop.
    stop_exceptions : tuple[type[BaseException], ...]
        Exception types that trigger a graceful stop (return the best
        model and history accumulated so far) instead of propagating;
        defaults to just ``KeyboardInterrupt``. See :func:`train`.
    prior_losses : LossesPerLesson or None
        Loss history to resume from; see :func:`train`.
    prior_grad_norms : GradNormsPerLesson or None
        Grad-norm history to resume from; see :func:`train`.
    grad_pars_history : GradParsHistory or None
        If given, records the raw gradient of ``model.pars`` at every
        successful step into it; see :func:`train`.
    normalize_axis : int or None
        Axis passed to ``jnp.mean``/``jnp.std`` when computing ``y_mean``/
        ``y_scale`` from ``ys``; see :func:`train`.
    disable_tqdm : bool
        Suppress the per-step progress bar; see :func:`train`.
    freeze : FreezeFn or None
        If given, an ``eqx.tree_at``-style selector picking the subtree of
        ``model`` to hold fixed; everything else is trained. See
        :func:`train`'s ``freeze``, which this mirrors.
    split_grad_fn : ProtoGradLossSplitFn
        Computes the (loss, grads) pair to apply at every step when
        ``freeze`` is given; defaults to :func:`proto_grad_loss_split`.
        Ignored when ``freeze`` is ``None`` (use ``grad_fn`` instead).

    Returns
    -------
    tuple[T, LossesPerLesson, GradNormsPerLesson]
        Best model encountered during training, losses per curriculum
        lesson, and per-step gradient norms per curriculum lesson --
        including any ``prior_losses``/``prior_grad_norms`` prepended.
    """
    if freeze is not None and grad_pars_history is not None:
        msg = "grad_pars_history is not supported together with freeze"
        raise ValueError(msg)

    loss = 0
    acc_loss = 0
    acc_count = 0
    global_step = 0
    total_steps = sum(s for s, _ in training_steps)

    raw_optim = optax.adabelief(learning_rate=1e-4) if optim is None else optim
    optim = optax.apply_if_finite(
        optax.chain(optax.adaptive_grad_clip(clip_norm), raw_optim),
        max_consecutive_errors=max_consecutive_nonfinite,
    )

    best_training_loss = jnp.inf
    best_model = model
    perturb_key: jax.Array = jax.random.PRNGKey(0) if key is None else key

    ctx = (
        IntegrationSettings() if integration_settings is None else integration_settings
    )

    y_mean = jnp.mean(ys, axis=normalize_axis)
    y_scale = jnp.std(ys, axis=normalize_axis)

    filter_spec = None
    if freeze is not None:
        filter_spec = jax.tree.map(eqx.is_inexact_array, model)
        filter_spec = eqx.tree_at(freeze, filter_spec, replace_fn=lambda _: False)

    def _init_opt_state(model: Model) -> optax.OptState:
        if filter_spec is None:
            return optim.init(eqx.filter(model, eqx.is_inexact_array))
        trainable, _frozen = eqx.partition(model, filter_spec)
        return optim.init(trainable)

    def _step(
        model: Model, opt_state: optax.OptState
    ) -> tuple[jax.Array, Model, optax.OptState, jax.Array]:
        if filter_spec is None:
            return proto_make_step(
                model=model,
                y0=y0,
                ts=_ts,
                ys=_ys,
                protocol=_protocol,
                opt_state=opt_state,
                optim=optim,
                y_mean=y_mean,
                y_scale=y_scale,
                ctx=ctx,
                simulation_fn=simulation_fn,
                global_step=jnp.asarray(global_step),
                total_steps=jnp.asarray(total_steps),
                grad_fn=grad_fn,
            )
        trainable, frozen = eqx.partition(model, filter_spec)
        loss, trainable, opt_state, grad_norm = proto_make_step_split(
            trainable=trainable,
            frozen=frozen,
            y0=y0,
            ts=_ts,
            ys=_ys,
            protocol=_protocol,
            opt_state=opt_state,
            optim=optim,
            y_mean=y_mean,
            y_scale=y_scale,
            ctx=ctx,
            simulation_fn=simulation_fn,
            global_step=jnp.asarray(global_step),
            total_steps=jnp.asarray(total_steps),
            grad_fn=split_grad_fn,
        )
        return loss, eqx.combine(trainable, frozen), opt_state, grad_norm

    def _perturb(model: Model, key: jax.Array, scale: float) -> Model:
        if filter_spec is None:
            return perturb_model(model, key, scale)
        trainable, frozen = eqx.partition(model, filter_spec)
        return eqx.combine(perturb_model(trainable, key, scale), frozen)

    losses_per_lesson: LossesPerLesson = (
        [] if prior_losses is None else list(prior_losses)
    )
    grad_norms_per_lesson: GradNormsPerLesson = (
        [] if prior_grad_norms is None else list(prior_grad_norms)
    )

    def _finish() -> tuple[Model, LossesPerLesson, GradNormsPerLesson]:
        return (
            best_model,
            [pd.Series(i) for i in losses_per_lesson],
            [pd.Series(i) for i in grad_norms_per_lesson],
        )

    try:
        for i, (steps, cutoff) in enumerate(training_steps, start=1):
            opt_state = _init_opt_state(model)
            # Fresh per stage, same as opt_state/losses/grad_norms below -- a
            # stage's retry budget must not be affected by failures that
            # already happened in a previous stage.
            consecutive_solver_errors = 0
            n_windows = _curriculum_length(len(ts), cutoff)
            _ts = ts[:n_windows]
            _protocol = protocol[:n_windows]
            n_obs = sum(t.shape[0] for t in _ts)
            _ys = ys[: n_obs + 1]
            losses: dict[int, float] = {}
            grad_norms: dict[int, float] = {}
            losses_per_lesson.append(losses)
            grad_norms_per_lesson.append(grad_norms)
            if grad_pars_history is not None:
                grad_pars_history._start_lesson()  # noqa: SLF001

            with trange(
                steps, position=1, leave=True, dynamic_ncols=True, disable=disable_tqdm
            ) as pbar:
                for step in pbar:
                    # proto_make_step's returned loss describes the model
                    # as it was BEFORE this step's update; see train()'s
                    # equivalent comment.
                    pre_step_model = model
                    try:
                        if grad_pars_history is None:
                            loss, model, opt_state, grad_norm = _step(model, opt_state)
                        else:
                            loss, model, opt_state, grad_norm, grad_pars = (
                                _proto_make_step_with_grad_pars(
                                    model=model,
                                    y0=y0,
                                    ts=_ts,
                                    ys=_ys,
                                    protocol=_protocol,
                                    opt_state=opt_state,
                                    optim=optim,
                                    y_mean=y_mean,
                                    y_scale=y_scale,
                                    ctx=ctx,
                                    simulation_fn=simulation_fn,
                                    global_step=jnp.asarray(global_step),
                                    total_steps=jnp.asarray(total_steps),
                                    grad_fn=grad_fn,
                                )
                            )
                            grad_pars_history._record(step, grad_pars)  # noqa: SLF001
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
                        model = _perturb(model, subkey, perturbation_scale)
                        logger.warning(
                            "Lesson %d, step %d: solver hit a stiff domain (%s); "
                            "nudging model parameters and retrying.",
                            i,
                            step,
                            e,
                        )
                        continue
                    consecutive_solver_errors = 0
                    global_step += 1
                    grad_norms[step] = float(grad_norm)
                    acc_loss += loss
                    acc_count += 1

                    if loss < best_training_loss:
                        best_model = pre_step_model
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
    except stop_exceptions as e:
        logger.warning("Training stopped (%s).", type(e).__name__)

    return _finish()
