"""Batched ensemble simulation utilities for JAX/equinox models."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["batch_simulate", "stack_models"]


def stack_models[T: eqx.Module](models: list[T]) -> T:
    """Stack N same-architecture models into one batched pytree.

    Only the array leaves are stacked (via ``eqx.partition``); non-array
    (static) leaves -- architecture ints, shared mechanistic functions --
    are taken from the first model, since they're identical across the
    ensemble by construction (same class, same hyperparameters). This is
    only correct as long as every per-member-varying piece of the model is
    a genuine array leaf, not something hidden in a static/non-pytree
    field (e.g. a :class:`~mxlpy.jax.models.LatentMapper` subclass that
    isn't a real ``eqx.Module`` would silently leak one member's value into
    every other member here).

    Parameters
    ----------
    models : list[T]
        Independently-trained instances of the same model class and
        architecture (e.g. different random seeds).

    Returns
    -------
    T
        A single model instance whose array leaves have an extra leading
        ``(n_models,)`` axis -- pass it through :func:`batch_simulate` (or
        ``eqx.filter_vmap`` directly) to run every member in one call.
    """
    params_list, static_list = zip(
        *[eqx.partition(m, eqx.is_array) for m in models], strict=True
    )
    params_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)
    return eqx.combine(params_batched, static_list[0])


def batch_simulate[T: eqx.Module, R](
    models: list[T],
    simulate: Callable[..., R],
    *args: object,
    **kwargs: object,
) -> R:
    """Run ``simulate`` for every model in ``models`` in one vmapped call.

    Parameters
    ----------
    models : list[T]
        Independently-trained instances of the same model class and
        architecture; see :func:`stack_models`.
    simulate : Callable[..., R]
        Function taking a single model (plus ``*args``/``**kwargs``, held
        constant across members) and returning an array/pytree of arrays.
    *args, **kwargs
        Additional arguments forwarded to ``simulate`` for every member,
        unbatched.

    Returns
    -------
    R
        ``simulate``'s return pytree, with every array leaf gaining a
        leading ``(n_models,)`` axis.
    """
    stacked = stack_models(models)

    @eqx.filter_jit
    def _run(model: T) -> R:
        return simulate(model, *args, **kwargs)

    return eqx.filter_vmap(_run)(stacked)
