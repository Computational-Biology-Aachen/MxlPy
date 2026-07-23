"""Batched ensemble simulation utilities for JAX/equinox models."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["batch_simulate", "stack_models", "unstack_models"]


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


def unstack_models[T](batched: T, n: int) -> list[T]:
    """Split a batched pytree back into ``n`` per-member instances.

    Inverse of :func:`stack_models`: every array leaf's leading axis is
    sliced off member-by-member; non-array/static structure is shared
    (taken directly from ``batched``) since it's identical across members
    by construction. Also applies to :func:`batch_simulate`'s return value
    when it's a batched :class:`~mxlpy.jax.simulation.FluxOdeSimulation` --
    unstacking it recovers per-member results whose ``.variables``/
    ``.fluxes`` accessors work as usual (they assume unbatched arrays).

    Parameters
    ----------
    batched : T
        A pytree whose array leaves all share a leading ``(n,)`` axis, as
        produced by :func:`stack_models` or :func:`batch_simulate`.
    n : int
        Number of members to split out.

    Returns
    -------
    list[T]
        ``n`` instances, each with ``batched``'s array leaves sliced at
        the corresponding index along axis 0.
    """
    params, static = eqx.partition(batched, eqx.is_array)
    return [
        eqx.combine(jax.tree.map(lambda x, i=i: x[i], params), static) for i in range(n)
    ]


@eqx.filter_jit
def _run_one[R](
    model: eqx.Module,
    simulate: Callable[..., R],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> R:
    return simulate(model, *args, **kwargs)


# Wraps a single module-level function rather than one freshly defined (and
# re-jitted/-vmapped) inside every batch_simulate call: JAX's own jit cache
# is keyed by the traced call's abstract signature -- including `simulate`
# and any static substructure of `args`/`kwargs`, since equinox treats
# non-array leaves as static -- so repeat calls with the same `simulate` and
# matching shapes now hit that cache instead of recompiling from scratch
# every time. `args`/`kwargs` are also genuine (broadcast) arguments here
# rather than values closed over from the enclosing scope, so their array
# leaves are traced -- not baked into the compiled program as constants --
# which is what makes the cache hit meaningful across calls with different
# data of the same shape.
_run_batched = eqx.filter_vmap(
    _run_one, in_axes=(eqx.if_array(0), None, None, None)
)


def batch_simulate[T: eqx.Module, R](
    models: list[T],
    simulate: Callable[..., R],
    *args: object,
    **kwargs: object,
) -> R:
    """Run ``simulate`` for every model in ``models`` in one vmapped call.

    ``simulate``'s return value ``R`` must be a genuine JAX pytree (a plain
    array, or a tuple/dict/registered-dataclass of arrays) -- passing a
    function that returns a non-pytree wrapper (e.g. a plain
    ``@dataclass`` that isn't registered as a pytree) silently produces
    leaked tracers in the unbatched fields instead of raising, since
    ``eqx.filter_vmap`` treats an unregistered object as a single opaque
    static leaf rather than batching through it.
    :class:`~mxlpy.jax.simulation.FluxOdeSimulation` is registered as a
    pytree specifically so it composes with this function.

    Parameters
    ----------
    models : list[T]
        Independently-trained instances of the same model class and
        architecture; see :func:`stack_models`.
    simulate : Callable[..., R]
        Function taking a single model (plus ``*args``/``**kwargs``, held
        constant across members) and returning a pytree of arrays.
    *args, **kwargs
        Additional arguments forwarded to ``simulate`` for every member,
        broadcast (not batched) across the ensemble -- so per-member-varying
        auxiliary data (as opposed to per-member-varying model parameters)
        isn't supported.

    Returns
    -------
    R
        ``simulate``'s return pytree, with every array leaf gaining a
        leading ``(n_models,)`` axis.
    """
    stacked = stack_models(models)
    return _run_batched(stacked, simulate, args, kwargs)
