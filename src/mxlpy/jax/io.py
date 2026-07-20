"""Persistence utilities for JAX/equinox models."""

from pathlib import Path
from typing import Any, cast

import dill
import equinox as eqx
import jax

__all__ = ["load", "recast_to_template", "save"]


def save(path: Path | str, model: eqx.Module) -> None:
    """Pickle an equinox model to disk via dill.

    Parameters
    ----------
    path : Path or str
        Destination file.
    model : eqx.Module
        Model to pickle.
    """
    with Path(path).open("wb") as fp:
        dill.dump(model, fp)


def load(path: Path | str) -> Any:
    """Unpickle a model saved by :func:`save`.

    Like any ``pickle``-based loader, this executes arbitrary code embedded
    in the file; only call it on files your own code (or another trusted
    party) wrote via :func:`save`, never on user-supplied or otherwise
    untrusted input.

    Loading a model saved while running as ``__main__`` (e.g. a standalone
    script) from a different entrypoint (a notebook, a different module)
    leaves it bound to a stale, no-longer-matching class identity -- pass
    the result through :func:`recast_to_template` to fix that up before
    using it inside a JAX transformation together with other instances of
    the same class.

    Parameters
    ----------
    path : Path or str
        File written by :func:`save`.

    Returns
    -------
    Any
        The unpickled object.
    """
    with Path(path).open("rb") as fp:
        return dill.load(fp)  # nosec -- trusted, caller-controlled file


def recast_to_template[T: eqx.Module](model: T, template: T) -> T:
    """Coerce a pytree onto ``template``'s treedef.

    dill-pickled equinox modules embed the class's fully-qualified import
    path; loading a model that was pickled while running as ``__main__``
    (e.g. a standalone script) from a different entrypoint (a notebook, a
    different module) leaves it bound to a stale class identity that no
    longer matches the live class, even though the model's structure and
    values are otherwise identical. Mixing pytree nodes of the stale and
    live class inside a JAX transformation (e.g. stacking an ensemble via
    :func:`mxlpy.jax.ensemble.stack_models`) produces baffling errors
    (leaked tracers, missing globals). Re-flattening the loaded model's
    leaves onto a freshly constructed template's treedef fixes the class
    identity while keeping the loaded (e.g. trained) values.

    Parameters
    ----------
    model : T
        The (possibly stale-class) loaded model.
    template : T
        A freshly-constructed instance of the same class and architecture;
        only its pytree treedef is used, its own leaf values are discarded.

    Returns
    -------
    T
        ``model``'s leaves, unflattened onto ``template``'s treedef.

    Raises
    ------
    ValueError
        If ``model`` and ``template`` don't have the same number of leaves
        (e.g. because a field was added to the class after ``model`` was
        saved). There's no general way to know where a new leaf belongs
        without model-specific knowledge, so this raises rather than
        guessing; backfill the missing value onto ``model`` yourself (e.g.
        via ``eqx.tree_at``) before recasting.
    """
    leaves, _ = jax.tree_util.tree_flatten(model)
    template_leaves, treedef = jax.tree_util.tree_flatten(template)

    if len(leaves) != len(template_leaves):
        msg = (
            f"Leaf count mismatch: loaded model has {len(leaves)} leaves, "
            f"template has {len(template_leaves)} -- the class's fields "
            "changed since this model was saved. Backfill the missing "
            "leaves' values onto model before recasting."
        )
        raise ValueError(msg)

    return cast(T, jax.tree_util.tree_unflatten(treedef, leaves))
