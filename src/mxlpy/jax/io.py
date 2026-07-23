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


def recast_to_template[T: eqx.Module](
    model: T,
    template: T,
    *,
    backfill: dict[str, Any] | None = None,
) -> T:
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
    backfill : dict[str, Any] or None
        Maps a leaf's :func:`jax.tree_util.keystr` path on ``template``
        (e.g. ``".derived_scale"``) to a default value to substitute for a
        leaf that exists on ``template`` but not on ``model`` -- e.g. a
        field added to the class after ``model`` was pickled. Every leaf
        missing from ``model`` relative to ``template`` must be named here;
        this is schema migration for old pickles, not a general-purpose
        leaf-alignment tool, so it only ever inserts named leaves at their
        known template position -- it never tries to guess where an
        unnamed missing leaf belongs.

    Returns
    -------
    T
        ``model``'s leaves (plus any ``backfill`` insertions), unflattened
        onto ``template``'s treedef.

    Raises
    ------
    ValueError
        If, after accounting for ``backfill``, ``model`` and ``template``
        still don't have the same number of leaves (e.g. because a field
        was added to the class after ``model`` was saved and isn't named in
        ``backfill``). Pass the missing leaves' default values via
        ``backfill``, or backfill them onto ``model`` yourself (e.g. via
        ``eqx.tree_at``) before recasting.
    """
    template_leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)

    if backfill:
        # Align by each leaf's own keystr path, not by position: model's
        # leaves are matched against template's by name, so a backfill dict
        # of the right size but naming the wrong (already-present) leaf can
        # never silently displace a real value into the wrong slot --
        # positional consumption alone can't tell that case apart from a
        # genuine migration. (If model already matches template 1:1 and
        # backfill merely names an already-present, harmless leaf, the
        # substitution below is still rejected -- but the leaf counts then
        # happen to already match, so no migration was needed in the first
        # place and the bogus backfill entry is just quietly unused.)
        model_leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(model)
        remaining = {
            jax.tree_util.keystr(path): leaf for path, leaf in model_leaves_with_path
        }
        matched_backfill_keys: set[str] = set()
        substituted = []
        aligned = True
        for path, _template_leaf in template_leaves_with_path:
            key = jax.tree_util.keystr(path)
            if key in backfill:
                substituted.append(backfill[key])
                matched_backfill_keys.add(key)
            elif key in remaining:
                substituted.append(remaining.pop(key))
            else:
                aligned = False
                break
        # Only accept the substitution if every template leaf resolved (by
        # backfill or by a same-named leaf on model), every one of model's
        # own leaves was matched by name (none left over), and every
        # backfill entry actually named a real template leaf (none unused)
        # -- otherwise fall through to the raw (non-substituted) leaves
        # below, surfacing the mismatch instead of silently accepting a
        # wrong-but-same-length leaf list.
        if aligned and not remaining and matched_backfill_keys == backfill.keys():
            leaves = substituted
        else:
            leaves, _ = jax.tree_util.tree_flatten(model)
    else:
        leaves, _ = jax.tree_util.tree_flatten(model)

    if len(leaves) != len(template_leaves_with_path):
        msg = (
            f"Leaf count mismatch: loaded model has {len(leaves)} leaves, "
            f"template has {len(template_leaves_with_path)} -- the class's "
            "fields changed since this model was saved. Pass the missing "
            "leaves' default values via `backfill` (keyed by "
            "jax.tree_util.keystr against template's leaves), or backfill "
            "them onto model yourself (e.g. via eqx.tree_at) before "
            "recasting."
        )
        raise ValueError(msg)

    return cast(T, jax.tree_util.tree_unflatten(treedef, leaves))
