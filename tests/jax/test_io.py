"""Tests for mxlpy.jax.io: save/load round-trip and recast_to_template."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from mxlpy.jax import io as jax_io
from mxlpy.jax.models import Node

_KEY = jax.random.PRNGKey(0)


class _ModelV1(eqx.Module):
    a: jax.Array


class _ModelV2(eqx.Module):
    """Simulates _ModelV1 gaining a field (e.g. derived_scale) later."""

    a: jax.Array
    b: jax.Array


class _ModelV1c(eqx.Module):
    """Same shape as _ModelV1, but with a *different* already-present field."""

    a: jax.Array
    c: jax.Array


class _ModelV2c(eqx.Module):
    """_ModelV1c gaining a field named 'b' (alphabetically between a and c)."""

    a: jax.Array
    b: jax.Array
    c: jax.Array


def test_save_load_round_trips_a_model(tmp_path: Path) -> None:
    model = Node(n_obs=2, width=4, depth=1, key=_KEY)
    path = tmp_path / "model.dill"

    jax_io.save(path, model)
    loaded = jax_io.load(path)

    assert isinstance(loaded, Node)
    assert jnp.allclose(loaded.out_scale, model.out_scale)


def test_recast_to_template_fixes_treedef() -> None:
    model = Node(n_obs=2, width=4, depth=1, key=_KEY)
    template = Node(n_obs=2, width=4, depth=1, key=_KEY)

    recast = jax_io.recast_to_template(model, template)

    assert type(recast) is type(template)
    assert jnp.allclose(recast.out_scale, model.out_scale)


def test_recast_to_template_raises_on_leaf_count_mismatch() -> None:
    model = Node(n_obs=2, width=4, depth=1, key=_KEY)
    # Different depth -> different number of MLP layers -> different leaf
    # count, not just different shapes.
    template = Node(n_obs=2, width=4, depth=2, key=_KEY)

    with pytest.raises(ValueError, match="Leaf count mismatch"):
        jax_io.recast_to_template(model, template)


# ---------------------------------------------------------------------------
# recast_to_template(..., backfill=...): schema migration for old pickles
# missing a field that was added to the class later.
# ---------------------------------------------------------------------------


def test_recast_to_template_backfills_a_missing_field_by_name() -> None:
    old = _ModelV1(a=jnp.array([1.0]))
    template = _ModelV2(a=jnp.array([0.0]), b=jnp.array([9.0]))

    recast = jax_io.recast_to_template(
        old, template, backfill={".b": jnp.array([9.0])}
    )

    assert type(recast) is _ModelV2
    assert jnp.allclose(recast.a, old.a)  # old model's own value, kept
    assert jnp.allclose(recast.b, jnp.array([9.0]))  # backfilled default


def test_recast_to_template_backfill_key_typo_still_raises() -> None:
    """A backfill key that doesn't match any real template leaf path must
    not silently produce a wrong-but-same-length leaf list.
    """
    old = _ModelV1(a=jnp.array([1.0]))
    template = _ModelV2(a=jnp.array([0.0]), b=jnp.array([9.0]))

    with pytest.raises(ValueError, match="Leaf count mismatch"):
        jax_io.recast_to_template(
            old, template, backfill={".wrong_name": jnp.array([9.0])}
        )


def test_recast_to_template_backfill_does_not_mask_unrelated_mismatch() -> None:
    """backfill covering the right number of leaves doesn't paper over a
    genuinely different mismatch (here: model has 2 fewer leaves than
    backfill accounts for, since old2 dropped both new fields itself).
    """

    class _ModelV3(eqx.Module):
        a: jax.Array
        b: jax.Array
        c: jax.Array

    old = _ModelV1(a=jnp.array([1.0]))
    template = _ModelV3(a=jnp.array([0.0]), b=jnp.array([9.0]), c=jnp.array([8.0]))

    with pytest.raises(ValueError, match="Leaf count mismatch"):
        jax_io.recast_to_template(old, template, backfill={".b": jnp.array([9.0])})


def test_recast_to_template_backfill_wrong_target_leaf_still_raises() -> None:
    """A backfill dict of exactly the right *size* but naming an
    already-present leaf (instead of the genuinely missing one) must not
    silently displace model's own value into the wrong slot -- alignment
    has to be by name, not just by matching leaf counts.
    """
    old = _ModelV1c(a=jnp.array([1.0]), c=jnp.array([3.0]))
    template = _ModelV2c(
        a=jnp.array([0.0]), b=jnp.array([9.0]), c=jnp.array([0.0])
    )

    # ".b" is the genuinely missing leaf; ".c" already exists on old.
    with pytest.raises(ValueError, match="Leaf count mismatch"):
        jax_io.recast_to_template(
            old, template, backfill={".c": jnp.array([999.0])}
        )
