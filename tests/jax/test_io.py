"""Tests for mxlpy.jax.io: save/load round-trip and recast_to_template."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from mxlpy.jax import io as jax_io
from mxlpy.jax.models import Node

_KEY = jax.random.PRNGKey(0)


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
