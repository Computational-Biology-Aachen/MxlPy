# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Tests for the neural JAX model classes: Node, FluxNode, Anode, FluxAnode, Ude, FluxUde, AugUde, FluxAugUde.

Until now these classes had zero test coverage, which is exactly how a real
``AttributeError`` in ``FluxUde.__call__`` (it referenced a field that didn't
exist) and a shape bug in ``FluxNode`` (no ``n_args`` slot despite accepting
``args``) went unnoticed.
"""

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mxlpy.jax.ensemble import batch_simulate, unstack_models
from mxlpy.jax.models import (
    Anode,
    AugUde,
    FluxAnode,
    FluxAugUde,
    FluxNode,
    FluxOde,
    FluxUde,
    FullLatentMapper,
    HardLatentMapper,
    LatentMapperFn,
    MixedLatentMapper,
    Node,
    Ode,
    SoftLatentMapper,
    TrainableZ0LatentMapper,
    Ude,
)
from mxlpy.jax.simulation import Simulation

_KEY = jax.random.PRNGKey(0)


def _names(n: int, prefix: str = "x") -> tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(n))


def _flux_ode() -> FluxOde:
    # n_obs == n_flux == 2 throughout, so nv (flux_vector -> dy/dt) can be a
    # simple shape-preserving map -- self-consistent enough to actually
    # integrate, not just construct.
    def fluxes(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([y[0], y[1]])

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([7.0])

    return FluxOde(
        flux_fn=fluxes,
        nv=nv,
        pars=jnp.array([]),
        variable_names=("a", "b"),
        flux_names=("ra", "rb"),
        readout_fn=readout_fn,
        readout_names=("r",),
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def test_node_forward_pass_shape() -> None:
    node = Node(variable_names=_names(3), width=4, depth=1, key=_KEY)
    dy = node(0.0, jnp.ones(3), jnp.array([]))
    assert dy.shape == (3,)


def test_node_readout_scale_multiplies_readout_fn_output() -> None:
    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([7.0])

    node = Node(
        variable_names=_names(2),
        width=4,
        depth=1,
        key=_KEY,
        readout_fn=readout_fn,
        readout_names=("r",),
        readout_scale=jnp.array(2.0),
    )
    d = node.readouts(0.0, jnp.ones(2), jnp.array([]))
    assert float(d[0]) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# FluxNode
# ---------------------------------------------------------------------------


def test_fluxnode_forward_pass_shape() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector[:1] - flux_vector[1:]

    fnode = FluxNode(
        variable_names=_names(2),
        flux_names=_names(2, "v"),
        width=4,
        depth=1,
        key=_KEY,
        nv=nv,
    )
    dy = fnode(0.0, jnp.ones(2), jnp.array([]))
    assert dy.shape == (1,)


def test_fluxnode_with_nonzero_n_args_does_not_raise() -> None:
    """Regression: flux_nn's in_size previously ignored n_args entirely, so
    any call with non-empty args raised a matmul shape error inside
    eqx.nn.MLP instead of running.
    """

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fnode = FluxNode(
        variable_names=_names(2),
        flux_names=_names(2, "v"),
        width=4,
        depth=1,
        key=_KEY,
        nv=nv,
        n_args=1,
    )
    dy = fnode(0.0, jnp.ones(2), jnp.array([0.5]))
    assert dy.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(dy)))


def test_fluxnode_readout_scale_multiplies_readout_fn_output() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([5.0])

    fnode = FluxNode(
        variable_names=_names(2),
        flux_names=_names(2, "v"),
        width=4,
        depth=1,
        key=_KEY,
        nv=nv,
        readout_fn=readout_fn,
        readout_names=("r",),
        readout_scale=jnp.array(3.0),
    )
    d = fnode.readouts(0.0, jnp.ones(2), jnp.array([]))
    assert float(d[0]) == pytest.approx(15.0)


def test_fluxnode_from_flux_ode_reuses_dims_nv_and_readout_fn() -> None:
    """Regression: constructing a FluxNode alongside its mechanistic
    FluxOde used to require the caller to separately compute n_obs/n_flux
    (e.g. len(model.get_variable_names())) and pass the same nv/readout_fn
    to both by hand.
    """
    flux_ode = _flux_ode()

    fnode = FluxNode.from_flux_ode(flux_ode, width=4, depth=1, key=_KEY)

    assert fnode.nv is flux_ode.nv
    assert fnode.readout_fn is flux_ode.readout_fn
    dy = fnode(0.0, jnp.ones(2), jnp.array([]))
    assert dy.shape == (2,)  # n_obs, derived from flux_ode
    assert bool(jnp.all(jnp.isfinite(dy)))


def test_fluxnode_simulate_time_course() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    sim = fnode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.ones(2), max_steps=4096)

    assert list(sim.variables_df.columns) == ["a", "b"]
    assert sim.variables_df.shape == (2, 2)
    assert list(sim.fluxes_df.columns) == ["ra", "rb"]
    assert sim.fluxes.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(sim.ys)))


def test_fluxnode_simulate_protocol_time_course_prepends_t0() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    sim = fnode.simulate_protocol_time_course(ts, jnp.ones(2), protocol, max_steps=4096)

    assert sim.variables_df.shape == (3, 2)  # t=0 row prepended
    assert float(sim.ts[0]) == 0.0


def test_fluxnode_simulate_to_steady_state() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    sim = fnode.simulate_to_steady_state(jnp.ones(2), max_steps=8192)

    assert sim.variables_df.shape == (1, 2)
    assert sim.fluxes.shape == (1, 2)


# ---------------------------------------------------------------------------
# LatentMapper: MixedLatentMapper, TrainableZ0LatentMapper
# ---------------------------------------------------------------------------


def test_mixed_latent_mapper_decode_requires_alpha() -> None:
    mapper = MixedLatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    zs = jnp.ones((3, 3))
    with pytest.raises(ValueError, match="alpha is required"):
        mapper.decode(zs)


def test_mixed_latent_mapper_decode_alpha_zero_is_pure_hard_slice() -> None:
    mapper = MixedLatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    zs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert jnp.allclose(mapper.decode(zs, alpha=0.0), zs[:, :2])


def test_mixed_latent_mapper_decode_alpha_one_is_pure_learned() -> None:
    mapper = MixedLatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    zs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert jnp.allclose(mapper.decode(zs, alpha=1.0), jax.vmap(mapper.decoder)(zs))


def test_trainable_z0_latent_mapper_decode_requires_alpha() -> None:
    mapper = TrainableZ0LatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    zs = jnp.ones((3, 3))
    with pytest.raises(ValueError, match="alpha is required"):
        mapper.decode(zs)


def test_trainable_z0_latent_mapper_decode_alpha_zero_is_pure_hard_slice() -> None:
    mapper = TrainableZ0LatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    zs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert jnp.allclose(mapper.decode(zs, alpha=0.0), zs[:, :2])


def test_trainable_z0_latent_mapper_encode_keeps_observed_slice_verbatim() -> None:
    mapper = TrainableZ0LatentMapper(n_obs=2, n_hidden=1, key=_KEY)
    y0 = jnp.array([1.0, 2.0])
    z0 = mapper.encode(y0)
    assert jnp.allclose(z0[:2], y0)


def test_trainable_z0_latent_mapper_hidden_slice_is_independent_of_y0() -> None:
    """The point of TrainableZ0LatentMapper: unlike every other LatentMapper,
    the hidden dims come from a fixed trainable field, not a function of y0.
    """
    mapper = TrainableZ0LatentMapper(n_obs=2, n_hidden=3, key=_KEY)
    z_a = mapper.encode(jnp.array([1.0, 2.0]))
    z_b = mapper.encode(jnp.array([-5.0, 100.0]))
    assert jnp.allclose(z_a[2:], z_b[2:])
    assert jnp.allclose(z_a[2:], mapper.z0)


def test_trainable_z0_latent_mapper_z0_init_range_is_configurable() -> None:
    mapper = TrainableZ0LatentMapper(
        n_obs=1, n_hidden=64, key=_KEY, z0_minval=5.0, z0_maxval=6.0
    )
    assert bool(jnp.all(mapper.z0 >= 5.0))
    assert bool(jnp.all(mapper.z0 <= 6.0))


# ---------------------------------------------------------------------------
# Anode
# ---------------------------------------------------------------------------


def _anode(n_args: int = 0) -> Anode:
    # HardLatentMapper's decode is a deterministic slice, so trajectory
    # shapes (rather than learned values) are what's worth asserting on.
    return Anode(
        variable_names=_names(2),
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        n_args=n_args,
        latent_mapper=HardLatentMapper,
    )


def test_anode_integrate_decodes_to_n_obs() -> None:
    anode = _anode()
    ts = jnp.array([0.0, 1.0])
    ys = anode.integrate(ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]))
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_anode_integrate_latent_is_integrate_without_decode() -> None:
    anode = _anode()
    ts = jnp.array([0.0, 1.0])
    zs, _ = anode._integrate_latent(
        ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]),
        rtol=1e-6, atol=1e-6, method=diffrax.Tsit5, dt0=None,
    )  # fmt: skip
    ys = anode.integrate(ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]))
    assert zs.shape[0] == 2
    assert zs.shape[1] > ys.shape[1]
    assert jnp.allclose(anode.latent_mapper.decode(zs), ys)


def test_anode_integrate_protocol_decodes_to_n_obs() -> None:
    """Regression: integrate_protocol used to be inherited unmodified from
    Base, feeding the raw n_obs-length y0 straight into a latent-space nn
    that expects n_obs+n_hidden(+n_args) inputs.
    """
    anode = _anode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = anode.integrate_protocol(ts, jnp.array([1.0, 2.0]), protocol, max_steps=4096)
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_anode_integrate_to_steady_state_decodes_to_n_obs() -> None:
    anode = _anode()
    t, y = anode.integrate_to_steady_state(
        jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([])
    )
    assert t.shape == ()
    assert y.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_anode_integrate_protocol_from_steady_state_decodes_to_n_obs() -> None:
    """Regression: the composed Base method must dispatch through the
    (latent-aware) public integrate_to_steady_state/integrate_protocol
    overrides, not the raw state-space helpers, without needing its own
    override on Anode.
    """
    anode = _anode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = anode.integrate_protocol_from_steady_state(
        ts, jnp.array([1.0, 2.0]), protocol, max_steps=4096
    )
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_anode_n_args_field_stores_raw_n_args() -> None:
    """Regression: Anode used to store n_obs + n_args in self.n_args, while
    FluxAnode stored plain n_args for the same-named field -- see
    test_fluxanode_n_args_field_stores_raw_n_args for the counterpart.
    """
    anode = _anode(n_args=3)
    assert anode.n_args == 3


def test_anode_integrate_alpha_defaults_to_none_for_non_blending_mapper() -> None:
    # HardLatentMapper ignores alpha entirely -- omitting it must still work.
    anode = _anode()
    ts = jnp.array([0.0, 1.0])
    ys = anode.integrate(ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]))
    assert ys.shape == (2, 2)


def test_anode_integrate_threads_alpha_to_blending_latent_mapper() -> None:
    anode = Anode(
        variable_names=_names(2),
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        latent_mapper=MixedLatentMapper,
    )
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])
    ys_hard = anode.integrate(ts, y0, max_steps=4096, args=jnp.array([]), alpha=0.0)
    ys_learned = anode.integrate(ts, y0, max_steps=4096, args=jnp.array([]), alpha=1.0)
    assert not jnp.allclose(ys_hard, ys_learned)
    with pytest.raises(ValueError, match="alpha is required"):
        anode.integrate(ts, y0, max_steps=4096, args=jnp.array([]))


def test_anode_readout_scale_multiplies_readout_fn_output() -> None:
    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([3.0])

    anode = Anode(
        variable_names=_names(2),
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        readout_fn=readout_fn,
        readout_names=("r",),
        readout_scale=jnp.array(2.0),
    )
    d = anode.readouts(0.0, jnp.ones(3), jnp.array([]))
    assert float(d[0]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# FluxAnode
# ---------------------------------------------------------------------------


def _fluxanode() -> FluxAnode:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    return FluxAnode(
        variable_names=_names(2),
        n_hidden=1,
        flux_names=_names(2, "v"),
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=nv,
        latent_mapper=HardLatentMapper,
    )


def test_fluxanode_n_args_defaults_to_zero() -> None:
    # Regression: n_args used to be a required positional argument even
    # though Anode's equivalent already defaulted to 0.
    _fluxanode()


def test_fluxanode_n_args_field_stores_raw_n_args() -> None:
    """Regression: see test_anode_n_args_field_stores_raw_n_args -- both
    classes must store the same, unmodified quantity in self.n_args.
    """

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fanode = FluxAnode(
        variable_names=_names(2),
        n_hidden=1,
        flux_names=_names(2, "v"),
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=nv,
        n_args=3,
        latent_mapper=HardLatentMapper,
    )
    assert fanode.n_args == 3


def test_fluxanode_integrate_threads_alpha_to_blending_latent_mapper() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fanode = FluxAnode(
        variable_names=_names(2),
        n_hidden=1,
        flux_names=_names(2, "v"),
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=nv,
        latent_mapper=MixedLatentMapper,
    )
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])
    ys_hard = fanode.integrate(ts, y0, max_steps=4096, args=jnp.array([]), alpha=0.0)
    ys_learned = fanode.integrate(ts, y0, max_steps=4096, args=jnp.array([]), alpha=1.0)
    assert not jnp.allclose(ys_hard, ys_learned)
    with pytest.raises(ValueError, match="alpha is required"):
        fanode.integrate(ts, y0, max_steps=4096, args=jnp.array([]))


def test_fluxanode_integrate_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    ys = fanode.integrate(ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]))
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_integrate_latent_is_integrate_without_decode() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    zs, _ = fanode._integrate_latent(
        ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]),
        rtol=1e-6, atol=1e-6, method=diffrax.Tsit5, dt0=None,
    )  # fmt: skip
    ys = fanode.integrate(ts, jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([]))
    assert zs.shape[0] == 2
    assert zs.shape[1] > ys.shape[1]
    assert jnp.allclose(fanode.latent_mapper.decode(zs), ys)


def test_fluxanode_integrate_dt0_defaults_to_none_and_is_unaffected() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])

    ys_omitted = fanode.integrate(ts, y0, max_steps=4096, args=jnp.array([]))
    ys_explicit_none = fanode.integrate(
        ts, y0, max_steps=4096, args=jnp.array([]), dt0=None
    )

    assert jnp.array_equal(ys_omitted, ys_explicit_none)


def test_fluxanode_integrate_accepts_explicit_dt0() -> None:
    """A model whose RHS needs a specific initial step (e.g. to line up
    with a driver signal's fixed spacing) can pass dt0 through integrate()
    instead of reimplementing _solve just to set it.

    Asserts more than shape/finiteness -- see
    test_ode_integrate_accepts_explicit_dt0's docstring for why a
    silently-dropped dt0 would still pass a shape/finiteness-only check
    under a PID step-size controller.
    """
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])

    ys_default = fanode.integrate(ts, y0, max_steps=4096, args=jnp.array([]))
    ys_explicit = fanode.integrate(
        ts, y0, max_steps=4096, args=jnp.array([]), dt0=ts[1] - ts[0]
    )

    assert ys_explicit.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys_explicit)))
    assert not jnp.array_equal(ys_default, ys_explicit)


def test_fluxanode_integrate_protocol_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = fanode.integrate_protocol(ts, jnp.array([1.0, 2.0]), protocol, max_steps=4096)
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_integrate_to_steady_state_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    t, y = fanode.integrate_to_steady_state(
        jnp.array([1.0, 2.0]), max_steps=4096, args=jnp.array([])
    )
    assert t.shape == ()
    assert y.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_fluxanode_integrate_protocol_from_steady_state_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = fanode.integrate_protocol_from_steady_state(
        ts, jnp.array([1.0, 2.0]), protocol, max_steps=4096
    )
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_readout_scale_multiplies_readout_fn_output() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([4.0])

    fanode = FluxAnode(
        variable_names=_names(2),
        n_hidden=1,
        flux_names=_names(2, "v"),
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=nv,
        readout_fn=readout_fn,
        readout_names=("r",),
        readout_scale=jnp.array(2.5),
    )
    d = fanode.readouts(0.0, jnp.ones(3), jnp.array([]))
    assert float(d[0]) == pytest.approx(10.0)


def test_fluxanode_from_flux_ode_reuses_dims_nv_and_readout_fn() -> None:
    """Regression: constructing a FluxAnode alongside its mechanistic
    FluxOde (e.g. for residual training on top of the FluxOde's own
    prediction) used to require the caller to separately compute
    n_obs/n_flux and pass the same nv/readout_fn to both by hand.
    """
    flux_ode = _flux_ode()

    fanode = FluxAnode.from_flux_ode(
        flux_ode, n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )

    assert fanode.nv is flux_ode.nv
    assert fanode.readout_fn is flux_ode.readout_fn
    assert fanode.n_obs == 2  # len(flux_ode.variable_names)
    ys = fanode.integrate(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]), max_steps=4096)
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_from_flux_ode_defaults_widths_from_dims() -> None:
    flux_ode = _flux_ode()

    fanode = FluxAnode.from_flux_ode(
        flux_ode, n_hidden=1, key=_KEY, n_args=3, latent_mapper=HardLatentMapper
    )

    # flux_width defaults to n_flux + n_args, markov_width to
    # n_obs + n_hidden + n_args -- verified indirectly via the MLPs
    # actually accepting inputs of those sizes without a matmul shape error.
    dy = fanode(
        0.0, jnp.ones(3), jnp.ones(3)
    )  # latent (n_obs+n_hidden=3), args (n_args=3)
    assert dy.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(dy)))


def test_fluxanode_simulate_time_course_decodes_and_recomputes_fluxes() -> None:
    """Regression: fluxes must be recomputed by re-encoding the decoded
    trajectory through latent_mapper, not by feeding decoded (n_obs-length)
    state straight into flux_nn (which expects the latent n_obs+n_hidden
    state) -- that would either shape-error or silently produce garbage.
    """
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    sim = fanode.simulate_time_course(
        jnp.array([0.0, 1.0]), jnp.ones(2), max_steps=4096
    )

    assert list(sim.variables_df.columns) == ["a", "b"]
    assert sim.variables_df.shape == (2, 2)  # decoded to n_obs, not n_obs+n_hidden
    assert list(sim.fluxes_df.columns) == ["ra", "rb"]
    assert bool(jnp.all(jnp.isfinite(sim.fluxes)))


@pytest.mark.parametrize(
    "latent_mapper", [HardLatentMapper, MixedLatentMapper, TrainableZ0LatentMapper]
)
def test_fluxanode_simulate_fluxes_use_integrated_hidden_state(
    latent_mapper: type,
) -> None:
    """Regression: fluxes were recomputed as ``fluxes(t, encode(decoded y))``,
    which replaces the hidden dims with their t=0 value (or a function of
    ``y``) instead of the hidden state the ODE actually reached.
    """
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=latent_mapper,
        markov_scale=jnp.array([1.0]),
    )  # fmt: skip
    ts = jnp.linspace(0.0, 2.0, 5)
    y0 = jnp.ones(2)
    sim = fanode.simulate_time_course(ts, y0, max_steps=4096, alpha=0.0)

    zs, _ = fanode._solve(
        t0=ts[0], t1=ts[-1], y0=fanode.latent_mapper.encode(y0), args=None,
        saveat_ts=ts, max_steps=4096, rtol=1e-6, atol=1e-6, method=diffrax.Tsit5,
    )  # fmt: skip
    expected = jax.vmap(lambda t, z: fanode.fluxes(t, z, jnp.zeros((0,))))(ts, zs)
    re_encoded = jax.vmap(
        lambda t, y: fanode.fluxes(t, fanode.latent_mapper.encode(y), jnp.zeros((0,)))
    )(ts, sim.ys)

    assert jnp.allclose(sim.fluxes, expected, atol=1e-5)
    assert not jnp.allclose(sim.fluxes[1:], re_encoded[1:], atol=1e-3)


def test_fluxanode_simulate_steady_state_and_protocol_fluxes_use_latent_state() -> None:
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=TrainableZ0LatentMapper
    )
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    sim = fanode.simulate_protocol_time_course(
        ts, jnp.ones(2), jnp.zeros((2, 0)), max_steps=4096, alpha=0.0
    )
    assert sim._latent_ys is not None
    assert sim._latent_ys.shape == (3, 3)
    expected = jax.vmap(lambda t, z: fanode.fluxes(t, z, jnp.zeros((0,))))(
        sim.ts, sim._latent_ys
    )
    assert jnp.allclose(sim.fluxes, expected)

    ss = fanode.simulate_to_steady_state(jnp.ones(2), max_steps=8192, alpha=0.0)
    assert ss._latent_ys is not None
    assert ss._latent_ys.shape == (1, 3)


def test_fluxanode_simulate_protocol_time_course_prepends_t0() -> None:
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    sim = fanode.simulate_protocol_time_course(
        ts, jnp.ones(2), protocol, max_steps=4096
    )

    assert sim.variables_df.shape == (3, 2)
    assert float(sim.ts[0]) == 0.0


def test_fluxanode_simulate_to_steady_state() -> None:
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    sim = fanode.simulate_to_steady_state(jnp.ones(2), max_steps=8192)

    assert sim.variables_df.shape == (1, 2)
    assert sim.fluxes.shape == (1, 2)


# ---------------------------------------------------------------------------
# Ude
# ---------------------------------------------------------------------------


def _ude_parts() -> tuple[Ode, Node]:
    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([9.0])

    ode = Ode(
        rhs=rhs,
        pars=jnp.array([]),
        variable_names=_names(1),
        readout_fn=readout_fn,
        readout_names=("r",),
    )
    return ode, Node(variable_names=_names(1), width=4, depth=1, key=_KEY)


@pytest.mark.parametrize(
    ("op", "combine"),
    [
        ("+", lambda o, n: o + n),
        ("-", lambda o, n: o - n),
        ("*", lambda o, n: o * n),
        ("/", lambda o, n: o / n),
        ("rel", lambda o, n: o * (1 + n)),
        ("exp", lambda o, n: o * jnp.exp(n)),
        ("pow", lambda o, n: o**n),
    ],
)
def test_ude_call_combines_ode_and_nn_via_op(op: str, combine: Callable) -> None:
    ode, nn = _ude_parts()
    ude = Ude(ode=ode, nn=nn, op=op)  # type: ignore[arg-type]
    y, args = jnp.array([2.0]), jnp.array([])
    expected = combine(ode(0.0, y, args), nn(0.0, y, args))
    assert jnp.allclose(ude(0.0, y, args), expected)


def test_ude_rejects_unknown_op() -> None:
    ode, nn = _ude_parts()
    with pytest.raises(ValueError, match="unknown op"):
        Ude(ode=ode, nn=nn, op="adjust")  # type: ignore[arg-type]


def test_ude_rejects_mismatched_variable_names() -> None:
    ode, _ = _ude_parts()
    nn = Node(variable_names=("other",), width=4, depth=1, key=_KEY)
    with pytest.raises(ValueError, match="variable_names"):
        Ude(ode=ode, nn=nn, op="+")


def test_ude_names_and_readouts_delegate_to_ode() -> None:
    ode, nn = _ude_parts()
    ude = Ude(ode=ode, nn=nn, op="+")
    assert ude.variable_names == ode.variable_names
    assert ude.readout_names == ("r",)
    assert float(ude.readouts(0.0, jnp.array([1.0]), jnp.array([]))[0]) == 9.0


# ---------------------------------------------------------------------------
# FluxUde
# ---------------------------------------------------------------------------


def _fluxude_parts() -> tuple[FluxOde, FluxNode]:
    def fluxes(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([y[0], 2 * y[1], y[0] * y[1]])

    def nv(f: jnp.ndarray) -> jnp.ndarray:
        # not shape-preserving: combining dy/dt instead of fluxes would differ
        return jnp.stack([f[0] - f[2], f[1] + f[2]])

    def readout_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([11.0])

    flux_ode = FluxOde(
        flux_fn=fluxes,
        nv=nv,
        pars=jnp.array([]),
        variable_names=("a", "b"),
        flux_names=("v0", "v1", "v2"),
        readout_fn=readout_fn,
        readout_names=("r",),
    )
    return flux_ode, FluxNode.from_flux_ode(flux_ode, width=4, depth=1, key=_KEY)


def test_fluxude_combines_fluxes_before_nv() -> None:
    flux_ode, flux_nn = _fluxude_parts()
    fude = FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="*")
    y, args = jnp.array([2.0, 3.0]), jnp.array([])
    fluxes = flux_ode.fluxes(0.0, y, args) * flux_nn.fluxes(0.0, y, args)
    assert jnp.allclose(fude.fluxes(0.0, y, args), fluxes)
    assert jnp.allclose(fude(0.0, y, args), flux_ode.nv(fluxes))
    assert not jnp.allclose(
        fude(0.0, y, args), flux_ode(0.0, y, args) * flux_nn(0.0, y, args)
    )


def test_fluxude_rejects_mismatched_flux_names() -> None:
    flux_ode, _ = _fluxude_parts()
    flux_nn = FluxNode(
        variable_names=("a", "b"),
        flux_names=("v0", "v1"),
        width=4,
        depth=1,
        key=_KEY,
        nv=flux_ode.nv,
    )
    with pytest.raises(ValueError, match="flux_names"):
        FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="*")


def test_fluxude_names_and_readouts_delegate_to_flux_ode() -> None:
    flux_ode, flux_nn = _fluxude_parts()
    fude = FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="+")
    assert fude.variable_names == ("a", "b")
    assert fude.flux_names == ("v0", "v1", "v2")
    assert fude.readout_names == ("r",)
    assert float(fude.readouts(0.0, jnp.ones(2), jnp.array([]))[0]) == 11.0


def test_fluxude_simulate_reports_combined_fluxes() -> None:
    flux_ode, flux_nn = _fluxude_parts()
    fude = FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="+")
    sim = fude.simulate_time_course(jnp.array([0.0, 0.1]), jnp.ones(2), max_steps=4096)
    assert list(sim.fluxes_df.columns) == ["v0", "v1", "v2"]
    expected = jax.vmap(lambda t, y: fude.fluxes(t, y, jnp.zeros((0,))))(sim.ts, sim.ys)
    assert jnp.allclose(sim.fluxes, expected)
    assert sim.ode_fluxes is not None
    assert sim.nn_fluxes is not None
    assert jnp.allclose(sim.fluxes, sim.ode_fluxes + sim.nn_fluxes)


# ---------------------------------------------------------------------------
# AugUde
# ---------------------------------------------------------------------------


def _augude_parts(
    latent_mapper: LatentMapperFn = HardLatentMapper,
) -> tuple[Ode, Anode]:
    def rhs(time: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return -y

    def readout_fn(time: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return jnp.array([9.0])

    ode = Ode(
        rhs=rhs,
        pars=jnp.array([]),
        variable_names=_names(2),
        readout_fn=readout_fn,
        readout_names=("r",),
    )
    nn = Anode(
        variable_names=_names(2),
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        latent_mapper=latent_mapper,
    )
    return ode, nn


@pytest.mark.parametrize("op", ["+", "*", "rel"])
def test_augude_call_combines_observed_slice_and_passes_hidden(op: str) -> None:
    ode, nn = _augude_parts()
    aude = AugUde(ode=ode, nn=nn, op=op)  # type: ignore[arg-type]
    z, args = jnp.array([2.0, 3.0, 0.5]), jnp.array([])
    d = nn(0.0, z, args)
    dz = aude(0.0, z, args)
    assert dz.shape == (3,)
    assert jnp.allclose(dz[:2], aude.op(ode(0.0, z[:2], args), d[:2]))
    assert jnp.allclose(dz[2:], d[2:])


def test_augude_mechanistic_part_ignores_hidden_dims() -> None:
    ode, nn = _augude_parts()
    aude = AugUde(ode=ode, nn=nn, op="-")
    args = jnp.array([])
    z1, z2 = jnp.array([2.0, 3.0, 0.5]), jnp.array([2.0, 3.0, -4.0])
    # with op "-", dy + nn_obs recovers the mechanistic term alone
    mech1 = aude(0.0, z1, args)[:2] + nn(0.0, z1, args)[:2]
    mech2 = aude(0.0, z2, args)[:2] + nn(0.0, z2, args)[:2]
    assert jnp.allclose(mech1, mech2)


def test_augude_rejects_mismatched_variable_names() -> None:
    ode, _ = _augude_parts()
    nn = Anode(variable_names=("p", "q"), n_hidden=1, width=4, depth=1, key=_KEY)
    with pytest.raises(ValueError, match="variable_names"):
        AugUde(ode=ode, nn=nn, op="+")


def test_augude_rejects_unknown_op() -> None:
    ode, nn = _augude_parts()
    with pytest.raises(ValueError, match="unknown op"):
        AugUde(ode=ode, nn=nn, op="adjust")  # type: ignore[arg-type]


def test_augude_rejects_full_latent_mapper() -> None:
    ode, nn = _augude_parts(FullLatentMapper)
    with pytest.raises(TypeError, match="FullLatentMapper"):
        AugUde(ode=ode, nn=nn, op="+")


@pytest.mark.parametrize(
    "latent_mapper", [HardLatentMapper, SoftLatentMapper, MixedLatentMapper]
)
def test_augude_accepts_observation_preserving_mappers(
    latent_mapper: LatentMapperFn,
) -> None:
    ode, nn = _augude_parts(latent_mapper)
    AugUde(ode=ode, nn=nn, op="+")


def test_augude_integrate_decodes_to_n_obs() -> None:
    ode, nn = _augude_parts()
    aude = AugUde(ode=ode, nn=nn, op="+")
    y0 = jnp.array([1.0, 2.0])
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = aude.integrate(ts, y0, max_steps=4096, args=jnp.array([]))
    assert ys.shape == (5, 2)
    assert jnp.allclose(ys[0], y0)
    protocol = jnp.zeros((2, 0))
    ys_p = aude.integrate_protocol(
        [jnp.array([0.5]), jnp.array([1.0])], y0, protocol, max_steps=4096
    )
    assert ys_p.shape == (2, 2)
    # random hidden dynamics need not settle; t1 caps the solve, only shapes matter
    _, y_ss = aude.integrate_to_steady_state(
        y0, max_steps=4096, args=jnp.array([]), t1=1.0
    )
    assert y_ss.shape == (2,)


def test_augude_simulate_keeps_latent_and_delegates_readouts() -> None:
    ode, nn = _augude_parts()
    aude = AugUde(ode=ode, nn=nn, op="+")
    sim = aude.simulate_time_course(
        jnp.array([0.0, 0.1]), jnp.array([1.0, 2.0]), max_steps=4096
    )
    assert sim.ys.shape == (2, 2)
    assert sim.readout_names == ("r",)
    assert jnp.allclose(sim.readouts, 9.0)


# ---------------------------------------------------------------------------
# FluxAugUde
# ---------------------------------------------------------------------------


def _fluxaugude_parts(
    latent_mapper: LatentMapperFn = HardLatentMapper,
) -> tuple[FluxOde, FluxAnode]:
    flux_ode, _ = _fluxude_parts()
    flux_nn = FluxAnode.from_flux_ode(
        flux_ode, n_hidden=1, key=_KEY, latent_mapper=latent_mapper
    )
    return flux_ode, flux_nn


def test_fluxaugude_combines_fluxes_on_observed_slice_before_nv() -> None:
    flux_ode, flux_nn = _fluxaugude_parts()
    faude = FluxAugUde(flux_ode=flux_ode, flux_nn=flux_nn, op="*")
    z, args = jnp.array([2.0, 3.0, 0.5]), jnp.array([])
    fluxes = flux_ode.fluxes(0.0, z[:2], args) * flux_nn.fluxes(0.0, z, args)
    assert jnp.allclose(faude.fluxes(0.0, z, args), fluxes)
    dz = faude(0.0, z, args)
    assert dz.shape == (3,)
    assert jnp.allclose(dz[:2], flux_ode.nv(fluxes))
    assert jnp.allclose(dz[2:], flux_nn(0.0, z, args)[2:])


def test_fluxaugude_rejects_mismatched_flux_names() -> None:
    flux_ode, _ = _fluxude_parts()
    flux_nn = FluxAnode(
        variable_names=("a", "b"),
        flux_names=("v0", "v1"),
        n_hidden=1,
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=flux_ode.nv,
    )
    with pytest.raises(ValueError, match="flux_names"):
        FluxAugUde(flux_ode=flux_ode, flux_nn=flux_nn, op="+")


def test_fluxaugude_rejects_full_latent_mapper() -> None:
    flux_ode, flux_nn = _fluxaugude_parts(FullLatentMapper)
    with pytest.raises(TypeError, match="FullLatentMapper"):
        FluxAugUde(flux_ode=flux_ode, flux_nn=flux_nn, op="+")


def test_fluxaugude_simulate_reports_flux_parts() -> None:
    flux_ode, flux_nn = _fluxaugude_parts()
    faude = FluxAugUde(flux_ode=flux_ode, flux_nn=flux_nn, op="+")
    sim = faude.simulate_time_course(jnp.array([0.0, 0.1]), jnp.ones(2), max_steps=4096)
    assert sim.ys.shape == (2, 2)
    assert list(sim.fluxes_df.columns) == ["v0", "v1", "v2"]
    assert sim.ode_fluxes is not None
    assert sim.nn_fluxes is not None
    assert jnp.allclose(sim.fluxes, sim.ode_fluxes + sim.nn_fluxes)
    assert jnp.allclose(
        sim.ode_fluxes,
        jax.vmap(lambda t, y: flux_ode.fluxes(t, y, jnp.zeros((0,))))(sim.ts, sim.ys),
    )
    assert float(sim.readouts[0, 0]) == 11.0


# ---------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------


def test_non_flux_models_simulate_to_simulation_with_readouts() -> None:
    def rhs(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return -y

    def readout_fn(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([y[0] + y[1]])

    ode = Ode(
        rhs=rhs,
        pars=jnp.array([]),
        variable_names=("a", "b"),
        readout_fn=readout_fn,
        readout_names=("total",),
    )
    ts = jnp.array([0.0, 1.0])
    sim = ode.simulate_time_course(ts, jnp.array([1.0, 2.0]), max_steps=4096)

    assert type(sim) is Simulation
    assert not hasattr(sim, "fluxes")
    assert bool(sim.success)
    assert list(sim.variables_df.columns) == ["a", "b"]
    assert list(sim.readouts_df.columns) == ["total"]
    np.testing.assert_allclose(sim.readouts[:, 0], sim.ys.sum(axis=1), rtol=1e-6)
    assert list(sim.get_combined().columns) == ["a", "b", "total"]


def test_simulate_raises_on_solver_failure_by_default() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    with pytest.raises(eqx.EquinoxRuntimeError):
        fnode.simulate_time_course(jnp.array([0.0, 10.0]), jnp.ones(2), max_steps=1)


def test_simulate_throw_false_reports_failure() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    sim = fnode.simulate_time_course(
        jnp.array([0.0, 10.0]), jnp.ones(2), max_steps=1, throw=False
    )
    assert not bool(sim.success)
    assert bool(jnp.all(jnp.isfinite(sim.ys)))  # zero-filled, not NaN
    assert sim.variables_df.isna().all().all()
    assert sim.fluxes_df.isna().all().all()


def test_steady_state_not_reached_is_reported_not_raised() -> None:
    # dy/dt = y grows forever: no steady state before t1
    fode = FluxOde(
        flux_fn=lambda _t, y, _args: y,
        nv=lambda f: f,
        pars=jnp.array([]),
        variable_names=("a",),
        flux_names=("v",),
    )
    sim = fode.simulate_to_steady_state(jnp.ones(1), max_steps=4096, t1=1.0)
    assert not bool(sim.success)
    assert sim.variables_df.isna().all().all()


def test_simulation_fluxes_are_differentiable() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    ts = jnp.array([0.0, 0.5, 1.0])

    @eqx.filter_grad
    def grad(model: FluxNode) -> jax.Array:
        sim = model.simulate_time_course(ts, jnp.ones(2), max_steps=4096)
        return jnp.sum(sim.fluxes**2)

    g = grad(fnode)
    assert float(jnp.max(jnp.abs(g.flux_nn.layers[0].weight))) > 0


def test_batched_simulation_reports_success_per_member() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    members = [fnode, eqx.tree_at(lambda m: m.flux_scale, fnode, jnp.array([0.5]))]
    batched = batch_simulate(
        members,
        FluxNode.simulate_time_course,
        jnp.array([0.0, 1.0]),
        jnp.ones(2),
        max_steps=4096,
    )
    assert batched.fluxes.shape == (2, 2, 2)
    assert batched.success.shape == (2,)
    assert bool(jnp.all(batched.success))
    unstacked = unstack_models(batched, 2)
    assert list(unstacked[1].fluxes_df.columns) == ["ra", "rb"]
    np.testing.assert_allclose(unstacked[0].fluxes, batched.fluxes[0])
