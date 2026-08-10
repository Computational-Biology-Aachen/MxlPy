"""Tests for the neural JAX model classes: Node, FluxNode, Anode, FluxAnode, Ude, FluxUde.

Until now these classes had zero test coverage, which is exactly how a real
``AttributeError`` in ``FluxUde.__call__`` (it referenced a field that didn't
exist) and a shape bug in ``FluxNode`` (no ``n_args`` slot despite accepting
``args``) went unnoticed.
"""

import jax
import jax.numpy as jnp
import pytest

from mxlpy.jax.models import (
    Anode,
    FluxAnode,
    FluxNode,
    FluxOde,
    FluxUde,
    HardLatentMapper,
    MixedLatentMapper,
    Node,
    Ode,
    TrainableZ0LatentMapper,
    Ude,
)

_KEY = jax.random.PRNGKey(0)


def _flux_ode(*, with_names: bool = True) -> FluxOde:
    # n_obs == n_flux == 2 throughout, so nv (flux_vector -> dy/dt) can be a
    # simple shape-preserving map -- self-consistent enough to actually
    # integrate, not just construct.
    def fluxes(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([y[0], y[1]])

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([7.0])

    return FluxOde(
        fluxes=fluxes,
        nv=nv,
        pars=jnp.array([]),
        variable_names=("a", "b") if with_names else None,
        flux_names=("ra", "rb") if with_names else None,
        derived_fn=derived_fn,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def test_node_forward_pass_shape() -> None:
    node = Node(n_obs=3, width=4, depth=1, key=_KEY)
    dy = node(0.0, jnp.ones(3), jnp.array([]))
    assert dy.shape == (3,)


def test_node_derived_scale_multiplies_derived_fn_output() -> None:
    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([7.0])

    node = Node(
        n_obs=2,
        width=4,
        depth=1,
        key=_KEY,
        derived_fn=derived_fn,
        derived_scale=jnp.array(2.0),
    )
    d = node.derived(0.0, jnp.ones(2), jnp.array([]))
    assert float(d[0]) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# FluxNode
# ---------------------------------------------------------------------------


def test_fluxnode_forward_pass_shape() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector[:1] - flux_vector[1:]

    fnode = FluxNode(n_obs=2, n_flux=2, width=4, depth=1, key=_KEY, nv=nv)
    dy = fnode(0.0, jnp.ones(2), jnp.array([]))
    assert dy.shape == (1,)


def test_fluxnode_with_nonzero_n_args_does_not_raise() -> None:
    """Regression: flux_nn's in_size previously ignored n_args entirely, so
    any call with non-empty args raised a matmul shape error inside
    eqx.nn.MLP instead of running.
    """

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fnode = FluxNode(n_obs=2, n_flux=2, width=4, depth=1, key=_KEY, nv=nv, n_args=1)
    dy = fnode(0.0, jnp.ones(2), jnp.array([0.5]))
    assert dy.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(dy)))


def test_fluxnode_derived_scale_multiplies_derived_fn_output() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([5.0])

    fnode = FluxNode(
        n_obs=2,
        n_flux=2,
        width=4,
        depth=1,
        key=_KEY,
        nv=nv,
        derived_fn=derived_fn,
        derived_scale=jnp.array(3.0),
    )
    d = fnode.derived(0.0, jnp.ones(2), jnp.array([]))
    assert float(d[0]) == pytest.approx(15.0)


def test_fluxnode_from_flux_ode_reuses_dims_nv_and_derived_fn() -> None:
    """Regression: constructing a FluxNode alongside its mechanistic
    FluxOde used to require the caller to separately compute n_obs/n_flux
    (e.g. len(model.get_variable_names())) and pass the same nv/derived_fn
    to both by hand.
    """
    flux_ode = _flux_ode()

    fnode = FluxNode.from_flux_ode(flux_ode, width=4, depth=1, key=_KEY)

    assert fnode.nv is flux_ode.nv
    assert fnode.derived_fn is flux_ode.derived_fn
    dy = fnode(0.0, jnp.ones(2), jnp.array([]))
    assert dy.shape == (2,)  # n_obs, derived from flux_ode
    assert bool(jnp.all(jnp.isfinite(dy)))


def test_fluxnode_from_flux_ode_requires_names() -> None:
    flux_ode = _flux_ode(with_names=False)

    with pytest.raises(ValueError, match="variable_names/flux_names"):
        FluxNode.from_flux_ode(flux_ode, width=4, depth=1, key=_KEY)


def test_fluxnode_simulate_time_course() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    sim = fnode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.ones(2), 4096)

    assert list(sim.variables.columns) == ["a", "b"]
    assert sim.variables.shape == (2, 2)
    assert list(sim.fluxes.columns) == ["ra", "rb"]
    assert sim.fluxes.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(sim.ys)))


def test_fluxnode_simulate_protocol_time_course_prepends_t0() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    sim = fnode.simulate_protocol_time_course(ts, jnp.ones(2), protocol, 4096)

    assert sim.variables.shape == (3, 2)  # t=0 row prepended
    assert float(sim.ts[0]) == 0.0


def test_fluxnode_simulate_to_steady_state() -> None:
    fnode = FluxNode.from_flux_ode(_flux_ode(), width=4, depth=1, key=_KEY)
    sim = fnode.simulate_to_steady_state(jnp.ones(2), 8192)

    assert sim.variables.shape == (1, 2)
    assert sim.fluxes.shape == (1, 2)


def test_fluxnode_simulate_requires_names() -> None:
    fnode = FluxNode(n_obs=2, n_flux=2, width=4, depth=1, key=_KEY, nv=lambda f: f)

    with pytest.raises(ValueError, match="variable_names/flux_names"):
        fnode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.ones(2), 4096)


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
        n_obs=2,
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
    ys = anode.integrate(ts, jnp.array([1.0, 2.0]), 4096, args=jnp.array([]))
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_anode_integrate_protocol_decodes_to_n_obs() -> None:
    """Regression: integrate_protocol used to be inherited unmodified from
    Base, feeding the raw n_obs-length y0 straight into a latent-space nn
    that expects n_obs+n_hidden(+n_args) inputs.
    """
    anode = _anode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = anode.integrate_protocol(ts, jnp.array([1.0, 2.0]), protocol, 4096)
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_anode_integrate_to_steady_state_decodes_to_n_obs() -> None:
    anode = _anode()
    t, y = anode.integrate_to_steady_state(
        jnp.array([1.0, 2.0]), 4096, args=jnp.array([])
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
        ts, jnp.array([1.0, 2.0]), protocol, 4096
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
    ys = anode.integrate(ts, jnp.array([1.0, 2.0]), 4096, args=jnp.array([]))
    assert ys.shape == (2, 2)


def test_anode_integrate_threads_alpha_to_blending_latent_mapper() -> None:
    anode = Anode(
        n_obs=2,
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        latent_mapper=MixedLatentMapper,
    )
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])
    ys_hard = anode.integrate(ts, y0, 4096, args=jnp.array([]), alpha=0.0)
    ys_learned = anode.integrate(ts, y0, 4096, args=jnp.array([]), alpha=1.0)
    assert not jnp.allclose(ys_hard, ys_learned)
    with pytest.raises(ValueError, match="alpha is required"):
        anode.integrate(ts, y0, 4096, args=jnp.array([]))


def test_anode_derived_scale_multiplies_derived_fn_output() -> None:
    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([3.0])

    anode = Anode(
        n_obs=2,
        n_hidden=1,
        width=4,
        depth=1,
        key=_KEY,
        derived_fn=derived_fn,
        derived_scale=jnp.array(2.0),
    )
    d = anode.derived(0.0, jnp.ones(3), jnp.array([]))
    assert float(d[0]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# FluxAnode
# ---------------------------------------------------------------------------


def _fluxanode() -> FluxAnode:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    return FluxAnode(
        n_obs=2,
        n_hidden=1,
        n_flux=2,
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
        n_obs=2,
        n_hidden=1,
        n_flux=2,
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
        n_obs=2,
        n_hidden=1,
        n_flux=2,
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
    ys_hard = fanode.integrate(ts, y0, 4096, args=jnp.array([]), alpha=0.0)
    ys_learned = fanode.integrate(ts, y0, 4096, args=jnp.array([]), alpha=1.0)
    assert not jnp.allclose(ys_hard, ys_learned)
    with pytest.raises(ValueError, match="alpha is required"):
        fanode.integrate(ts, y0, 4096, args=jnp.array([]))


def test_fluxanode_integrate_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    ys = fanode.integrate(ts, jnp.array([1.0, 2.0]), 4096, args=jnp.array([]))
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_integrate_dt0_defaults_to_none_and_is_unaffected() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0, 2.0])

    ys_omitted = fanode.integrate(ts, y0, 4096, args=jnp.array([]))
    ys_explicit_none = fanode.integrate(ts, y0, 4096, args=jnp.array([]), dt0=None)

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

    ys_default = fanode.integrate(ts, y0, 4096, args=jnp.array([]))
    ys_explicit = fanode.integrate(ts, y0, 4096, args=jnp.array([]), dt0=ts[1] - ts[0])

    assert ys_explicit.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys_explicit)))
    assert not jnp.array_equal(ys_default, ys_explicit)


def test_fluxanode_integrate_protocol_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = fanode.integrate_protocol(ts, jnp.array([1.0, 2.0]), protocol, 4096)
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_integrate_to_steady_state_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    t, y = fanode.integrate_to_steady_state(
        jnp.array([1.0, 2.0]), 4096, args=jnp.array([])
    )
    assert t.shape == ()
    assert y.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_fluxanode_integrate_protocol_from_steady_state_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    ys = fanode.integrate_protocol_from_steady_state(
        ts, jnp.array([1.0, 2.0]), protocol, 4096
    )
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


def test_fluxanode_derived_scale_multiplies_derived_fn_output() -> None:
    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([4.0])

    fanode = FluxAnode(
        n_obs=2,
        n_hidden=1,
        n_flux=2,
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=nv,
        derived_fn=derived_fn,
        derived_scale=jnp.array(2.5),
    )
    d = fanode.derived(0.0, jnp.ones(3), jnp.array([]))
    assert float(d[0]) == pytest.approx(10.0)


def test_fluxanode_from_flux_ode_reuses_dims_nv_and_derived_fn() -> None:
    """Regression: constructing a FluxAnode alongside its mechanistic
    FluxOde (e.g. for residual training on top of the FluxOde's own
    prediction) used to require the caller to separately compute
    n_obs/n_flux and pass the same nv/derived_fn to both by hand.
    """
    flux_ode = _flux_ode()

    fanode = FluxAnode.from_flux_ode(
        flux_ode, n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )

    assert fanode.nv is flux_ode.nv
    assert fanode.derived_fn is flux_ode.derived_fn
    assert fanode.n_obs == 2  # len(flux_ode.variable_names)
    ys = fanode.integrate(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]), 4096)
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


def test_fluxanode_from_flux_ode_requires_names() -> None:
    flux_ode = _flux_ode(with_names=False)

    with pytest.raises(ValueError, match="variable_names/flux_names"):
        FluxAnode.from_flux_ode(flux_ode, n_hidden=1, key=_KEY)


def test_fluxanode_simulate_time_course_decodes_and_recomputes_fluxes() -> None:
    """Regression: fluxes must be recomputed by re-encoding the decoded
    trajectory through latent_mapper, not by feeding decoded (n_obs-length)
    state straight into flux_nn (which expects the latent n_obs+n_hidden
    state) -- that would either shape-error or silently produce garbage.
    """
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    sim = fanode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.ones(2), 4096)

    assert list(sim.variables.columns) == ["a", "b"]
    assert sim.variables.shape == (2, 2)  # decoded to n_obs, not n_obs+n_hidden
    assert list(sim.fluxes.columns) == ["ra", "rb"]
    assert bool(jnp.all(jnp.isfinite(sim.fluxes.to_numpy())))


def test_fluxanode_simulate_protocol_time_course_prepends_t0() -> None:
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    sim = fanode.simulate_protocol_time_course(ts, jnp.ones(2), protocol, 4096)

    assert sim.variables.shape == (3, 2)
    assert float(sim.ts[0]) == 0.0


def test_fluxanode_simulate_to_steady_state() -> None:
    fanode = FluxAnode.from_flux_ode(
        _flux_ode(), n_hidden=1, key=_KEY, latent_mapper=HardLatentMapper
    )
    sim = fanode.simulate_to_steady_state(jnp.ones(2), 8192)

    assert sim.variables.shape == (1, 2)
    assert sim.fluxes.shape == (1, 2)


def test_fluxanode_simulate_requires_names() -> None:
    fanode = FluxAnode(
        n_obs=2,
        n_hidden=1,
        n_flux=2,
        flux_width=4,
        flux_depth=1,
        markov_width=4,
        markov_depth=1,
        key=_KEY,
        nv=lambda f: f,
        latent_mapper=HardLatentMapper,
    )

    with pytest.raises(ValueError, match="variable_names/flux_names"):
        fanode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.ones(2), 4096)


# ---------------------------------------------------------------------------
# Ude
# ---------------------------------------------------------------------------


def test_ude_call_combines_ode_and_nn_via_op() -> None:
    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    ode = Ode(rhs=rhs, pars=jnp.array([]))
    nn = Node(n_obs=1, width=4, depth=1, key=_KEY)

    ude = Ude(ode=ode, nn=nn, op="+")
    y, args = jnp.array([2.0]), jnp.array([])
    expected = ode(0.0, y, args) + nn(0.0, y, args)
    assert jnp.allclose(ude(0.0, y, args), expected)


def test_ude_derived_delegates_to_ode() -> None:
    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([9.0])

    ode = Ode(rhs=rhs, pars=jnp.array([]), derived_fn=derived_fn)
    nn = Node(n_obs=1, width=4, depth=1, key=_KEY)
    ude = Ude(ode=ode, nn=nn, op="+")
    assert float(ude.derived(0.0, jnp.array([1.0]), jnp.array([]))[0]) == 9.0


# ---------------------------------------------------------------------------
# FluxUde
# ---------------------------------------------------------------------------


def test_fluxude_call_combines_flux_ode_and_flux_nn_via_op() -> None:
    """Regression: __call__ referenced ``self.flux_nn`` before that field
    existed on the class (it was named ``flux_node``), so calling any
    FluxUde raised an AttributeError.
    """

    def fluxes(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    flux_ode = FluxOde(fluxes=fluxes, nv=nv, pars=jnp.array([]))
    flux_nn = FluxNode(n_obs=1, n_flux=1, width=4, depth=1, key=_KEY, nv=nv)

    fude = FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="*")
    y, args = jnp.array([2.0]), jnp.array([])
    expected = flux_ode(0.0, y, args) * flux_nn(0.0, y, args)
    assert jnp.allclose(fude(0.0, y, args), expected)


def test_fluxude_derived_delegates_to_flux_ode() -> None:
    def fluxes(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    def derived_fn(_t: float, _y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([11.0])

    flux_ode = FluxOde(fluxes=fluxes, nv=nv, pars=jnp.array([]), derived_fn=derived_fn)
    flux_nn = FluxNode(n_obs=1, n_flux=1, width=4, depth=1, key=_KEY, nv=nv)
    fude = FluxUde(flux_ode=flux_ode, flux_nn=flux_nn, op="*")
    assert float(fude.derived(0.0, jnp.array([1.0]), jnp.array([]))[0]) == 11.0
