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
    Node,
    Ode,
    Ude,
)

_KEY = jax.random.PRNGKey(0)

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


def test_fluxanode_integrate_decodes_to_n_obs() -> None:
    fanode = _fluxanode()
    ts = jnp.array([0.0, 1.0])
    ys = fanode.integrate(ts, jnp.array([1.0, 2.0]), 4096, args=jnp.array([]))
    assert ys.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(ys)))


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
