"""Tests for mxlpy.jax.ensemble: stack_models and batch_simulate."""

import jax.numpy as jnp

from mxlpy.jax import ensemble as jax_ensemble
from mxlpy.jax.models import Ode


def _rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
    return -args[-1] * y


def test_stack_models_adds_leading_axis_to_array_leaves() -> None:
    models = [Ode(rhs=_rhs, pars=jnp.array([float(k)])) for k in range(3)]
    stacked = jax_ensemble.stack_models(models)

    assert stacked.pars.shape == (3, 1)
    assert jnp.allclose(stacked.pars[:, 0], jnp.array([0.0, 1.0, 2.0]))
    # non-array (static) leaves come from the first model
    assert stacked.rhs is models[0].rhs


def test_batch_simulate_runs_every_member_in_one_call() -> None:
    models = [Ode(rhs=_rhs, pars=jnp.array([float(k) + 1.0])) for k in range(3)]
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0])

    def _simulate(model: Ode, ts: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        return model.integrate(ts, y0, 8192, args=jnp.array([]))

    out = jax_ensemble.batch_simulate(models, _simulate, ts, y0)

    assert out.shape == (3, 2, 1)
    # dv/dt = -k*v with k = pars[0]; each member decays at its own rate, so
    # this must match running each model individually, not just be finite.
    expected = jnp.stack(
        [m.integrate(ts, y0, 8192, args=jnp.array([])) for m in models], axis=0
    )
    assert jnp.allclose(out, expected, atol=1e-4)
