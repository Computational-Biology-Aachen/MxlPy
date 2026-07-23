"""Tests for mxlpy.jax.ensemble: stack_models, batch_simulate, unstack_models.

Also covers two regressions in batch_simulate found in review:

- it used to define and jit/vmap-wrap a fresh closure on every call, with
  `*args`/`**kwargs` captured from that closure (baked in as trace-time
  constants) rather than passed as traced arguments -- so it never hit
  JAX's jit cache across repeated calls, even with identical shapes.
- passing a `simulate` function that returns a plain (unregistered)
  dataclass -- e.g. FluxOde.simulate_time_course, which returns a
  FluxOdeSimulation -- silently produced leaked tracers instead of a
  correctly batched result, since eqx.filter_vmap treated the whole
  dataclass as one opaque, non-batched leaf.
"""

import jax.numpy as jnp

from mxlpy.jax import ensemble as jax_ensemble
from mxlpy.jax.models import FluxOde, Ode


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


def test_batch_simulate_reuses_jit_cache_across_calls() -> None:
    """Regression: batch_simulate used to define+jit a fresh closure every
    call (with extra args captured from that closure, not passed as traced
    arguments), so it recompiled from scratch on every single call even
    with identical shapes.
    """
    models = [Ode(rhs=_rhs, pars=jnp.array([float(k) + 1.0])) for k in range(3)]
    ts = jnp.array([0.0, 1.0])
    y0 = jnp.array([1.0])
    calls = {"n": 0}

    def _simulate(model: Ode, ts: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        calls["n"] += 1
        return model.integrate(ts, y0, 8192, args=jnp.array([]))

    jax_ensemble.batch_simulate(models, _simulate, ts, y0)
    jax_ensemble.batch_simulate(models, _simulate, ts, y0)
    # Same shapes, different underlying data -- a genuine jit cache hit
    # reuses the compiled program regardless of the concrete values.
    jax_ensemble.batch_simulate(models, _simulate, ts + 1.0, y0)

    assert calls["n"] == 1


def test_batch_simulate_composes_with_fluxode_simulate_time_course() -> None:
    """Regression: FluxOde.simulate_time_course returns a FluxOdeSimulation;
    passing it to batch_simulate used to silently leak tracers (the plain
    dataclass wasn't a registered pytree, so eqx.filter_vmap couldn't batch
    through it) instead of raising or producing a usable result.
    """

    def fluxes(_t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        return -args[-1] * y

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    models = [
        FluxOde(
            fluxes=fluxes,
            nv=nv,
            pars=jnp.array([float(k) + 1.0]),
            variable_names=("x",),
            flux_names=("v",),
        )
        for k in range(3)
    ]
    ts = jnp.linspace(0.0, 1.0, 4)
    y0 = jnp.array([1.0])

    batched = jax_ensemble.batch_simulate(
        models, FluxOde.simulate_time_course, ts, y0, 8192
    )
    assert batched.ys.shape == (3, 4, 1)
    assert not jnp.allclose(batched.ys[0], batched.ys[1])  # distinct pars, distinct traj

    # No leaked tracers: unstacking recovers a real, usable per-member result.
    members = jax_ensemble.unstack_models(batched, n=3)
    assert len(members) == 3
    for member, model in zip(members, models, strict=True):
        expected = model.simulate_time_course(ts, y0, 8192)
        assert jnp.allclose(member.ys, expected.ys)
        assert jnp.allclose(member.variables.to_numpy(), expected.variables.to_numpy())


# ---------------------------------------------------------------------------
# unstack_models
# ---------------------------------------------------------------------------


def test_unstack_models_round_trips_stack_models() -> None:
    models = [Ode(rhs=_rhs, pars=jnp.array([float(k)])) for k in range(3)]
    stacked = jax_ensemble.stack_models(models)

    members = jax_ensemble.unstack_models(stacked, n=3)

    assert len(members) == 3
    for member, model in zip(members, models, strict=True):
        assert jnp.allclose(member.pars, model.pars)
        assert member.rhs is model.rhs
