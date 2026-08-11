"""Tests for mxlpy.jax.train: squeeze_derived, perturb_model, and train()'s
solver-retry / KeyboardInterrupt / grad-norm-history behaviour.

Also covers regressions for four bugs found in a review of this module:

- train_protocol's early-stopping path used to return the current step's
  model instead of the best model seen so far.
- proto_make_step's gradient "clipping" used to unconditionally rescale
  gradients to unit norm instead of only scaling down when the norm
  exceeded the clip threshold.
- train_only_nde's curriculum loop used ``enumerate(training_steps)``
  (0-based) while checking ``i == len(training_steps)`` (only reachable
  1-based), making its best-model tracking and early stopping dead code;
  it also had no grad-norm history and no solver-error recovery, unlike
  train()/train_protocol().
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import pytest

from mxlpy.jax import train as jax_train
from mxlpy.jax.models import Node, Ode, Ude

_KEY = jax.random.PRNGKey(0)


def _decay_rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
    return -args[-1] * y


def _decay_model() -> Ode:
    return Ode(rhs=_decay_rhs, pars=jnp.array([0.5]))


def _ude_model() -> Ude:
    return Ude(ode=_decay_model(), nn=Node(n_obs=1, width=4, depth=1, key=_KEY), op="+")


def _tiny_training_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.exp(-0.3 * ts)[:, None]
    return ts, ys


def _multi_quantity_training_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    # Two observed quantities on very different scales, so per-quantity
    # (axis=0) and flattened (axis=None) mean/std differ measurably.
    ts = jnp.linspace(0.0, 1.0, 5)
    a = jnp.exp(-0.3 * ts)
    b = 100.0 * jnp.exp(-0.1 * ts)
    ys = jnp.stack([a, b], axis=1)
    return ts, ys


def _array_leaves(model: eqx.Module) -> list[jnp.ndarray]:
    return jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))


def _allclose_models(a: eqx.Module, b: eqx.Module) -> bool:
    a_leaves, b_leaves = _array_leaves(a), _array_leaves(b)
    return len(a_leaves) == len(b_leaves) and all(
        jnp.allclose(x, y) for x, y in zip(a_leaves, b_leaves, strict=True)
    )


# ---------------------------------------------------------------------------
# squeeze_derived
# ---------------------------------------------------------------------------


def test_squeeze_derived_squeezes_single_quantity() -> None:
    derived = jnp.ones((4, 1))
    out = jax_train.squeeze_derived(derived)
    assert out.shape == (4,)


def test_squeeze_derived_leaves_multi_quantity_unchanged() -> None:
    derived = jnp.ones((4, 3))
    out = jax_train.squeeze_derived(derived)
    assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# derived_simulation_fn: train_protocol's simulation_fn for fitting a
# derived observable instead of raw state
# ---------------------------------------------------------------------------


def test_derived_simulation_fn_maps_states_through_derived() -> None:
    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ctx = jax_train.IntegrationSettings()

    states = model.integrate_protocol(
        ts=ts,
        y0=y0,
        protocol=protocol,
        max_steps=ctx.max_steps,
        rtol=ctx.rtol,
        atol=ctx.atol,
        method=ctx.method,
    )
    derived = jax_train.derived_simulation_fn(model, ts, y0, protocol, ctx)

    # keeps the trailing quantity axis, matching ys's own (T, n_obs) shape
    # convention (see train_protocol's proto_grad_loss concatenation with
    # ys[0][None], which requires this)
    assert derived.shape == (2, 1)
    assert jnp.allclose(derived, states * 2.0)


def test_derived_simulation_fn_pairs_each_row_with_its_active_protocol_arg() -> None:
    """Every saved row must see the protocol value active during its own
    window (held constant for that window's duration), not e.g. always
    the first or last step's value.
    """

    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,
    ) -> jnp.ndarray:
        return args[0] * jnp.ones_like(y)

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.25, 0.5]), jnp.array([0.75, 1.0])]
    protocol = jnp.array([[1.0], [2.0]])
    ctx = jax_train.IntegrationSettings()

    derived = jax_train.derived_simulation_fn(model, ts, y0, protocol, ctx)

    assert jnp.allclose(derived, jnp.array([[1.0], [1.0], [2.0], [2.0]]))


def test_derived_simulation_fn_pairs_ragged_steps_with_their_active_protocol_arg() -> (
    None
):
    """Same as the uniform-length case above, but with differing per-step
    save-point counts -- the exact shape _integrate_protocol_raw's own
    padding logic exists to handle, and the shape that broke the two
    jnp.repeat-based implementations tried before this one.
    """

    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,
    ) -> jnp.ndarray:
        return args[0] * jnp.ones_like(y)

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.2, 0.4]), jnp.array([0.6]), jnp.array([0.8, 0.9, 1.0])]
    protocol = jnp.array([[10.0], [20.0], [30.0]])
    ctx = jax_train.IntegrationSettings()

    derived = jax_train.derived_simulation_fn(model, ts, y0, protocol, ctx)

    assert jnp.allclose(
        derived,
        jnp.array([[10.0], [10.0], [20.0], [30.0], [30.0], [30.0]]),
    )


def test_train_protocol_freeze_works_with_derived_simulation_fn() -> None:
    """freeze routes through proto_grad_loss_split/proto_make_step_split
    rather than proto_grad_loss/proto_make_step -- confirm
    derived_simulation_fn's un-squeezed output is shape-compatible with
    that path too, and that the frozen partition stays untouched.
    """

    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    ode = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    model = Ude(ode=ode, nn=Node(n_obs=1, width=4, depth=1, key=_KEY), op="+")
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = 2.0 * jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, _, grad_norms = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        simulation_fn=jax_train.derived_simulation_fn,
        freeze=lambda m: m.ode,
    )

    assert jnp.array_equal(trained.ode.pars, model.ode.pars)
    assert len(grad_norms[0]) == 3
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


def test_train_protocol_with_derived_simulation_fn_trains_end_to_end() -> None:
    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = 2.0 * jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, _, grad_norms = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        simulation_fn=jax_train.derived_simulation_fn,
    )

    assert isinstance(trained, Ode)
    assert len(grad_norms[0]) == 3
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


def test_derived_simulation_fn_without_t0_args_is_unchanged() -> None:
    """t0_args=None (the default) must reproduce the exact old shape/values --
    every existing caller relies on this.
    """

    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ctx = jax_train.IntegrationSettings()

    without_kw = jax_train.derived_simulation_fn(model, ts, y0, protocol, ctx)
    with_none = jax_train.derived_simulation_fn(
        model, ts, y0, protocol, ctx, t0_args=None
    )

    assert without_kw.shape == (2, 1)
    assert jnp.array_equal(without_kw, with_none)


def test_derived_simulation_fn_t0_args_prepends_genuine_t0_row() -> None:
    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray,
    ) -> jnp.ndarray:
        # depends only on the external arg (args[0]; Ode.derived appends
        # trainable pars after it), so the t=0 row is trivially
        # distinguishable from a state-derived value
        return args[0] * jnp.ones((1,))

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.array([[2.0], [3.0]])
    ctx = jax_train.IntegrationSettings()

    derived = jax_train.derived_simulation_fn(
        model, ts, y0, protocol, ctx, t0_args=jnp.array([9.0])
    )

    assert derived.shape == (3, 1)
    assert jnp.allclose(derived, jnp.array([[9.0], [2.0], [3.0]]))


def test_proto_grad_loss_t0_uses_full_prediction_no_ys0_substitution() -> None:
    """Unlike proto_grad_loss, which free-passes ys[0] as a perfect t=0
    prediction, proto_grad_loss_t0 must let a wrong t=0 prediction
    contribute to the loss.
    """

    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray,
    ) -> jnp.ndarray:
        return args[0] * jnp.ones((1,))

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.array([[2.0], [3.0]])
    ctx = jax_train.IntegrationSettings()
    t0_args = jnp.array([9.0])
    # deliberately wrong t=0 target, to confirm it isn't ignored
    ys = jnp.array([[0.0], [2.0], [3.0]])

    simulation_fn = partial(jax_train.derived_simulation_fn, t0_args=t0_args)
    loss, _ = jax_train.proto_grad_loss_t0(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        y_mean=jnp.zeros(1),
        y_scale=jnp.ones(1),
        ctx=ctx,
        simulation_fn=simulation_fn,
        global_step=jnp.asarray(0),
        total_steps=jnp.asarray(1),
    )

    # prediction is [9, 2, 3], target is [0, 2, 3] -> only the t=0 row
    # contributes: mean((9-0)^2, 0, 0) = 81/3 = 27
    assert float(loss) == pytest.approx(27.0)


def test_train_protocol_with_derived_simulation_fn_t0_trains_end_to_end() -> None:
    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = 2.0 * jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, _, grad_norms = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        simulation_fn=partial(jax_train.derived_simulation_fn, t0_args=jnp.zeros((0,))),
        grad_fn=jax_train.proto_grad_loss_t0,
    )

    assert isinstance(trained, Ode)
    assert len(grad_norms[0]) == 3
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


def test_train_protocol_freeze_works_with_derived_simulation_fn_t0() -> None:
    def derived_fn(
        t: jnp.ndarray,  # noqa: ARG001
        y: jnp.ndarray,
        args: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return y * 2.0

    ode = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]), derived_fn=derived_fn)
    model = Ude(ode=ode, nn=Node(n_obs=1, width=4, depth=1, key=_KEY), op="+")
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = 2.0 * jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, _, grad_norms = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        simulation_fn=partial(jax_train.derived_simulation_fn, t0_args=jnp.zeros((0,))),
        grad_fn=jax_train.proto_grad_loss_t0,
        split_grad_fn=jax_train.proto_grad_loss_split_t0,
        freeze=lambda m: m.ode,
    )

    assert jnp.array_equal(trained.ode.pars, model.ode.pars)
    assert len(grad_norms[0]) == 3
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


# ---------------------------------------------------------------------------
# sigmoid_schedule
# ---------------------------------------------------------------------------


def test_sigmoid_schedule_is_close_to_one_early_in_training() -> None:
    alpha = jax_train.sigmoid_schedule(
        jnp.asarray(0), jnp.asarray(1000), steepness=10.0, midpoint=0.5
    )
    assert float(alpha) == pytest.approx(1.0, abs=1e-2)


def test_sigmoid_schedule_is_close_to_zero_late_in_training() -> None:
    alpha = jax_train.sigmoid_schedule(
        jnp.asarray(999), jnp.asarray(1000), steepness=10.0, midpoint=0.5
    )
    assert float(alpha) == pytest.approx(0.0, abs=1e-2)


def test_sigmoid_schedule_is_one_half_at_the_midpoint() -> None:
    alpha = jax_train.sigmoid_schedule(
        jnp.asarray(500), jnp.asarray(1000), steepness=10.0, midpoint=0.5
    )
    assert float(alpha) == pytest.approx(0.5, abs=1e-6)


def test_sigmoid_schedule_decreases_monotonically() -> None:
    steps = jnp.arange(0, 1001, 100)
    alphas = [
        float(
            jax_train.sigmoid_schedule(
                s, jnp.asarray(1000), steepness=5.0, midpoint=0.5
            )
        )
        for s in steps
    ]
    assert alphas == sorted(alphas, reverse=True)


# ---------------------------------------------------------------------------
# perturb_model
# ---------------------------------------------------------------------------


def test_perturb_model_changes_trainable_leaves() -> None:
    model = _decay_model()
    perturbed = jax_train.perturb_model(model, _KEY, scale=0.1)
    assert not jnp.allclose(model.pars, perturbed.pars)


def test_perturb_model_leaves_zero_scale_unchanged() -> None:
    model = _decay_model()
    perturbed = jax_train.perturb_model(model, _KEY, scale=0.0)
    assert jnp.allclose(model.pars, perturbed.pars)


# ---------------------------------------------------------------------------
# train(): return shape, grad-norm history
# ---------------------------------------------------------------------------


def test_train_returns_losses_and_grad_norms_per_lesson() -> None:
    # avg_every=100 (>> steps) shows the intended density asymmetry: losses
    # only logs at step 0 (step % avg_every == 0) and the last step, while
    # grad_norms records unconditionally on every step.
    model = _decay_model()
    ts, ys = _tiny_training_data()

    trained, losses, grad_norms = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=100,
        target_loss=-1.0,  # never trip early stopping
    )

    assert isinstance(trained, Ode)
    assert len(losses) == 1
    assert len(grad_norms) == 1
    assert isinstance(losses[0], pd.Series)
    assert isinstance(grad_norms[0], pd.Series)
    assert len(grad_norms[0]) == 5  # one entry per step
    assert len(losses[0]) == 2  # only step 0 and the last step
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


def test_train_disable_tqdm_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()
    seen_disable: list[bool] = []
    real_trange = jax_train.trange

    def _spy_trange(*args: object, **kwargs: object) -> object:
        seen_disable.append(kwargs["disable"])
        return real_trange(*args, **kwargs)

    monkeypatch.setattr(jax_train, "trange", _spy_trange)
    jax_train.train(model, ts=ts, ys=ys, training_steps=[(2, 1.0)], target_loss=-1.0)

    assert seen_disable == [False]


def test_train_disable_tqdm_true_suppresses_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()
    seen_disable: list[bool] = []
    real_trange = jax_train.trange

    def _spy_trange(*args: object, **kwargs: object) -> object:
        seen_disable.append(kwargs["disable"])
        return real_trange(*args, **kwargs)

    monkeypatch.setattr(jax_train, "trange", _spy_trange)
    jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0)],
        target_loss=-1.0,
        disable_tqdm=True,
    )

    assert seen_disable == [True]


def test_train_protocol_disable_tqdm_true_suppresses_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    seen_disable: list[bool] = []
    real_trange = jax_train.trange

    def _spy_trange(*args: object, **kwargs: object) -> object:
        seen_disable.append(kwargs["disable"])
        return real_trange(*args, **kwargs)

    monkeypatch.setattr(jax_train, "trange", _spy_trange)
    jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(2, 1.0)],
        target_loss=-1.0,
        disable_tqdm=True,
    )

    assert seen_disable == [True]


# ---------------------------------------------------------------------------
# train(): solver-failure retry
# ---------------------------------------------------------------------------


def test_train_retries_after_solver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A retry does NOT re-attempt the same step: on failure the model is
    # perturbed and the loop moves on to the next step index (same as
    # train.py's original pattern), so 2 synthetic failures consume 2 of
    # the 4 step slots, leaving 2 that actually succeed and get recorded.
    model = _decay_model()
    ts, ys = _tiny_training_data()
    real_make_step = jax_train.make_step
    calls = {"n": 0}

    def _flaky_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 2:
            msg = "synthetic solver failure"
            raise eqx.EquinoxRuntimeError(msg)
        return real_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "make_step", _flaky_make_step)

    trained, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(4, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )

    assert calls["n"] == 4
    assert isinstance(trained, Ode)
    assert len(losses[0]) == 2


def test_train_resets_solver_error_budget_per_curriculum_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the retry budget must not leak across curriculum stages.

    Stage 1 (2 steps, budget=1) fails twice in a row and gives up, leaving
    a stale error count. Stage 2 must start with a fresh budget: its own
    first failure should still be retried, not treated as already having
    exhausted the (stage 1) budget.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()
    real_make_step = jax_train.make_step
    calls = {"n": 0}

    def _flaky_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        # Fails on stage 1's two steps (calls 1, 2) and on stage 2's first
        # step (call 3); succeeds on stage 2's second step (call 4).
        if calls["n"] != 4:
            msg = "synthetic solver failure"
            raise eqx.EquinoxRuntimeError(msg)
        return real_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "make_step", _flaky_make_step)

    _, losses, grad_norms = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0), (2, 1.0)],
        avg_every=1,
        max_consecutive_solver_errors=1,
        target_loss=-1.0,
    )

    # If the budget leaked from stage 1, stage 2 would give up on its first
    # failure (call 3) without ever reaching call 4.
    assert calls["n"] == 4
    assert len(losses[0]) == 0
    assert len(grad_norms[0]) == 0
    assert len(losses[1]) == 1
    assert len(grad_norms[1]) == 1


def test_train_gives_up_after_max_consecutive_solver_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()

    def _always_fails(**_kwargs: object) -> object:
        msg = "synthetic solver failure"
        raise eqx.EquinoxRuntimeError(msg)

    monkeypatch.setattr(jax_train, "make_step", _always_fails)

    # Must not raise -- gives up on the stage and returns the initial model.
    trained, losses, grad_norms = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=1,
        max_consecutive_solver_errors=2,
        target_loss=-1.0,
    )
    assert trained is model
    assert len(losses[0]) == 0
    assert len(grad_norms[0]) == 0


# ---------------------------------------------------------------------------
# train(): graceful KeyboardInterrupt
# ---------------------------------------------------------------------------


def test_train_keyboard_interrupt_returns_best_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()
    real_make_step = jax_train.make_step
    calls = {"n": 0}

    def _interrupting_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "make_step", _interrupting_make_step)

    # Must not raise -- KeyboardInterrupt is caught and the best model so
    # far (plus partial loss/grad-norm history) is returned instead.
    trained, losses, grad_norms = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(10, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )
    assert calls["n"] == 3
    assert isinstance(trained, Ode)
    assert len(losses[0]) == 2
    assert len(grad_norms[0]) == 2


class _SchedulerShutdown(Exception):
    """Stand-in for a caller-defined exception raised from e.g. a SIGTERM
    handler (mirrors a downstream project's TrainingStopRequested)."""


def test_train_stops_on_custom_stop_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-defined exception (e.g. from a SIGTERM handler translating
    a scheduler's shutdown signal) is wound down from gracefully, the same
    as a manual KeyboardInterrupt, when listed in stop_exceptions.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()
    real_make_step = jax_train.make_step
    calls = {"n": 0}

    def _interrupting_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise _SchedulerShutdown
        return real_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "make_step", _interrupting_make_step)

    trained, losses, grad_norms = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(10, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        stop_exceptions=(KeyboardInterrupt, _SchedulerShutdown),
    )
    assert calls["n"] == 3
    assert isinstance(trained, Ode)
    assert len(losses[0]) == 2
    assert len(grad_norms[0]) == 2


def test_train_does_not_catch_unlisted_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop_exceptions is not a catch-all: an exception not listed still
    propagates, unlike KeyboardInterrupt/whatever's explicitly passed.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()

    def _raising_make_step(**_kwargs: object) -> object:
        raise _SchedulerShutdown

    monkeypatch.setattr(jax_train, "make_step", _raising_make_step)

    with pytest.raises(_SchedulerShutdown):
        jax_train.train(
            model,
            ts=ts,
            ys=ys,
            training_steps=[(2, 1.0)],
            avg_every=1,
            target_loss=-1.0,
        )


# ---------------------------------------------------------------------------
# proto_make_step: gradient clipping must only scale down, never up
# ---------------------------------------------------------------------------


def _fixed_grad_proto_grad_loss(magnitude: float) -> object:
    """Fake proto_grad_loss returning a constant-magnitude gradient for
    every trainable leaf, so the resulting global norm is exactly known.
    """

    def _fake(model: Ode, **_kwargs: object) -> tuple[jnp.ndarray, Ode]:
        params, _ = eqx.partition(model, eqx.is_inexact_array)
        grads = jax.tree.map(lambda x: jnp.full_like(x, magnitude), params)
        return jnp.array(0.123), grads

    return _fake


def _run_proto_make_step_with_fixed_grad(
    magnitude: float,
) -> tuple[Ode, Ode, jnp.ndarray]:
    """Runs proto_make_step with a fixed-magnitude grad_fn under
    jax.disable_jit(), so results don't depend on a cached compilation
    from another shape/value.
    """
    model = _decay_model()
    optim = optax.sgd(learning_rate=1.0)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    ctx = jax_train.IntegrationSettings()

    with jax.disable_jit():
        _, new_model, _, grad_norm = jax_train.proto_make_step(
            model=model,
            y0=jnp.array([1.0]),
            ts=[jnp.array([1.0])],
            ys=jnp.array([[1.0], [0.5]]),
            protocol=jnp.zeros((1, 0)),
            y_mean=jnp.array(0.0),
            y_scale=jnp.array(1.0),
            opt_state=opt_state,
            optim=optim,
            ctx=ctx,
            simulation_fn=jax_train.integrate_protocol_states,
            global_step=jnp.asarray(0),
            total_steps=jnp.asarray(1),
            grad_fn=_fixed_grad_proto_grad_loss(magnitude),
        )
    return model, new_model, grad_norm


def test_proto_make_step_does_not_rescale_small_gradients_upward() -> None:
    """Regression: clipping used to always rescale the gradient to unit
    global norm (``g * clip_value / (grad_norm + 1e-6)``), even when the
    raw gradient was already tiny -- a norm-0.0001 gradient got blown up
    by a factor of ~10000 instead of being left alone.
    """
    model, new_model, grad_norm = _run_proto_make_step_with_fixed_grad(magnitude=1e-4)
    assert float(grad_norm) == pytest.approx(1e-4 * (2**0.5), rel=1e-3)

    # SGD(lr=1) with an unclipped (scale ~= 1) gradient moves each leaf by
    # ~magnitude; under the old bug it would have moved by ~clip_value (1.0)
    # instead, seven orders of magnitude too far.
    assert float(jnp.abs(new_model.pars[0] - model.pars[0])) == pytest.approx(
        1e-4, rel=1e-2
    )
    assert float(jnp.abs(new_model.out_scale[0] - model.out_scale[0])) == pytest.approx(
        1e-4, rel=1e-2
    )


def test_proto_make_step_clips_large_gradients_to_unit_norm() -> None:
    """A genuinely large gradient must still be scaled down to (approximately)
    the clip norm, i.e. clipping in the "should actually clip" direction
    still works after the fix.
    """
    model, new_model, grad_norm = _run_proto_make_step_with_fixed_grad(magnitude=5.0)
    assert float(grad_norm) == pytest.approx(5.0 * (2**0.5), rel=1e-3)

    step_norm = jnp.sqrt(
        (new_model.pars[0] - model.pars[0]) ** 2
        + (new_model.out_scale[0] - model.out_scale[0]) ** 2
    )
    assert float(step_norm) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# train_protocol(): wraps optim like train() does (previously used the
# caller's optim directly, forcing callers who wanted the same robustness
# to hand-wrap it themselves before passing it in)
# ---------------------------------------------------------------------------


def test_train_protocol_wraps_optim_with_apply_if_finite_and_adaptive_grad_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    real_apply_if_finite = optax.apply_if_finite
    real_adaptive_grad_clip = optax.adaptive_grad_clip

    def _capturing_apply_if_finite(
        inner: object, *, max_consecutive_errors: int
    ) -> object:
        seen["max_consecutive_errors"] = max_consecutive_errors
        return real_apply_if_finite(
            inner, max_consecutive_errors=max_consecutive_errors
        )

    def _capturing_adaptive_grad_clip(clipping: float) -> object:
        seen["clip_norm"] = clipping
        return real_adaptive_grad_clip(clipping)

    monkeypatch.setattr(jax_train.optax, "apply_if_finite", _capturing_apply_if_finite)
    monkeypatch.setattr(
        jax_train.optax, "adaptive_grad_clip", _capturing_adaptive_grad_clip
    )

    model = _decay_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(1, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        clip_norm=0.5,
        max_consecutive_nonfinite=7,
    )

    assert seen["max_consecutive_errors"] == 7
    assert seen["clip_norm"] == 0.5


# ---------------------------------------------------------------------------
# train_protocol(): basic shape sanity (previously zero coverage)
# ---------------------------------------------------------------------------


def test_train_protocol_returns_losses_and_grad_norms_per_lesson() -> None:
    model = _decay_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, losses, grad_norms = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
    )

    assert isinstance(trained, Ode)
    assert len(losses) == 1
    assert len(grad_norms) == 1
    assert len(grad_norms[0]) == 3
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


# ---------------------------------------------------------------------------
# train_protocol(): early-stopping must return the best model, not the
# current step's (possibly worse) model
# ---------------------------------------------------------------------------


def test_train_protocol_early_stop_returns_best_model_not_current_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the early-stopping return path used to return ``model``
    (the just-computed model for the triggering step) instead of
    ``best_model``. Here stage 1's step reports a good loss for
    ``initial_model`` (the model it was actually called with), then stage
    2's step reports a worse (but still below target_loss) loss for
    whatever stage 1 produced -- proto_make_step's real contract is that
    the returned loss describes the model it was CALLED with (computed
    before that step's update is applied), so the tracked best is
    ``initial_model``, not either of the two models proto_make_step
    returns as its "new" model.
    """
    initial_model = Ode(rhs=_decay_rhs, pars=jnp.array([0.1]))  # the true best
    stage1_new_model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]))  # stage 1's update
    stage2_new_model = Ode(rhs=_decay_rhs, pars=jnp.array([0.9]))  # stage 2's update
    # Each entry's loss describes kwargs["model"] as passed into that call
    # (0.02 for initial_model, 0.5 for stage1_new_model) -- matching
    # proto_make_step's real pre-update-loss/post-update-model contract.
    sequence = [(0.02, stage1_new_model), (0.5, stage2_new_model)]
    calls = {"n": 0}

    def _fake_proto_make_step(
        **kwargs: object,
    ) -> tuple[jnp.ndarray, Ode, object, jnp.ndarray]:
        loss, new_model = sequence[calls["n"]]
        calls["n"] += 1
        return jnp.array(loss), new_model, kwargs["opt_state"], jnp.array(0.1)

    monkeypatch.setattr(jax_train, "proto_make_step", _fake_proto_make_step)

    trained, _, _ = jax_train.train_protocol(
        initial_model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(1, 1.0), (1, 1.0)],
        avg_every=1,
        target_loss=1.0,  # stage 2's 0.5 < 1.0 trips early stopping
    )

    assert calls["n"] == 2
    assert _allclose_models(trained, initial_model)
    assert not _allclose_models(trained, stage1_new_model)
    assert not _allclose_models(trained, stage2_new_model)


# ---------------------------------------------------------------------------
# train_protocol(): graceful stop -- previously had NO handling at all, so a
# KeyboardInterrupt (or any other exception) during training propagated and
# aborted the whole run instead of returning the best model/history so far.
# ---------------------------------------------------------------------------


def test_train_protocol_keyboard_interrupt_returns_best_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    real_proto_make_step = jax_train.proto_make_step
    calls = {"n": 0}

    def _interrupting_proto_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_proto_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "proto_make_step", _interrupting_proto_make_step)

    # Must not raise -- KeyboardInterrupt is caught and the best model so
    # far (plus partial loss/grad-norm history) is returned instead.
    trained, losses, grad_norms = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(10, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )
    assert calls["n"] == 3
    assert isinstance(trained, Ode)
    assert len(losses[0]) == 2
    assert len(grad_norms[0]) == 2


def test_train_protocol_stops_on_custom_stop_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    real_proto_make_step = jax_train.proto_make_step
    calls = {"n": 0}

    def _interrupting_proto_make_step(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise _SchedulerShutdown
        return real_proto_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "proto_make_step", _interrupting_proto_make_step)

    trained, losses, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(10, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        stop_exceptions=(KeyboardInterrupt, _SchedulerShutdown),
    )
    assert calls["n"] == 3
    assert isinstance(trained, Ode)
    assert len(losses[0]) == 2


def test_train_protocol_does_not_catch_unlisted_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()

    def _raising_proto_make_step(**_kwargs: object) -> object:
        raise _SchedulerShutdown

    monkeypatch.setattr(jax_train, "proto_make_step", _raising_proto_make_step)

    with pytest.raises(_SchedulerShutdown):
        jax_train.train_protocol(
            model,
            y0=jnp.array([1.0]),
            ts=[jnp.array([1.0])],
            ys=jnp.array([[1.0], [0.5]]),
            protocol=jnp.zeros((1, 0)),
            training_steps=[(2, 1.0)],
            avg_every=1,
            target_loss=-1.0,
        )


# ---------------------------------------------------------------------------
# train_only_nde(): basic shape sanity (previously returned a 2-tuple with
# no grad-norm history)
# ---------------------------------------------------------------------------


def test_train_only_nde_returns_losses_and_grad_norms_per_lesson() -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()

    trained, losses, grad_norms = jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=100,
        target_loss=-1.0,
    )

    assert isinstance(trained, Ude)
    assert len(losses) == 1
    assert len(grad_norms) == 1
    assert isinstance(losses[0], pd.Series)
    assert isinstance(grad_norms[0], pd.Series)
    assert len(grad_norms[0]) == 5
    assert len(losses[0]) == 2
    assert bool(jnp.all(jnp.isfinite(grad_norms[0].to_numpy())))


# ---------------------------------------------------------------------------
# train_only_nde(): the enumerate(training_steps) off-by-one used to make
# "i == len(training_steps)" unreachable, so early stopping and best-model
# tracking were both dead code
# ---------------------------------------------------------------------------


def test_train_only_nde_triggers_early_stopping_on_last_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with the old 0-based ``enumerate(training_steps)``,
    ``i == len(training_steps)`` was never true, so a loss under
    target_loss never returned early -- training always ran every step of
    every stage.
    """
    model = _ude_model()
    ts, ys = _tiny_training_data()
    calls = {"n": 0}

    def _fake_make_step_split(
        **kwargs: object,
    ) -> tuple[jnp.ndarray, object, object, jnp.ndarray]:
        calls["n"] += 1
        return (
            jnp.array(0.001),
            kwargs["trainable"],
            kwargs["opt_state"],
            jnp.array(0.1),
        )

    monkeypatch.setattr(jax_train, "make_step_split", _fake_make_step_split)

    jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=1,
        target_loss=0.01,  # every fake loss (0.001) is below this
    )

    assert calls["n"] == 1  # returned after the first step, not all 5


def test_train_only_nde_returns_best_model_not_last_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: with the dead best-model-tracking code, train_only_nde
    always returned the *last* step's model (or, prior to any step
    running, the untrained initial model) rather than the best one seen.

    make_step_split's real contract is that the returned loss describes
    ``trainable`` as it was CALLED with (computed before that step's
    update is applied) -- so each fake call's loss below describes the
    PREVIOUS call's returned trainable (trainable0 for call 0, trainable_a
    for call 1, trainable_b for call 2), and the best (lowest, 0.01) is
    reported for trainable_a -- the input to call 1, not either of the two
    trainables make_step_split returns as "new" results.
    """
    model = _ude_model()
    ts, ys = _tiny_training_data()

    filter_spec = jax.tree.map(eqx.is_inexact_array, model)
    filter_spec = eqx.tree_at(lambda m: m.ode, filter_spec, replace_fn=lambda _: False)
    trainable0, frozen = eqx.partition(model, filter_spec)
    trainable_a = jax_train.perturb_model(trainable0, jax.random.PRNGKey(1), 1.0)
    trainable_b = jax_train.perturb_model(trainable0, jax.random.PRNGKey(2), 1.0)
    trainable_c = jax_train.perturb_model(trainable0, jax.random.PRNGKey(3), 1.0)

    # (loss describing the call's input, new trainable to return)
    sequence = [
        (0.5, trainable_a),  # loss describes trainable0 (call 0's input)
        (0.01, trainable_b),  # loss describes trainable_a -- the true best
        (0.9, trainable_c),  # loss describes trainable_b (call 2's input)
    ]
    calls = {"n": 0}

    def _fake_make_step_split(
        **kwargs: object,
    ) -> tuple[jnp.ndarray, object, object, jnp.ndarray]:
        loss, new_trainable = sequence[calls["n"]]
        calls["n"] += 1
        return jnp.array(loss), new_trainable, kwargs["opt_state"], jnp.array(0.1)

    monkeypatch.setattr(jax_train, "make_step_split", _fake_make_step_split)

    trained, _, _ = jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0)],
        avg_every=1,
        target_loss=-1.0,  # never trip early stopping; exercise all 3 steps
    )

    expected_best = eqx.combine(trainable_a, frozen)
    assert calls["n"] == 3
    assert _allclose_models(trained, expected_best)
    assert not _allclose_models(trained, eqx.combine(trainable_b, frozen))
    assert not _allclose_models(trained, eqx.combine(trainable_c, frozen))


# ---------------------------------------------------------------------------
# train_only_nde(): solver-failure retry (previously absent entirely --
# an EquinoxRuntimeError would have propagated and aborted the whole run)
# ---------------------------------------------------------------------------


def test_train_only_nde_retries_after_solver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()
    real_make_step_split = jax_train.make_step_split
    calls = {"n": 0}

    def _flaky_make_step_split(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 2:
            msg = "synthetic solver failure"
            raise eqx.EquinoxRuntimeError(msg)
        return real_make_step_split(**kwargs)

    monkeypatch.setattr(jax_train, "make_step_split", _flaky_make_step_split)

    trained, losses, _ = jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(4, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )

    assert calls["n"] == 4
    assert isinstance(trained, Ude)
    assert len(losses[0]) == 2


def test_train_only_nde_gives_up_after_max_consecutive_solver_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()

    def _always_fails(**_kwargs: object) -> object:
        msg = "synthetic solver failure"
        raise eqx.EquinoxRuntimeError(msg)

    monkeypatch.setattr(jax_train, "make_step_split", _always_fails)

    # Must not raise -- gives up on the stage and returns the initial model.
    trained, losses, grad_norms = jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=1,
        max_consecutive_solver_errors=2,
        target_loss=-1.0,
    )
    assert trained is model
    assert len(losses[0]) == 0
    assert len(grad_norms[0]) == 0


# ---------------------------------------------------------------------------
# grad_fn: pluggable loss for train()/train_protocol()/train_only_nde()
# ---------------------------------------------------------------------------


def test_train_uses_custom_grad_fn() -> None:
    """A custom grad_fn (e.g. a residual loss against a frozen prior, or a
    different error metric) replaces the default normalised-MSE loss
    end-to-end, without needing to reimplement train()'s curriculum loop.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()

    @eqx.filter_value_and_grad
    def constant_loss_grad_fn(
        model: Ode,
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        # A loss value real training could never produce, so recording it
        # in the returned history proves this function -- not the default
        # grad_loss -- actually ran.
        return jnp.sum(model.pars) * 0.0 + 42.0

    _, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=constant_loss_grad_fn,
    )

    assert all(v == pytest.approx(42.0) for v in losses[0].to_numpy())


def test_train_protocol_uses_custom_grad_fn() -> None:
    model = _decay_model()

    @eqx.filter_value_and_grad
    def constant_loss_grad_fn(
        model: Ode,
        y0: jnp.ndarray,  # noqa: ARG001
        ts: list[jnp.ndarray],  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        protocol: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        simulation_fn: jax_train.ProtoSimulationFn,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + 42.0

    _, losses, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=constant_loss_grad_fn,
    )

    assert all(v == pytest.approx(42.0) for v in losses[0].to_numpy())


# ---------------------------------------------------------------------------
# train()/train_protocol(): normalize_axis
# ---------------------------------------------------------------------------


def test_train_normalize_axis_defaults_to_flattened_scalar() -> None:
    """The default (axis=None) matches the pre-existing behaviour: one
    scalar y_mean/y_scale shared across every observed quantity.
    """
    model = _decay_model()
    ts, ys = _multi_quantity_training_data()
    expected = float(jnp.mean(ys)) + float(jnp.std(ys))

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,
        y_scale: jnp.ndarray,
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + jnp.sum(y_mean) + jnp.sum(y_scale)

    _, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(1, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
    )

    assert losses[0].to_numpy()[0] == pytest.approx(expected)


def test_train_normalize_axis_zero_normalizes_per_quantity() -> None:
    """axis=0 normalises each column of ys independently -- needed when
    jointly-fit quantities have different scales, so a shared scalar
    doesn't let the higher-magnitude quantity dominate the loss.
    """
    model = _decay_model()
    ts, ys = _multi_quantity_training_data()
    expected = float(jnp.sum(jnp.mean(ys, axis=0)) + jnp.sum(jnp.std(ys, axis=0)))
    assert expected != pytest.approx(float(jnp.mean(ys)) + float(jnp.std(ys)))

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,
        y_scale: jnp.ndarray,
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + jnp.sum(y_mean) + jnp.sum(y_scale)

    _, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(1, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
        normalize_axis=0,
    )

    assert losses[0].to_numpy()[0] == pytest.approx(expected)


def test_train_protocol_normalize_axis_defaults_to_flattened_scalar() -> None:
    model = _decay_model()
    ys = jnp.array([[1.0, 2.0], [0.5, 3.0]])
    expected = float(jnp.mean(ys)) + float(jnp.std(ys))

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        y0: jnp.ndarray,  # noqa: ARG001
        ts: list[jnp.ndarray],  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        protocol: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,
        y_scale: jnp.ndarray,
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        simulation_fn: jax_train.ProtoSimulationFn,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + jnp.sum(y_mean) + jnp.sum(y_scale)

    _, losses, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=ys,
        protocol=jnp.zeros((1, 0)),
        training_steps=[(1, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
    )

    assert losses[0].to_numpy()[0] == pytest.approx(expected)


def test_train_protocol_normalize_axis_zero_normalizes_per_quantity() -> None:
    model = _decay_model()
    ys = jnp.array([[1.0, 2.0], [0.5, 3.0]])
    expected = float(jnp.sum(jnp.mean(ys, axis=0)) + jnp.sum(jnp.std(ys, axis=0)))
    assert expected != pytest.approx(float(jnp.mean(ys)) + float(jnp.std(ys)))

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        y0: jnp.ndarray,  # noqa: ARG001
        ts: list[jnp.ndarray],  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        protocol: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,
        y_scale: jnp.ndarray,
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        simulation_fn: jax_train.ProtoSimulationFn,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + jnp.sum(y_mean) + jnp.sum(y_scale)

    _, losses, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=ys,
        protocol=jnp.zeros((1, 0)),
        training_steps=[(1, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
        normalize_axis=0,
    )

    assert losses[0].to_numpy()[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# train()/train_protocol(): global_step/total_steps threading
# ---------------------------------------------------------------------------


def test_train_global_step_persists_across_curriculum_stages() -> None:
    """global_step must not reset between lessons -- it's a monotonic count
    across the WHOLE curriculum, not per-stage (needed e.g. to anneal a
    schedule like sigmoid_schedule smoothly across stage boundaries).
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,
        total_steps: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + global_step.astype(jnp.float32)

    _, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0), (2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
    )

    assert len(losses) == 2
    assert losses[0].to_numpy().tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert losses[1].to_numpy().tolist() == pytest.approx([3.0, 4.0])


def test_train_total_steps_is_the_curriculum_sum_throughout() -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + total_steps.astype(jnp.float32)

    _, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0), (2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
    )

    assert losses[0].to_numpy().tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert losses[1].to_numpy().tolist() == pytest.approx([5.0, 5.0])


def test_train_protocol_global_step_persists_across_curriculum_stages() -> None:
    model = _decay_model()
    ys = jnp.array([[1.0], [0.9], [0.8], [0.7]])

    @eqx.filter_value_and_grad
    def probe_grad_fn(
        model: Ode,
        y0: jnp.ndarray,  # noqa: ARG001
        ts: list[jnp.ndarray],  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        protocol: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        simulation_fn: jax_train.ProtoSimulationFn,  # noqa: ARG001
        global_step: jnp.ndarray,
        total_steps: jnp.ndarray,  # noqa: ARG001
    ) -> jnp.ndarray:
        return jnp.sum(model.pars) * 0.0 + global_step.astype(jnp.float32)

    _, losses, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])],
        ys=ys,
        protocol=jnp.zeros((3, 0)),
        training_steps=[(2, 1.0), (2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=probe_grad_fn,
    )

    assert len(losses) == 2
    assert losses[0].to_numpy().tolist() == pytest.approx([0.0, 1.0])
    assert losses[1].to_numpy().tolist() == pytest.approx([2.0, 3.0])


def test_train_only_nde_uses_custom_grad_fn() -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()

    @eqx.filter_value_and_grad
    def constant_loss_grad_fn(
        trainable: Ude,
        frozen: Ude,  # noqa: ARG001
        ts: jnp.ndarray,  # noqa: ARG001
        ys: jnp.ndarray,  # noqa: ARG001
        y_mean: jnp.ndarray,  # noqa: ARG001
        y_scale: jnp.ndarray,  # noqa: ARG001
        ctx: jax_train.IntegrationSettings,  # noqa: ARG001
        global_step: jnp.ndarray,  # noqa: ARG001
        total_steps: jnp.ndarray,  # noqa: ARG001
        args: jnp.ndarray | None = None,  # noqa: ARG001
    ) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(eqx.filter(trainable, eqx.is_array))
        return sum((jnp.sum(leaf) * 0.0 for leaf in leaves), jnp.array(0.0)) + 42.0

    _, losses, _ = jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_fn=constant_loss_grad_fn,
    )

    assert all(v == pytest.approx(42.0) for v in losses[0].to_numpy())


# ---------------------------------------------------------------------------
# prior_losses/prior_grad_norms: resume training across a checkpoint boundary
# ---------------------------------------------------------------------------


def test_train_resume_prepends_prior_history() -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()

    trained1, losses1, grad_norms1 = jax_train.train(
        model, ts=ts, ys=ys, training_steps=[(3, 1.0)], avg_every=1, target_loss=-1.0
    )
    assert len(losses1) == 1

    # Simulate resuming in a fresh process from a checkpointed
    # (model, losses, grad_norms) tuple.
    trained2, losses2, grad_norms2 = jax_train.train(
        trained1,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        prior_losses=losses1,
        prior_grad_norms=grad_norms1,
    )

    assert len(losses2) == 2
    assert len(grad_norms2) == 2
    assert losses2[0].equals(losses1[0])  # prior lesson preserved verbatim
    assert grad_norms2[0].equals(grad_norms1[0])
    assert isinstance(trained2, Ode)


def test_train_resume_never_regresses_below_the_starting_model() -> None:
    """Regression: best_training_loss started at jnp.inf on every call, so
    a resumed call whose optimiser (re-initialised for the new call)
    immediately overshoots could return a best_model objectively WORSE
    than the model resuming started from -- exactly the scenario a
    checkpoint/resume feature exists to protect against. Root cause was a
    separate, pre-existing off-by-one: make_step's returned loss describes
    the model as it was BEFORE that step's update, but the loop paired it
    with the (already updated) returned model instead of the pre-update
    one.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()

    trained1, losses1, _ = jax_train.train(
        model, ts=ts, ys=ys, training_steps=[(20, 1.0)], avg_every=100, target_loss=-1.0
    )
    ctx = jax_train.IntegrationSettings()
    y_mean, y_scale = jnp.mean(ys), jnp.std(ys)
    loss_before_resume, _ = jax_train.grad_loss(
        trained1,
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        global_step=jnp.asarray(0),
        total_steps=jnp.asarray(1),
    )

    # A deliberately huge learning rate simulates an optimiser overshoot
    # right after resuming (e.g. AdaBelief's momentum/variance state reset
    # to zero by a fresh optim.init(), same as this project's own
    # optax.adabelief default would do on any resumed call).
    trained2, _, _ = jax_train.train(
        trained1,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        optim=optax.sgd(learning_rate=50.0),
        prior_losses=losses1,
    )
    loss_after_resume, _ = jax_train.grad_loss(
        trained2,
        ts=ts,
        ys=ys,
        y_mean=y_mean,
        y_scale=y_scale,
        ctx=ctx,
        global_step=jnp.asarray(0),
        total_steps=jnp.asarray(1),
    )

    assert float(loss_after_resume) <= float(loss_before_resume) + 1e-4


def test_train_without_prior_history_starts_fresh() -> None:
    """Omitting prior_losses/prior_grad_norms (the default) must not change
    existing behaviour -- a single fresh lesson list, as before.
    """
    model = _decay_model()
    ts, ys = _tiny_training_data()

    _, losses, grad_norms = jax_train.train(
        model, ts=ts, ys=ys, training_steps=[(3, 1.0)], avg_every=1, target_loss=-1.0
    )
    assert len(losses) == 1
    assert len(grad_norms) == 1


def test_train_protocol_resume_prepends_prior_history() -> None:
    model = _decay_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([1.0])]
    ys = jnp.array([[1.0], [0.5]])
    protocol = jnp.zeros((1, 0))

    trained1, losses1, grad_norms1 = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )
    trained2, losses2, grad_norms2 = jax_train.train_protocol(
        trained1,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        prior_losses=losses1,
        prior_grad_norms=grad_norms1,
    )

    assert len(losses2) == 2
    assert len(grad_norms2) == 2
    assert losses2[0].equals(losses1[0])
    assert isinstance(trained2, Ode)


def test_train_only_nde_resume_prepends_prior_history() -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()

    trained1, losses1, grad_norms1 = jax_train.train_only_nde(
        model, ts=ts, ys=ys, training_steps=[(2, 1.0)], avg_every=1, target_loss=-1.0
    )
    trained2, losses2, grad_norms2 = jax_train.train_only_nde(
        trained1,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        prior_losses=losses1,
        prior_grad_norms=grad_norms1,
    )

    assert len(losses2) == 2
    assert len(grad_norms2) == 2
    assert losses2[0].equals(losses1[0])
    assert isinstance(trained2, Ude)


# ---------------------------------------------------------------------------
# GradParsHistory: optional per-parameter gradient tracking for train()/
# train_protocol() (Ode/FluxOde's flat .pars vector specifically)
# ---------------------------------------------------------------------------


def test_train_grad_pars_history_records_per_parameter_gradients() -> None:
    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5, 0.2]))
    ts, ys = _tiny_training_data()
    history = jax_train.GradParsHistory()

    trained, _, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(3, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_pars_history=history,
    )

    frames = history.to_frames(par_names=["k1", "k2"])
    assert isinstance(trained, Ode)
    assert len(frames) == 1
    assert list(frames[0].columns) == ["k1", "k2"]
    assert len(frames[0]) == 3
    # _decay_rhs uses args[-1] (pars[-1] = k2) only, so k1's gradient is
    # exactly zero throughout -- a real, checkable per-parameter signal,
    # not just "some numbers came back".
    assert (frames[0]["k1"] == 0.0).all()
    assert not (frames[0]["k2"] == 0.0).any()


def test_train_grad_pars_history_defaults_columns_to_positional_index() -> None:
    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5, 0.2]))
    ts, ys = _tiny_training_data()
    history = jax_train.GradParsHistory()

    jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_pars_history=history,
    )

    frames = history.to_frames()
    assert list(frames[0].columns) == [0, 1]


def test_train_without_grad_pars_history_is_unaffected() -> None:
    """Omitting grad_pars_history (the default) must not change existing
    behaviour or require the model to have a flat .pars at all.
    """
    model = _ude_model()  # no flat .pars -- would break if tracking were forced on
    ts, ys = _tiny_training_data()

    trained, losses, _ = jax_train.train_only_nde(
        model, ts=ts, ys=ys, training_steps=[(2, 1.0)], avg_every=1, target_loss=-1.0
    )
    assert isinstance(trained, Ude)
    assert len(losses) == 1


def test_train_protocol_grad_pars_history_records_per_parameter_gradients() -> None:
    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5, 0.2]))
    history = jax_train.GradParsHistory()

    trained, _, _ = jax_train.train_protocol(
        model,
        y0=jnp.array([1.0]),
        ts=[jnp.array([1.0])],
        ys=jnp.array([[1.0], [0.5]]),
        protocol=jnp.zeros((1, 0)),
        training_steps=[(2, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_pars_history=history,
    )

    frames = history.to_frames(par_names=["k1", "k2"])
    assert isinstance(trained, Ode)
    assert len(frames) == 1
    assert list(frames[0].columns) == ["k1", "k2"]
    assert len(frames[0]) == 2
    assert (frames[0]["k1"] == 0.0).all()
    assert not (frames[0]["k2"] == 0.0).any()


def test_grad_pars_history_multiple_lessons_kept_separate() -> None:
    model = Ode(rhs=_decay_rhs, pars=jnp.array([0.5]))
    ts, ys = _tiny_training_data()
    history = jax_train.GradParsHistory()

    jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(2, 1.0), (3, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        grad_pars_history=history,
    )

    frames = history.to_frames()
    assert len(frames) == 2
    assert len(frames[0]) == 2
    assert len(frames[1]) == 3


# ---------------------------------------------------------------------------
# training_steps cutoffs: float fraction-of-data vs. explicit int count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "cutoff", "expected"),
    [
        (5, 1.0, 5),  # float: fraction of the full length
        (5, 0.5, 3),  # float: rounded up (ceil(2.5) == 3)
        (5, 3, 3),  # int: explicit count, within range
        (5, 10, 5),  # int: explicit count, clipped to total
    ],
)
def test_curriculum_length(total: int, cutoff: float, expected: int) -> None:
    assert jax_train._curriculum_length(total, cutoff) == expected


def test_train_explicit_int_cutoff_slices_exact_point_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    ts, ys = _tiny_training_data()  # 5 points
    real_make_step = jax_train.make_step
    seen_lengths = []

    def _capturing_make_step(**kwargs: object) -> object:
        seen_lengths.append(kwargs["ts"].shape[0])  # type: ignore[union-attr]
        return real_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "make_step", _capturing_make_step)

    jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(1, 3)],  # explicit cutoff: first 3 points, not 3/5
        avg_every=1,
        target_loss=-1.0,
    )

    assert seen_lengths == [3]


def test_train_protocol_explicit_int_cutoff_slices_exact_window_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _decay_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0]), jnp.array([1.5])]
    protocol = jnp.zeros((3, 0))
    ys = jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0, 1.5]))[:, None]
    real_proto_make_step = jax_train.proto_make_step
    seen_lengths = []

    def _capturing_proto_make_step(**kwargs: object) -> object:
        seen_lengths.append(len(kwargs["ts"]))  # type: ignore[arg-type]
        return real_proto_make_step(**kwargs)

    monkeypatch.setattr(jax_train, "proto_make_step", _capturing_proto_make_step)

    jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(1, 2)],  # explicit cutoff: first 2 windows, not 2/3
        avg_every=1,
        target_loss=-1.0,
    )

    assert seen_lengths == [2]


def test_train_only_nde_explicit_int_cutoff_slices_exact_point_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()  # 5 points
    real_make_step_split = jax_train.make_step_split
    seen_lengths = []

    def _capturing_make_step_split(**kwargs: object) -> object:
        seen_lengths.append(kwargs["ts"].shape[0])  # type: ignore[union-attr]
        return real_make_step_split(**kwargs)

    monkeypatch.setattr(jax_train, "make_step_split", _capturing_make_step_split)

    jax_train.train_only_nde(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(1, 3)],  # explicit cutoff: first 3 points, not 3/5
        avg_every=1,
        target_loss=-1.0,
    )

    assert seen_lengths == [3]


# ---------------------------------------------------------------------------
# freeze: generalizes train_only_nde's hardcoded ``m.ode`` freeze to an
# arbitrary subtree, for train()/train_protocol()
# ---------------------------------------------------------------------------


def test_train_freeze_keeps_frozen_subtree_exactly_unchanged() -> None:
    model = _ude_model()
    ts, ys = _tiny_training_data()

    trained, _, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(5, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        freeze=lambda m: m.ode,
    )

    assert jnp.array_equal(trained.ode.pars, model.ode.pars)
    trained_nn_leaf = jax.tree_util.tree_leaves(eqx.filter(trained.nn, eqx.is_array))[0]
    initial_nn_leaf = jax.tree_util.tree_leaves(eqx.filter(model.nn, eqx.is_array))[0]
    assert not jnp.allclose(trained_nn_leaf, initial_nn_leaf)


def test_train_freeze_retries_after_solver_error_without_perturbing_frozen_subtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver failure's perturb-and-retry recovery (see
    test_train_retries_after_solver_error) must perturb the trainable
    partition only -- perturbing the frozen one would silently drift a
    value that's supposed to stay fixed forever.
    """
    model = _ude_model()
    ts, ys = _tiny_training_data()
    real_make_step_split = jax_train.make_step_split
    calls = {"n": 0}

    def _flaky_make_step_split(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 2:
            msg = "synthetic solver failure"
            raise eqx.EquinoxRuntimeError(msg)
        return real_make_step_split(**kwargs)

    monkeypatch.setattr(jax_train, "make_step_split", _flaky_make_step_split)

    trained, losses, _ = jax_train.train(
        model,
        ts=ts,
        ys=ys,
        training_steps=[(4, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        freeze=lambda m: m.ode,
    )

    assert calls["n"] == 4
    assert jnp.array_equal(trained.ode.pars, model.ode.pars)
    assert len(losses[0]) == 2


def test_train_freeze_defaults_to_none_and_matches_a_hand_rolled_make_step_loop() -> (
    None
):
    """freeze's default (None) must reproduce train()'s pre-existing,
    whole-model-trainable behaviour: this replays the exact sequence of
    public make_step() calls a hand-rolled loop would make -- what train()
    did before the freeze/split_grad_fn refactor -- and checks train()
    matches it step-by-step. This catches a bug shared by both the
    omitted-freeze and freeze=None paths (which now share the same
    internal closures), unlike comparing train() against another call of
    itself.
    """
    ts, ys = _tiny_training_data()
    steps = 3

    optim = optax.apply_if_finite(
        optax.chain(
            optax.adaptive_grad_clip(1.0),
            optax.adabelief(learning_rate=1e-4),
        ),
        max_consecutive_errors=50,
    )
    ctx = jax_train.IntegrationSettings()
    y_mean = jnp.mean(ys)
    y_scale = jnp.std(ys)

    hand_rolled_model = _decay_model()
    opt_state = optim.init(eqx.filter(hand_rolled_model, eqx.is_inexact_array))
    expected_losses: dict[int, float] = {}
    best_pre_step_model = hand_rolled_model
    best_loss = jnp.inf
    for step in range(steps):
        pre_step_model = hand_rolled_model
        loss, hand_rolled_model, opt_state, _ = jax_train.make_step(
            model=hand_rolled_model,
            ts=ts,
            ys=ys,
            opt_state=opt_state,
            optim=optim,
            y_mean=y_mean,
            y_scale=y_scale,
            ctx=ctx,
            global_step=jnp.asarray(step),
            total_steps=jnp.asarray(steps),
        )
        expected_losses[step] = float(loss)
        if loss < best_loss:
            best_loss = loss
            best_pre_step_model = pre_step_model

    trained, losses, _ = jax_train.train(
        _decay_model(),
        ts=ts,
        ys=ys,
        training_steps=[(steps, 1.0)],
        avg_every=1,
        target_loss=-1.0,
    )

    assert losses[0].to_dict() == pytest.approx(expected_losses)
    assert _allclose_models(trained, best_pre_step_model)


def test_train_freeze_raises_with_grad_pars_history() -> None:
    ts, ys = _tiny_training_data()
    with pytest.raises(ValueError, match="freeze"):
        jax_train.train(
            _ude_model(),
            ts=ts,
            ys=ys,
            training_steps=[(1, 1.0)],
            freeze=lambda m: m.ode,
            grad_pars_history=jax_train.GradParsHistory(),
        )


def test_train_protocol_freeze_keeps_frozen_subtree_exactly_unchanged() -> None:
    model = _ude_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]

    trained, _, _ = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(3, 1.0)],
        avg_every=100,
        target_loss=-1.0,
        freeze=lambda m: m.ode,
    )

    assert jnp.array_equal(trained.ode.pars, model.ode.pars)


def test_train_protocol_freeze_retries_after_solver_error_without_perturbing_frozen_subtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors test_train_freeze_retries_after_solver_error_without_perturbing_frozen_subtree
    for train_protocol()/proto_make_step_split.
    """
    model = _ude_model()
    y0 = jnp.array([1.0])
    ts = [jnp.array([0.5]), jnp.array([1.0])]
    protocol = jnp.zeros((2, 0))
    ys = jnp.exp(-0.3 * jnp.array([0.0, 0.5, 1.0]))[:, None]
    real_proto_make_step_split = jax_train.proto_make_step_split
    calls = {"n": 0}

    def _flaky_proto_make_step_split(**kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] <= 2:
            msg = "synthetic solver failure"
            raise eqx.EquinoxRuntimeError(msg)
        return real_proto_make_step_split(**kwargs)

    monkeypatch.setattr(
        jax_train, "proto_make_step_split", _flaky_proto_make_step_split
    )

    trained, losses, _ = jax_train.train_protocol(
        model,
        y0=y0,
        ts=ts,
        ys=ys,
        protocol=protocol,
        training_steps=[(4, 1.0)],
        avg_every=1,
        target_loss=-1.0,
        freeze=lambda m: m.ode,
    )

    assert calls["n"] == 4
    assert jnp.array_equal(trained.ode.pars, model.ode.pars)
    assert len(losses[0]) == 2


def test_train_protocol_freeze_raises_with_grad_pars_history() -> None:
    with pytest.raises(ValueError, match="freeze"):
        jax_train.train_protocol(
            _ude_model(),
            y0=jnp.array([1.0]),
            ts=[jnp.array([0.5])],
            ys=jnp.exp(-0.3 * jnp.array([0.0, 0.5]))[:, None],
            protocol=jnp.zeros((1, 0)),
            training_steps=[(1, 1.0)],
            freeze=lambda m: m.ode,
            grad_pars_history=jax_train.GradParsHistory(),
        )
