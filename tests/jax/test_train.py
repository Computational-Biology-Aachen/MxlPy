"""Tests for mxlpy.jax.train: squeeze_derived, _perturb_model, and train()'s
solver-retry / KeyboardInterrupt / grad-norm-history behaviour.

train() and make_step previously had zero test coverage.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from mxlpy.jax import train as jax_train
from mxlpy.jax.models import Ode

_KEY = jax.random.PRNGKey(0)


def _decay_rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
    return -args[-1] * y


def _decay_model() -> Ode:
    return Ode(rhs=_decay_rhs, pars=jnp.array([0.5]))


def _tiny_training_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    ts = jnp.linspace(0.0, 1.0, 5)
    ys = jnp.exp(-0.3 * ts)[:, None]
    return ts, ys


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
# _perturb_model
# ---------------------------------------------------------------------------


def test_perturb_model_changes_trainable_leaves() -> None:
    model = _decay_model()
    perturbed = jax_train._perturb_model(model, _KEY, scale=0.1)
    assert not jnp.allclose(model.pars, perturbed.pars)


def test_perturb_model_leaves_zero_scale_unchanged() -> None:
    model = _decay_model()
    perturbed = jax_train._perturb_model(model, _KEY, scale=0.0)
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
