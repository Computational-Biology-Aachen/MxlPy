"""Physical invariants of the JAX models' simulation and training paths.

Line coverage of ``mxlpy.jax`` was already ~98% when ``FluxAnode`` was found
to recompute fluxes from re-encoded observations -- resetting the hidden
latent dimensions to their ``t=0`` encoding at every time point -- because
no test checked that the reported fluxes actually *produce* the reported
trajectory. These tests check invariants independent of how a quantity is
computed internally:

* **Balance**: on a dense grid, the cumulative trapezoid integral of
  ``nv(fluxes)`` must equal the change in the (latent) state.
* **Steady state**: ``nv(fluxes)`` -- and, for ANODEs, the Markov part --
  must vanish at the returned steady state.
* **Path agreement**: every integration route (plain, protocol, split
  protocol, from steady state, ``simulate_*``, batched, saved/loaded,
  training losses) must see the same trajectory.

Each state-reset invariant has a negative control showing it rejects the
old re-encoding behaviour, so a passing test is not vacuous.

Memory: every distinct model structure and save-grid shape compiles its own
diffrax solver, and JAX keeps each executable for the process lifetime. To
keep this suite runnable on small CI runners, all trajectories share one
grid (``_TS``) and one two-step protocol shape, ``max_steps`` is small (it
sizes the solver's buffers), and checks on the same model are merged into
one test (reported individually via ``subtests``) so each model compiles in
exactly one xdist worker.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mxlpy import Model, fns
from mxlpy.jax import io as jax_io
from mxlpy.jax.ensemble import (
    batch_simulate,
    load_stacked_models,
    stack_models,
    unstack_models,
)
from mxlpy.jax.models import (
    Anode,
    FluxAnode,
    FluxNode,
    FluxOde,
    FluxUde,
    FullLatentMapper,
    HardLatentMapper,
    MixedLatentMapper,
    Node,
    Ode,
    SoftLatentMapper,
    TrainableZ0LatentMapper,
    Ude,
)
from mxlpy.jax.simulation import FluxOdeSimulation
from mxlpy.jax.train import (
    IntegrationSettings,
    derived_simulation_fn,
    grad_loss,
    integrate_protocol_states,
    make_step_split,
    proto_grad_loss,
)

_KEY = jax.random.PRNGKey(42)
_MAX_STEPS = 4096
_RTOL = 1e-6
# One grid for every trajectory: dt = 0.01, fine enough for the trapezoid
# balance check. The protocol splits it into two equal-length steps.
_TS = jnp.linspace(0.0, 2.0, 201)
_PROTO_TS = [_TS[1:101], _TS[101:]]
_Y0 = jnp.array([1.0, 0.5, 0.2])
_U = jnp.array([0.8])
_PROTOCOL = jnp.array([[0.8], [1.5]])  # u changes at t = 1
_PROTOCOL_CONST = jnp.stack([_U, _U])  # same args in both steps
_N_HIDDEN = 2

ALL_MAPPERS = [
    HardLatentMapper,
    SoftLatentMapper,
    FullLatentMapper,
    MixedLatentMapper,
    TrainableZ0LatentMapper,
]
_ANNEALED = (MixedLatentMapper, TrainableZ0LatentMapper)
_SLICE_DECODING = (HardLatentMapper, *_ANNEALED)
# alpha=0.0 makes the annealed mappers decode by slicing, so the observed
# state also obeys the balance; mappers without alpha don't accept it.
_ALPHA = 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _times(x: float, c: float) -> float:
    return x * c


def _model() -> Model:
    """``-> A -> B``, ``2 B -> S``, ``S ->``; linear so a steady state exists.

    The ``2 B`` stoichiometry makes the balance check sensitive to ``nv``
    being applied correctly rather than e.g. as an identity map.
    """
    m = Model()
    m = m.add_variables({"A": 1.0, "B": 0.5, "S": 0.2})
    m = m.add_parameters({"u": 0.8, "k1": 1.5, "k2": 0.7, "k3": 0.9, "c": 5.0})
    m = m.add_derived("d", _times, args=["B", "c"])
    m = m.add_reaction("v_in", fns.constant, args=["u"], stoichiometry={"A": 1})
    m = m.add_reaction(
        "v1", fns.mass_action_1s, args=["A", "k1"], stoichiometry={"A": -1, "B": 1}
    )
    m = m.add_reaction(
        "v2", fns.mass_action_1s, args=["B", "k2"], stoichiometry={"B": -2, "S": 1}
    )
    return m.add_reaction(
        "v3", fns.mass_action_1s, args=["S", "k3"], stoichiometry={"S": -1}
    )


@functools.cache
def _flux_ode() -> FluxOde:
    # Cached: from_mxlpy exec()s fresh fluxes/nv/derived functions on every
    # call, and as static leaves they key JAX's compile caches -- uncached,
    # every fixture model would compile its own solver.
    return FluxOde.from_mxlpy(
        _model(),
        parameters_to_fit=["k1", "c"],
        free_parameters=["u"],
        derived_to_calculate=["d"],
    )


@functools.cache
def _ode() -> Ode:
    return Ode.from_mxlpy(_model(), parameters_to_fit=["k1"], free_parameters=["u"])


def _flux_node() -> FluxNode:
    return FluxNode.from_flux_ode(
        _flux_ode(), width=8, depth=1, key=_KEY, n_args=1, out_scale=jnp.array([1.0])
    )


def _node() -> Node:
    return Node(n_obs=3, width=8, depth=1, key=_KEY, n_args=1)


def _flux_anode(mapper: type) -> FluxAnode:
    """A linear ANODE with a unique steady state and a strongly coupled hidden state.

    Fluxes are the mechanistic mass-action rates plus a dependence on the
    hidden dims, which relax linearly towards ``h* = 0.5`` from wherever the
    encoder puts them. So the hidden state drifts, feeds back into the
    fluxes, and the steady-state fluxes depend on ``h*`` -- not on what the
    encoder maps the observations to -- which is what makes a hidden-state
    reset detectable.
    """
    model = FluxAnode.from_flux_ode(
        _flux_ode(),
        n_hidden=_N_HIDDEN,
        key=_KEY,
        n_args=1,
        flux_depth=0,
        markov_depth=0,
        flux_scale=jnp.array([1.0]),
        markov_scale=jnp.array([1.0]),
        latent_mapper=mapper,
    )
    n_latent = cast(Any, model.latent_mapper).n_latent
    # inputs: z (n_latent) then u; outputs: v_in, v1, v2, v3
    w_flux = jnp.zeros((4, n_latent + 1))
    w_flux = w_flux.at[0, n_latent].set(1.0)  # v_in = u
    w_flux = w_flux.at[1, 0].set(1.5)  # v1 = k1 A
    w_flux = w_flux.at[2, 1].set(0.7)  # v2 = k2 B
    w_flux = w_flux.at[3, 2].set(0.9)  # v3 = k3 S
    w_flux = w_flux.at[1, 3].set(1.0)  # v1 += h0
    w_flux = w_flux.at[3, 4].set(1.0)  # v3 += h1
    w_markov = jnp.zeros((_N_HIDDEN, n_latent + 1))
    w_markov = w_markov.at[0, 3].set(-1.0).at[1, 4].set(-1.0)
    model = eqx.tree_at(
        lambda m: (
            m.flux_nn.layers[0].weight,
            m.flux_nn.layers[0].bias,
            m.markov_nn.layers[0].weight,
            m.markov_nn.layers[0].bias,
        ),
        model,
        (w_flux, jnp.zeros(4), w_markov, jnp.full(_N_HIDDEN, 0.5)),
    )
    mapper_ = cast(Any, model.latent_mapper)
    # Hidden dims start at -1 (far from h* = 0.5): via the encoder bias, or
    # directly for TrainableZ0's learned z0.
    if isinstance(mapper_, TrainableZ0LatentMapper):
        model = eqx.tree_at(lambda m: m.latent_mapper.z0, model, -jnp.ones(_N_HIDDEN))
    elif isinstance(mapper_, FullLatentMapper):
        # z0 = encoder(y0): observed part = observations, hidden part = -1
        enc = jnp.zeros((n_latent, 3)).at[:3, :3].set(jnp.eye(3))
        bias = jnp.zeros(n_latent).at[3:].set(-1.0)
        model = eqx.tree_at(
            lambda m: (m.latent_mapper.encoder.weight, m.latent_mapper.encoder.bias),
            model,
            (enc, bias),
        )
    else:
        model = eqx.tree_at(
            lambda m: (m.latent_mapper.encoder.weight, m.latent_mapper.encoder.bias),
            model,
            (jnp.zeros((_N_HIDDEN, 3)), -jnp.ones(_N_HIDDEN)),
        )
    return model


def _anode(mapper: type) -> Anode:
    """Linear, stable Anode whose observed dims are driven by the hidden ones.

    Every latent dim relaxes with rate 1 towards ``0.5 + 0.2 u``; observed
    dims 0 and 2 are additionally driven by hidden dims 0 and 1, so a reset
    of the hidden state shows up in the decoded trajectory even for the
    slice-decoding mappers.
    """
    model = Anode(
        n_obs=3,
        n_hidden=_N_HIDDEN,
        width=8,
        depth=0,
        key=_KEY,
        n_args=1,
        out_scale=jnp.array([1.0]),
        latent_mapper=mapper,
    )
    n_latent = cast(Any, model.latent_mapper).n_latent
    w = jnp.zeros((n_latent, n_latent + 1))
    w = w.at[:, :n_latent].set(-jnp.eye(n_latent)).at[:, n_latent].set(0.2)
    w = w.at[0, 3].set(0.5).at[2, 4].set(0.5)
    return eqx.tree_at(
        lambda m: (m.nn.layers[0].weight, m.nn.layers[0].bias),
        model,
        (w, jnp.full(n_latent, 0.5)),
    )


def _alpha_kw(model: Any) -> dict[str, Any]:
    mapper = getattr(model, "latent_mapper", None)
    return {"alpha": _ALPHA} if isinstance(mapper, _ANNEALED) else {}


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

_BALANCE_TOL = 1e-3


def _cumtrapz(ts: jax.Array, f: jax.Array) -> jax.Array:
    dt = jnp.diff(ts)[:, None]
    steps = 0.5 * dt * (f[1:] + f[:-1])
    return jnp.concatenate((jnp.zeros((1, f.shape[1])), jnp.cumsum(steps, axis=0)))


def _balance_error(
    ts: jax.Array,
    state: jax.Array,
    fluxes: jax.Array,
    nv: Callable[[jax.Array], jax.Array],
) -> float:
    """Max |∫ nv(v) dt - Δstate|, relative to the largest state change."""
    produced = _cumtrapz(ts, jax.vmap(nv)(fluxes))
    n = produced.shape[1]
    change = state[:, :n] - state[0, :n]
    scale = float(jnp.max(jnp.abs(change)))
    assert scale > 1e-2, "state barely moves; the balance check would be vacuous"
    return float(jnp.max(jnp.abs(produced - change))) / scale


def _sim_fluxes(sim: FluxOdeSimulation) -> jax.Array:
    return jnp.asarray(sim.fluxes.to_numpy())


def _state(sim: FluxOdeSimulation) -> jax.Array:
    return sim.ys if sim.latent_ys is None else sim.latent_ys


def _assert_balanced(sim: FluxOdeSimulation, nv: Callable) -> None:
    err = _balance_error(sim.ts, _state(sim), _sim_fluxes(sim), nv)
    assert err < _BALANCE_TOL, f"flux/state balance violated: rel. error {err:.2e}"


def _assert_protocol_balanced(sim: FluxOdeSimulation, nv: Callable) -> None:
    """Balance within each protocol step.

    Fluxes jump where ``u`` changes, and the trapezoid rule would smear that
    jump across the grid spacing straddling the boundary, so only rows
    belonging to the same step are integrated together (row 0 is ``t=0``).
    """
    state, fluxes = _state(sim), _sim_fluxes(sim)
    offset = 1
    for step_ts in _PROTO_TS:
        seg = slice(offset, offset + step_ts.shape[0])
        err = _balance_error(sim.ts[seg], state[seg], fluxes[seg], nv)
        assert err < _BALANCE_TOL, f"protocol step balance violated: {err:.2e}"
        offset += step_ts.shape[0]


def _assert_steady(sim: FluxOdeSimulation, nv: Callable) -> None:
    residual = nv(_sim_fluxes(sim)[0])
    assert float(jnp.max(jnp.abs(residual))) < 1e-3 * float(_U[0])


def _integrate(model: Any, **kw: Any) -> jax.Array:
    return model.integrate(_TS, _Y0, _MAX_STEPS, args=_U, rtol=_RTOL, **kw)


def _integrate_protocol(model: Any, protocol: jax.Array, **kw: Any) -> jax.Array:
    return model.integrate_protocol(
        _PROTO_TS, _Y0, protocol, _MAX_STEPS, rtol=_RTOL, **kw
    )


def _simulate(model: Any, **kw: Any) -> FluxOdeSimulation:
    return model.simulate_time_course(_TS, _Y0, _MAX_STEPS, args=_U, rtol=_RTOL, **kw)


# ---------------------------------------------------------------------------
# Every model class: path agreement, ensembles/io, flux balance, losses
# ---------------------------------------------------------------------------


def _members(model: Any, n: int = 3) -> list[Any]:
    """``n`` copies of ``model`` with every float leaf perturbed differently.

    Perturbing *all* leaves (weights, pars, z0, scales, ...) means a leaf
    that stacking/saving accidentally shares between members, or resets,
    changes some member's output.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jax.tree.flatten(params)
    out = []
    for i in range(n):
        keys = jax.random.split(jax.random.PRNGKey(100 + i), len(leaves))
        noisy = [
            leaf * (1 + 0.05 * jax.random.normal(k, leaf.shape))
            for leaf, k in zip(leaves, keys, strict=True)
        ]
        out.append(eqx.combine(jax.tree.unflatten(treedef, noisy), static))
    return out


# Module-level, so batch_simulate's compile cache (keyed on the callable)
# is shared between the checks below.
def _integrate_plain(model: Any) -> jax.Array:
    return _integrate(model)


def _integrate_alpha(model: Any) -> jax.Array:
    return _integrate(model, alpha=_ALPHA)


def _simulate_plain(model: Any) -> FluxOdeSimulation:
    return _simulate(model)


def _simulate_alpha(model: Any) -> FluxOdeSimulation:
    return _simulate(model, alpha=_ALPHA)


# vmapped vs. single adaptive solves differ by float32 noise (~1e-5); the
# perturbed members differ by > 5e-3 (asserted below).
_BATCH_RTOL = _BATCH_ATOL = 1e-4


def _check_path_agreement(model: Any) -> None:
    kw = _alpha_kw(model)
    ys = _integrate(model, **kw)
    # Two protocol steps with identical args == one plain integration;
    # catches any reset of state carried across a step boundary.
    ys_proto = _integrate_protocol(model, _PROTOCOL_CONST, **kw)
    np.testing.assert_allclose(ys_proto, ys[1:], rtol=1e-4, atol=1e-5)
    if hasattr(model, "simulate_time_course"):
        np.testing.assert_allclose(_simulate(model, **kw).ys, ys, rtol=1e-6, atol=1e-7)
        sim_p = model.simulate_protocol_time_course(
            _PROTO_TS, _Y0, _PROTOCOL, _MAX_STEPS, rtol=_RTOL, **kw
        )
        np.testing.assert_allclose(sim_p.ys[0], _Y0)
        np.testing.assert_allclose(
            sim_p.ys[1:],
            _integrate_protocol(model, _PROTOCOL, **kw),
            rtol=1e-6,
            atol=1e-7,
        )


def _check_ensembles_and_io(model: Any, tmp_path: Path) -> None:
    kw = _alpha_kw(model)
    members = _members(model)
    singles = [_integrate(m, **kw) for m in members]
    assert float(jnp.max(jnp.abs(singles[0] - singles[1]))) > 5e-3, (
        "perturbed members indistinguishable; the checks below would be vacuous"
    )
    integrate_fn = _integrate_alpha if kw else _integrate_plain

    batched = batch_simulate(members, integrate_fn)
    for i, want in enumerate(singles):
        np.testing.assert_allclose(batched[i], want, rtol=_BATCH_RTOL, atol=_BATCH_ATOL)

    for orig, back in zip(
        members, unstack_models(stack_models(members), len(members)), strict=True
    ):
        a, b = jax.tree.leaves(orig), jax.tree.leaves(back)
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            if eqx.is_array(x):
                np.testing.assert_array_equal(x, y)
            else:
                assert x is y or x == y

    paths = [tmp_path / f"{i}.dill" for i in range(len(members))]
    for p, m in zip(paths, members, strict=True):
        jax_io.save(p, m)
    stacked = load_stacked_models(paths, template=model)
    loaded = batch_simulate(unstack_models(stacked, len(members)), integrate_fn)
    for i, want in enumerate(singles):
        np.testing.assert_allclose(loaded[i], want, rtol=_BATCH_RTOL, atol=_BATCH_ATOL)

    if isinstance(model, FluxAnode):
        # Batched FluxOdeSimulations must carry each member's own latent
        # trajectory, so unstacked fluxes match the member's own simulation.
        simulate_fn = _simulate_alpha if kw else _simulate_plain
        unstacked = unstack_models(batch_simulate(members, simulate_fn), len(members))
        for m, got in zip(members, unstacked, strict=True):
            want = simulate_fn(m)
            np.testing.assert_allclose(
                got.ys, want.ys, rtol=_BATCH_RTOL, atol=_BATCH_ATOL
            )
            if want.latent_ys is not None:
                assert got.latent_ys is not None
                np.testing.assert_allclose(
                    got.latent_ys, want.latent_ys, rtol=_BATCH_RTOL, atol=_BATCH_ATOL
                )
            np.testing.assert_allclose(
                got.fluxes.to_numpy(),
                want.fluxes.to_numpy(),
                rtol=_BATCH_RTOL,
                atol=_BATCH_ATOL,
            )


_Y_MEAN = jnp.array([0.5, 0.5, 0.5])
_Y_SCALE = jnp.array([1.0, 2.0, 0.5])
_STEP: dict[str, Any] = {"global_step": jnp.array(0), "total_steps": jnp.array(1)}


def _mse(ys: jax.Array, y_pred: jax.Array) -> jax.Array:
    return jnp.mean(((ys - _Y_MEAN) / _Y_SCALE - (y_pred - _Y_MEAN) / _Y_SCALE) ** 2)


def _ctx(model: Any) -> IntegrationSettings:
    """Solver settings; annealed mappers get their alpha through ``ctx``."""
    return IntegrationSettings(rtol=_RTOL, max_steps=_MAX_STEPS, **_alpha_kw(model))


def _filter_spec(model: Any) -> object:
    """Freeze every other float leaf, so both groups are non-trivial."""
    flags, treedef = jax.tree.flatten(jax.tree.map(eqx.is_inexact_array, model))
    out, n = [], 0
    for f in flags:
        out.append(f and n % 2 == 0)
        n += f
    return jax.tree.unflatten(treedef, out)


def _check_training_losses(model: Any) -> None:
    """Losses must equal an MSE computed from the public ``integrate*``, and
    a split training step must only move trainable leaves, by their full
    gradient."""
    kw = _alpha_kw(model)
    ctx = _ctx(model)
    sibling = _members(model, 1)[0]
    ys = _integrate(sibling, **kw)

    loss, g_full = grad_loss(model, _TS, ys, _Y_MEAN, _Y_SCALE, ctx, args=_U, **_STEP)
    assert float(loss) > 0
    np.testing.assert_allclose(loss, _mse(ys, _integrate(model, **kw)), rtol=1e-5)

    ys_p = jnp.concatenate((_Y0[None], _integrate_protocol(sibling, _PROTOCOL, **kw)))
    loss_p, _ = proto_grad_loss(
        model,
        _Y0,
        _PROTO_TS,
        ys_p,
        _PROTOCOL,
        _Y_MEAN,
        _Y_SCALE,
        ctx,
        integrate_protocol_states,
        **_STEP,
    )
    pred_p = _integrate_protocol(model, _PROTOCOL, **kw)
    np.testing.assert_allclose(
        loss_p, _mse(ys_p, jnp.concatenate((_Y0[None], pred_p))), rtol=1e-5
    )

    # A split step must update trainable leaves by exactly the full model's
    # gradient (plain SGD) and leave frozen leaves untouched.
    lr = 1e-2
    spec = _filter_spec(model)
    trainable, frozen = eqx.partition(model, spec)
    optim = optax.sgd(lr)
    loss_split, new_trainable, _, _ = make_step_split(
        trainable=trainable,
        frozen=frozen,
        ts=_TS,
        ys=ys,
        y_mean=_Y_MEAN,
        y_scale=_Y_SCALE,
        opt_state=optim.init(eqx.filter(trainable, eqx.is_array)),
        optim=optim,
        ctx=ctx,
        args=_U,
        **_STEP,
    )
    np.testing.assert_allclose(loss_split, loss, rtol=1e-6)
    g_trainable, _ = eqx.partition(g_full, spec)
    expected = eqx.apply_updates(
        trainable, jax.tree.map(lambda g: -lr * g, g_trainable)
    )
    moved = 0.0
    for new, want, old in zip(
        jax.tree.leaves(new_trainable),
        jax.tree.leaves(expected),
        jax.tree.leaves(trainable),
        strict=True,
    ):
        np.testing.assert_allclose(new, want, rtol=1e-4, atol=1e-7)
        moved = max(moved, float(jnp.max(jnp.abs(new - old))))
    assert moved > 0
    _, frozen_after = eqx.partition(eqx.combine(new_trainable, frozen), spec)
    for a, b in zip(
        jax.tree.leaves(frozen_after), jax.tree.leaves(frozen), strict=True
    ):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a, b)


def _generic_models() -> list[tuple[str, Any, bool]]:
    """``(id, model, check_training)``: one model per class.

    The latent mapper only enters encode/decode, so one mapper per ANODE
    class suffices here, plus TrainableZ0 as the annealed (alpha) mapper
    with an extra ``z0`` leaf; the full mapper sweep is below. Training
    losses are model-agnostic code (each model's integration path is
    covered by the path-agreement check), and gradient programs are the
    costliest to compile, so they're checked on one latent model with an
    annealed mapper, which also covers threading ``alpha`` through ``ctx``.
    """
    return [
        ("Ode", _ode(), False),
        ("FluxOde", _flux_ode(), False),
        ("Node", _node(), False),
        ("FluxNode", _flux_node(), False),
        ("Ude", Ude(ode=_ode(), nn=_node(), op="+"), False),
        ("FluxUde", FluxUde(flux_ode=_flux_ode(), flux_nn=_flux_node(), op="+"), False),
        ("Anode", _anode(HardLatentMapper), False),
        ("FluxAnode", _flux_anode(HardLatentMapper), False),
        ("FluxAnode-TrainableZ0", _flux_anode(TrainableZ0LatentMapper), True),
    ]


_GENERIC = _generic_models()


@pytest.mark.parametrize(
    ("model", "check_training"),
    [(m, t) for _, m, t in _GENERIC],
    ids=[n for n, _, _ in _GENERIC],
)
def test_model_invariants(
    model: Any, *, check_training: bool, subtests: Any, tmp_path: Path
) -> None:
    with subtests.test("path agreement"):
        _check_path_agreement(model)
    with subtests.test("ensembles and io"):
        _check_ensembles_and_io(model, tmp_path)
    if check_training:
        with subtests.test("training losses"):
            _check_training_losses(model)


# ---------------------------------------------------------------------------
# Mechanistic and neural flux models: fluxes balance the state
# ---------------------------------------------------------------------------


def test_fluxode_invariants(subtests: Any) -> None:
    model = _flux_ode()
    with subtests.test("time course balance"):
        _assert_balanced(_simulate(model), model.nv)
    with subtests.test("time course balance with out_scale"):
        # out_scale is a trainable array field; the reported fluxes must
        # still be the ones that drive the trajectory.
        scaled = eqx.tree_at(lambda m: m.out_scale, model, jnp.array([2.0]))
        _assert_balanced(_simulate(scaled), scaled.nv)
    with subtests.test("protocol balance"):
        sim = model.simulate_protocol_time_course(
            _PROTO_TS, _Y0, _PROTOCOL, _MAX_STEPS, rtol=_RTOL
        )
        _assert_protocol_balanced(sim, model.nv)
    with subtests.test("steady state"):
        _assert_steady(
            model.simulate_to_steady_state(_Y0, _MAX_STEPS, args=_U), model.nv
        )
    with subtests.test("protocol from steady state is stationary"):
        _, y_ss = model.integrate_to_steady_state(_Y0, _MAX_STEPS, args=_U)
        ys = model.integrate_protocol_from_steady_state(
            _PROTO_TS, _Y0, _PROTOCOL_CONST, _MAX_STEPS
        )
        np.testing.assert_allclose(ys, jnp.broadcast_to(y_ss, ys.shape), atol=1e-3)


def test_fluxnode_invariants(subtests: Any) -> None:
    model = _flux_node()
    with subtests.test("time course balance"):
        _assert_balanced(_simulate(model), model.nv)
    with subtests.test("protocol balance"):
        sim = model.simulate_protocol_time_course(
            _PROTO_TS, _Y0, _PROTOCOL, _MAX_STEPS, rtol=_RTOL
        )
        _assert_protocol_balanced(sim, model.nv)
    with subtests.test("derived matches source FluxOde"):
        _check_derived(model)


def test_fluxude_additive_fluxes_balance_state() -> None:
    flux_ode = _flux_ode()
    model = FluxUde(flux_ode=flux_ode, flux_nn=_flux_node(), op="+")
    ys = _integrate(model)
    # nv is linear, so nv(v_ode) + nv(v_nn) == nv(v_ode + v_nn)
    fluxes = model.ode_fluxes(_TS, ys, _U) + model.node_fluxes(_TS, ys, _U)
    assert _balance_error(_TS, ys, fluxes, flux_ode.nv) < _BALANCE_TOL


# ---------------------------------------------------------------------------
# FluxAnode, every latent mapper: hidden state is never reset
# ---------------------------------------------------------------------------

_Y_PROBE = jnp.array([1.3, 0.4, 0.9])


def _check_derived(model: Any) -> None:
    """``derived`` reproduces the source FluxOde's, whose parameters stay
    trainable leaves of the derived model."""
    expected = _flux_ode().derived(0.0, _Y_PROBE, _U)
    np.testing.assert_allclose(expected, [0.4 * 5.0], rtol=1e-6)  # d = B * c
    np.testing.assert_allclose(model.derived(0.0, _Y_PROBE, _U), expected)

    @eqx.filter_grad
    def grad(m: Any) -> jax.Array:
        return jnp.sum(m.derived(0.0, _Y_PROBE, _U))

    # d(B * c)/dc = B; k1 does not enter d
    np.testing.assert_allclose(grad(model).pars, [0.0, 0.4], rtol=1e-6)


@pytest.mark.parametrize("mapper", ALL_MAPPERS, ids=lambda m: m.__name__)
def test_fluxanode_hidden_state_invariants(mapper: type, subtests: Any) -> None:
    model = _flux_anode(mapper)
    kw = _alpha_kw(model)
    n_obs = model.n_obs
    sim = _simulate(model, **kw)
    assert sim.latent_ys is not None
    latent = sim.latent_ys
    hidden = latent[:, n_obs:]
    assert float(jnp.max(jnp.abs(hidden - hidden[0]))) > 0.1, (
        "hidden state barely moves; a hidden-state reset would go unnoticed"
    )

    with subtests.test("time course balance (latent)"):
        _assert_balanced(sim, model.nv)
    if mapper in _SLICE_DECODING:
        with subtests.test("time course balance (decoded)"):
            err = _balance_error(sim.ts, sim.ys, _sim_fluxes(sim), model.nv)
            assert err < _BALANCE_TOL
    with subtests.test("re-encoded fluxes violate balance (negative control)"):
        args = jnp.broadcast_to(_U, (_TS.shape[0], 1))
        reencoded = jax.vmap(model.latent_mapper.encode)(sim.ys)
        buggy = jax.vmap(model.fluxes)(sim.ts, reencoded, args)
        assert _balance_error(sim.ts, latent, buggy, model.nv) > 10 * _BALANCE_TOL
    with subtests.test("protocol balance"):
        sim_p = model.simulate_protocol_time_course(
            _PROTO_TS, _Y0, _PROTOCOL, _MAX_STEPS, rtol=_RTOL, **kw
        )
        _assert_protocol_balanced(sim_p, model.nv)

    sim_ss = model.simulate_to_steady_state(_Y0, _MAX_STEPS, args=_U, **kw)
    assert sim_ss.latent_ys is not None
    z_ss = sim_ss.latent_ys[0]
    with subtests.test("steady state: fluxes and markov vanish"):
        _assert_steady(sim_ss, model.nv)
        markov = model.markov_scale * model.markov_nn(jnp.concat((z_ss, _U)))
        assert float(jnp.max(jnp.abs(markov))) < 1e-3
        np.testing.assert_allclose(z_ss[n_obs:], 0.5, atol=1e-3)
    with subtests.test("re-encoded steady state is not stationary (negative control)"):
        z_reencoded = model.latent_mapper.encode(sim_ss.ys[0])
        residual = model.nv(model.fluxes(0.0, z_reencoded, _U))
        assert float(jnp.max(jnp.abs(residual))) > 1e-2

    _check_protocol_from_steady_state(model, subtests)

    with subtests.test("derived matches source FluxOde"):
        _check_derived(model)
    with subtests.test("derived_simulation_fn sees the decoded trajectory"):
        got = derived_simulation_fn(model, _PROTO_TS, _Y0, _PROTOCOL, _ctx(model))
        ys = _integrate_protocol(model, _PROTOCOL, **kw)
        np.testing.assert_allclose(got[:, 0], ys[:, 1] * 5.0, rtol=1e-5)


def _check_protocol_from_steady_state(model: Any, subtests: Any) -> None:
    kw = _alpha_kw(model)
    _, y_ss = model.integrate_to_steady_state(_Y0, _MAX_STEPS, args=_U, **kw)
    with subtests.test("protocol from steady state is stationary"):
        ys = model.integrate_protocol_from_steady_state(
            _PROTO_TS, _Y0, _PROTOCOL_CONST, _MAX_STEPS, **kw
        )
        np.testing.assert_allclose(ys, jnp.broadcast_to(y_ss, ys.shape), atol=1e-3)
    with subtests.test(
        "protocol from re-encoded steady state drifts (negative control)"
    ):
        ys = model.integrate_protocol(
            _PROTO_TS, y_ss, _PROTOCOL_CONST, _MAX_STEPS, **kw
        )
        assert float(jnp.max(jnp.abs(ys - y_ss))) > 1e-2


@pytest.mark.parametrize(
    "mapper", [HardLatentMapper, TrainableZ0LatentMapper], ids=lambda m: m.__name__
)
def test_anode_protocol_from_steady_state(mapper: type, subtests: Any) -> None:
    _check_protocol_from_steady_state(_anode(mapper), subtests)


def test_fluxanode_without_pars_attribute_still_works() -> None:
    """Models pickled before ``pars`` existed have no such attribute; they
    must keep flattening to the same leaves and evaluating."""
    model = _flux_anode(HardLatentMapper)
    old = object.__new__(FluxAnode)
    for k, v in vars(model).items():
        if k != "pars":
            object.__setattr__(old, k, v)
    assert old.pars is None  # falls back to the class default
    # pars=None contributes no leaf, so the old pytree structure is unchanged
    assert len(jax.tree.leaves(old)) == len(jax.tree.leaves(model)) - 1
    np.testing.assert_allclose(_integrate(old), _integrate(model))


# ---------------------------------------------------------------------------
# Compilation: repeated identical calls must hit JAX's caches
# ---------------------------------------------------------------------------


def test_repeated_integration_does_not_recompile() -> None:
    """Every compiled executable stays cached for the process lifetime, so a
    call that recompiles each time (e.g. ``lax.cond`` over fresh lambdas
    outside ``jit``) leaks memory linearly in the number of calls."""
    n = [0]

    def listener(event: str, duration_secs: float, **kwargs: str | int) -> None:  # noqa: ARG001
        if event == "/jax/core/compile/backend_compile_duration":
            n[0] += 1

    model = _flux_anode(HardLatentMapper)
    calls = {
        "integrate": lambda: _integrate(model),
        "integrate_protocol": lambda: _integrate_protocol(model, _PROTOCOL),
        "integrate_to_steady_state": lambda: model.integrate_to_steady_state(
            _Y0, _MAX_STEPS, args=_U
        )[1],
    }
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        for name, call in calls.items():
            call().block_until_ready()
            before = n[0]
            for _ in range(3):
                call().block_until_ready()
            assert n[0] == before, f"{name} recompiles on every call"
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
