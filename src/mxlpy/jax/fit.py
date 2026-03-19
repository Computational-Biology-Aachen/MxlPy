"""JAX-based fitting for Universal Differential Equations.

Provides fitting routines analogous to :mod:`mxlpy.fit` but for
:class:`~mxlpy.jax.UDE` models: gradient-based training via Optax and
Diffrax, fully differentiable through the ODE solver.

Examples
--------
>>> from mxlpy.jax import fit as fit_jax
>>> result = fit_jax.time_course(
...     ude, data,
...     minimizer=fit_jax.OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
... )
>>> result = fit_jax.protocol_time_course(
...     ude, protocol, data,
...     minimizer=fit_jax.OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
... )
>>> result = fit_jax.steady_state(
...     ude, data,
...     minimizer=fit_jax.OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
... )

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from diffrax import AbstractSolver, Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from mxlpy.jax._ude import JaxFit

__all__ = ["OptaxMinimizer", "protocol_time_course", "steady_state", "time_course"]

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.jax._ude import UDE


@dataclass
class OptaxMinimizer:
    """Configuration for an Optax gradient-based optimizer.

    Parameters
    ----------
    lr
        Learning rate passed to the optimizer factory.
    method
        Optax optimizer factory (e.g. ``optax.adam``, ``optax.sgd``).
    n_epochs
        Number of gradient-descent steps to perform.
    solver
        Diffrax ODE solver class or instance.  Defaults to ``Dopri5``.
    rtol
        Relative tolerance for the adaptive step-size controller.
    atol
        Absolute tolerance for the adaptive step-size controller.
    max_steps
        Maximum number of internal ODE solver steps per segment.

    Examples
    --------
    >>> import optax
    >>> minimizer = OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400)

    """

    lr: float
    method: Callable[..., Any]
    n_epochs: int
    solver: Any = field(default_factory=Dopri5)
    rtol: float = 1e-3
    atol: float = 1e-5
    max_steps: int = 4096


# ============================================================================ #
# Private helpers                                                               #
# ============================================================================ #


def _resolve_solver(minimizer: OptaxMinimizer) -> AbstractSolver:
    raw = minimizer.solver
    return raw() if isinstance(raw, type) else raw  # type: ignore[return-value]


def _normalise_timedelta_index(protocol: pd.DataFrame) -> pd.DataFrame:
    if isinstance(protocol.index, pd.TimedeltaIndex):
        return protocol
    msg = "Protocol index must be a pd.TimedeltaIndex (use pd.to_timedelta)."
    raise TypeError(msg)


def _run_training(
    ude: UDE,
    minimizer: OptaxMinimizer,
    compute_loss: Callable[[UDE], Any],
) -> JaxFit:
    """Generic Optax training loop.

    Parameters
    ----------
    ude
        Initial UDE (Equinox pytree).
    minimizer
        Optimizer configuration.
    compute_loss
        Differentiable loss function ``f(ude) -> scalar``.

    Returns
    -------
    JaxFit
        Trained UDE and per-epoch loss history.

    """

    @eqx.filter_jit
    def _make_step(ude_model: UDE, opt_state: Any) -> tuple[UDE, Any, Any]:
        loss, grads = eqx.filter_value_and_grad(compute_loss)(ude_model)
        updates, new_opt_state = optimizer.update(grads, opt_state, ude_model)
        return eqx.apply_updates(ude_model, updates), new_opt_state, loss

    optimizer = minimizer.method(minimizer.lr)
    opt_state = optimizer.init(eqx.filter(ude, eqx.is_array))

    losses: list[float] = []
    for _ in range(minimizer.n_epochs):
        ude, opt_state, loss = _make_step(ude, opt_state)
        losses.append(float(loss))

    return JaxFit(ude=ude, losses=losses)


def _build_protocol_segments(
    ude: UDE,
    protocol: pd.DataFrame,
    t_data: np.ndarray,
) -> list[dict[str, Any]]:
    """Parse a protocol into per-segment {t0, t1, args, t_eval} dicts."""
    base_args = ude.ode.get_args()
    segments: list[dict[str, Any]] = []
    t_cursor = 0.0

    for td, row in protocol.iterrows():
        t_end = float(td.total_seconds())  # type: ignore[union-attr]

        args = base_args
        for col, val in row.items():
            col_str = str(col)
            if col_str in ude.ode.parameter_names:
                idx = ude.ode.parameter_names.index(col_str)
                args = args.at[idx].set(float(val))  # noqa: PD008

        if not segments:
            t_pts = t_data[(t_data >= t_cursor) & (t_data <= t_end)]
        else:
            t_pts = t_data[(t_data > t_cursor) & (t_data <= t_end)]

        segments.append({"t0": t_cursor, "t1": t_end, "args": args, "t_eval": t_pts})
        t_cursor = t_end

    return segments


# ============================================================================ #
# Public fitting functions                                                      #
# ============================================================================ #


def time_course(
    ude: UDE,
    data: pd.DataFrame,
    *,
    minimizer: OptaxMinimizer,
) -> JaxFit:
    """Train a UDE on time-course data.

    Integrates the UDE over the full time span of the data with fixed
    parameters, minimising MSE against the target trajectory.

    Parameters
    ----------
    ude
        UDE instance to train.
    data
        Training data.  ``data.index`` must be numeric time points and
        ``data.columns`` must be a subset of ``ude.ode.variable_names``.
    minimizer
        Optimizer configuration.

    Returns
    -------
    JaxFit
        Trained UDE and per-epoch loss history.

    Examples
    --------
    >>> result = time_course(
    ...     ude, data,
    ...     minimizer=OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
    ... )

    """
    t_data_np = np.asarray(data.index, dtype=float)
    t_data = jnp.array(t_data_np)
    y0 = ude.ode.get_y0()
    base_args = ude.ode.get_args()
    target = jnp.array(data.values, dtype=float)
    var_indices = [ude.ode.variable_names.index(c) for c in data.columns]
    solver_instance = _resolve_solver(minimizer)
    step_ctrl = PIDController(rtol=minimizer.rtol, atol=minimizer.atol)

    def _simulate(ude_model: UDE) -> Any:
        sol = diffeqsolve(
            ODETerm(ude_model),
            solver_instance,
            t0=float(t_data_np[0]),
            t1=float(t_data_np[-1]),
            dt0=0.5,
            y0=y0,
            saveat=SaveAt(ts=t_data),
            stepsize_controller=step_ctrl,
            args=base_args,
            max_steps=minimizer.max_steps,
        )
        ys = sol.ys
        assert ys is not None  # noqa: S101
        return ys  # (n_time, n_vars)

    def _compute_loss(ude_model: UDE) -> Any:
        pred = _simulate(ude_model)
        pred_sel = jnp.stack([pred[:, i] for i in var_indices], axis=1)
        return jnp.mean((pred_sel - target) ** 2)

    return _run_training(ude, minimizer, _compute_loss)


def steady_state(
    ude: UDE,
    data: pd.Series,
    *,
    minimizer: OptaxMinimizer,
    t_steady: float = 1e6,
) -> JaxFit:
    """Train a UDE to match steady-state experimental data.

    Integrates the UDE to ``t_steady`` with fixed parameters and compares
    the final state against the target steady-state values.

    Parameters
    ----------
    ude
        UDE instance to train.
    data
        Target steady-state values.  Index must be a subset of
        ``ude.ode.variable_names``.
    minimizer
        Optimizer configuration.
    t_steady
        Time to integrate to before reading off the "steady state".
        Should be large enough that transients have died out.

    Returns
    -------
    JaxFit
        Trained UDE and per-epoch loss history.

    Examples
    --------
    >>> result = steady_state(
    ...     ude, ss_data,
    ...     minimizer=OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
    ... )

    """
    y0 = ude.ode.get_y0()
    base_args = ude.ode.get_args()
    var_indices = [ude.ode.variable_names.index(v) for v in data.index]
    target = jnp.array(data.values, dtype=float)
    solver_instance = _resolve_solver(minimizer)
    step_ctrl = PIDController(rtol=minimizer.rtol, atol=minimizer.atol)

    def _simulate(ude_model: UDE) -> Any:
        sol = diffeqsolve(
            ODETerm(ude_model),
            solver_instance,
            t0=0.0,
            t1=t_steady,
            dt0=0.5,
            y0=y0,
            saveat=SaveAt(t1=True),
            stepsize_controller=step_ctrl,
            args=base_args,
            max_steps=minimizer.max_steps,
        )
        ys = sol.ys
        assert ys is not None  # noqa: S101
        return ys[0]  # (n_vars,) — single final state

    def _compute_loss(ude_model: UDE) -> Any:
        pred = _simulate(ude_model)
        pred_sel = jnp.stack([pred[i] for i in var_indices])
        return jnp.mean((pred_sel - target) ** 2)

    return _run_training(ude, minimizer, _compute_loss)


def protocol_time_course(
    ude: UDE,
    protocol: pd.DataFrame,
    data: pd.DataFrame,
    *,
    minimizer: OptaxMinimizer,
) -> JaxFit:
    """Train a UDE on protocol time-course data.

    Integrates the UDE segment by segment following the protocol, computes
    MSE against the target data, and runs the Optax optimizer for
    ``minimizer.n_epochs`` steps.

    Parameters
    ----------
    ude
        UDE instance to train.  Must be a :class:`~mxlpy.jax.UDE` subclass
        with ``ode``, ``mlp``, and a ``__call__(t, y, args) -> dy`` method.
    protocol
        mxlpy protocol DataFrame with a ``pd.TimedeltaIndex``.  Each row
        specifies parameter values (column names matching model parameter
        names) valid until that time.
    data
        Training data.  ``data.index`` must be numeric time points (same
        units as the protocol) and ``data.columns`` must be a subset of
        ``ude.ode.variable_names``.
    minimizer
        Optimizer configuration.

    Returns
    -------
    JaxFit
        Trained UDE and per-epoch loss history.

    Examples
    --------
    >>> result = protocol_time_course(
    ...     ude, protocol, data,
    ...     minimizer=OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
    ... )

    """
    protocol = _normalise_timedelta_index(protocol)
    t_data = np.asarray(data.index, dtype=float)
    segments = _build_protocol_segments(ude, protocol, t_data)

    y0 = ude.ode.get_y0()
    target = jnp.array(data.values, dtype=float)
    var_indices = [ude.ode.variable_names.index(c) for c in data.columns]
    solver_instance = _resolve_solver(minimizer)
    step_ctrl = PIDController(rtol=minimizer.rtol, atol=minimizer.atol)

    def _simulate(ude_model: UDE) -> Any:
        y = y0
        all_ys: list[Any] = []
        for seg in segments:
            sol = diffeqsolve(
                ODETerm(ude_model),
                solver_instance,
                t0=seg["t0"],
                t1=seg["t1"],
                dt0=0.5,
                y0=y,
                saveat=SaveAt(ts=seg["t_eval"]),
                stepsize_controller=step_ctrl,
                args=seg["args"],
                max_steps=minimizer.max_steps,
            )
            ys = sol.ys
            assert ys is not None  # noqa: S101
            y = ys[-1]
            all_ys.append(ys)
        return jnp.concatenate(all_ys, axis=0)  # (n_time, n_vars)

    def _compute_loss(ude_model: UDE) -> Any:
        pred = _simulate(ude_model)
        pred_sel = jnp.stack([pred[:, i] for i in var_indices], axis=1)
        return jnp.mean((pred_sel - target) ** 2)

    return _run_training(ude, minimizer, _compute_loss)
