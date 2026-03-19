"""JAX-based fitting for Universal Differential Equations.

Provides :func:`protocol_time_course` and :class:`OptaxMinimizer` for
training :class:`~mxlpy.jax.UDE` models against experimental data using
segment-by-segment Diffrax integration and Optax gradient-based optimizers.

Examples
--------
>>> from mxlpy.jax import fit as fit_jax
>>> result = fit_jax.protocol_time_course(
...     ude,
...     protocol,
...     data,
...     minimizer=fit_jax.OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
... )
>>> trained_ude = result.ude

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

__all__ = ["OptaxMinimizer", "protocol_time_course"]

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
        Diffrax ODE solver class to use during integration.
        Defaults to ``Dopri5()``.
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


def _normalise_timedelta_index(protocol: pd.DataFrame) -> pd.DataFrame:
    """Ensure protocol index is a TimedeltaIndex."""
    if isinstance(protocol.index, pd.TimedeltaIndex):
        return protocol
    msg = "Protocol index must be a pd.TimedeltaIndex (use pd.to_timedelta)."
    raise TypeError(msg)


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
        Contains the trained UDE and the per-epoch loss history.

    Examples
    --------
    >>> result = protocol_time_course(
    ...     ude, protocol, data,
    ...     minimizer=OptaxMinimizer(lr=3e-3, method=optax.adam, n_epochs=400),
    ... )
    >>> trained = result.ude

    """
    protocol = _normalise_timedelta_index(protocol)

    # ------------------------------------------------------------------ #
    # 1. Build segment descriptors                                         #
    # ------------------------------------------------------------------ #
    t_data = np.asarray(data.index, dtype=float)
    base_args = ude.ode.get_args()

    segments: list[dict[str, Any]] = []
    t_cursor = 0.0
    for td, row in protocol.iterrows():
        t_end = float(td.total_seconds())  # type: ignore[union-attr]

        # Override parameters that appear in this protocol row
        args = base_args
        for col, val in row.items():
            col_str = str(col)
            if col_str in ude.ode.parameter_names:
                idx = ude.ode.parameter_names.index(col_str)
                args = args.at[idx].set(float(val))  # noqa: PD008

        # Time points for this segment (first segment includes t=t_cursor)
        if not segments:
            t_pts = t_data[(t_data >= t_cursor) & (t_data <= t_end)]
        else:
            t_pts = t_data[(t_data > t_cursor) & (t_data <= t_end)]

        segments.append(
            {
                "t0": t_cursor,
                "t1": t_end,
                "args": args,  # JAX array (static per segment)
                "t_eval": t_pts,  # NumPy array → concrete in JIT
            }
        )
        t_cursor = t_end

    # ------------------------------------------------------------------ #
    # 2. Pre-compute static quantities                                     #
    # ------------------------------------------------------------------ #
    y0 = ude.ode.get_y0()
    target = jnp.array(data.values, dtype=float)  # (n_time, n_cols)
    var_indices = [ude.ode.variable_names.index(c) for c in data.columns]
    raw_solver = minimizer.solver
    solver_instance: AbstractSolver = (
        raw_solver() if isinstance(raw_solver, type) else raw_solver
    )  # type: ignore[assignment]
    step_ctrl = PIDController(rtol=minimizer.rtol, atol=minimizer.atol)

    # ------------------------------------------------------------------ #
    # 3. Define differentiable simulation                                  #
    # ------------------------------------------------------------------ #
    def _simulate(ude_model: UDE) -> Any:
        """Integrate UDE across all protocol segments."""
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

    @eqx.filter_jit
    def _make_step(ude_model: UDE, opt_state: Any) -> tuple[UDE, Any, Any]:
        loss, grads = eqx.filter_value_and_grad(_compute_loss)(ude_model)
        updates, new_opt_state = optimizer.update(grads, opt_state, ude_model)
        return eqx.apply_updates(ude_model, updates), new_opt_state, loss

    # ------------------------------------------------------------------ #
    # 4. Training loop                                                     #
    # ------------------------------------------------------------------ #
    optimizer = minimizer.method(minimizer.lr)
    opt_state = optimizer.init(eqx.filter(ude, eqx.is_array))

    losses: list[float] = []
    for _ in range(minimizer.n_epochs):
        ude, opt_state, loss = _make_step(ude, opt_state)
        losses.append(float(loss))

    return JaxFit(ude=ude, losses=losses)
