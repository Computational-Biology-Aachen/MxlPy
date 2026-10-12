"""Simulation results for JAX models.

Returned by every model's ``simulate_*`` methods. All results are JAX pytrees
of arrays computed at simulation time, so they work inside ``jit``/``grad``/
``vmap`` (e.g. a loss penalising ``sim.fluxes``) and batch through
:func:`~mxlpy.jax.ensemble.batch_simulate`. The ``*_df`` properties give
named, time-indexed DataFrames of a single (unbatched) result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import numpy as np
import pandas as pd

__all__ = ["FluxSimulation", "Simulation"]


def _frame(
    values: jax.Array, ts: jax.Array, names: tuple[str, ...], success: jax.Array
) -> pd.DataFrame:
    """Time-indexed DataFrame of ``values``; NaN-filled if the solve failed."""
    data = np.asarray(values)
    if not bool(success):
        data = np.full(data.shape, np.nan)
    return pd.DataFrame(data, index=np.asarray(ts), columns=list(names))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["ts", "ys", "args", "readouts", "success", "_latent_ys"],
    meta_fields=["variable_names", "readout_names"],
)
@dataclass(frozen=True)
class Simulation:
    """Result of a JAX model simulation.

    Parameters
    ----------
    ts : jax.Array
        Time points, shape ``(T,)``.
    ys : jax.Array
        Observed variables, shape ``(T, n_obs)``.
    args : jax.Array
        External arguments active at every row, shape ``(T, n_args)``;
        excludes trainable ``pars``.
    readouts : jax.Array
        Readouts at every row, shape ``(T, n_readouts)``.
    success : jax.Array
        Whether every solve behind this result succeeded (boolean scalar).
        Solver failures raise unless ``simulate_*(throw=False)``; then, like
        a steady state not reached by ``t1``, they leave zero-filled arrays
        and ``success=False``, and the ``*_df`` properties return NaN.
    variable_names, readout_names : tuple[str, ...]
        Column names of ``ys`` and ``readouts``.
    _latent_ys : jax.Array
        The ODE-state trajectory, shape ``(T, n_state)``; differs from
        ``ys`` only for latent models. Private: for diagnostics, not a
        stable interface.
    """

    ts: jax.Array
    ys: jax.Array
    args: jax.Array
    readouts: jax.Array
    success: jax.Array
    variable_names: tuple[str, ...]
    readout_names: tuple[str, ...]
    _latent_ys: jax.Array

    @property
    def variables_df(self) -> pd.DataFrame:
        """Observed variables as a time-indexed DataFrame."""
        return _frame(self.ys, self.ts, self.variable_names, self.success)

    @property
    def readouts_df(self) -> pd.DataFrame:
        """Readouts as a time-indexed DataFrame."""
        return _frame(self.readouts, self.ts, self.readout_names, self.success)

    def get_combined(self) -> pd.DataFrame:
        """All named quantities as a single time-indexed DataFrame."""
        return pd.concat((self.variables_df, self.readouts_df), axis=1)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "ts",
        "ys",
        "args",
        "readouts",
        "success",
        "_latent_ys",
        "fluxes",
        "ode_fluxes",
        "nn_fluxes",
    ],
    meta_fields=["variable_names", "readout_names", "flux_names"],
)
@dataclass(frozen=True)
class FluxSimulation(Simulation):
    """Result of a flux-based JAX model simulation.

    Parameters
    ----------
    fluxes : jax.Array
        The fluxes driving the ODE at every row, evaluated on the
        ODE-state trajectory, shape ``(T, n_flux)``.
    flux_names : tuple[str, ...]
        Column names of ``fluxes``.
    ode_fluxes, nn_fluxes : jax.Array or None
        For :class:`~mxlpy.jax.models.FluxUde`: the mechanistic and neural
        fluxes that ``op`` combines into ``fluxes``; ``None`` otherwise.

    See :class:`Simulation` for the other parameters.
    """

    fluxes: jax.Array = field(kw_only=True)
    flux_names: tuple[str, ...] = field(kw_only=True)
    ode_fluxes: jax.Array | None = field(default=None, kw_only=True)
    nn_fluxes: jax.Array | None = field(default=None, kw_only=True)

    @property
    def fluxes_df(self) -> pd.DataFrame:
        """Fluxes as a time-indexed DataFrame."""
        return _frame(self.fluxes, self.ts, self.flux_names, self.success)

    def get_combined(self) -> pd.DataFrame:
        """All named quantities as a single time-indexed DataFrame."""
        return pd.concat((self.variables_df, self.fluxes_df, self.readouts_df), axis=1)
