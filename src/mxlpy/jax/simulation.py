"""Named, DataFrame-based simulation results for JAX models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mxlpy.jax.models import FluxOde

__all__ = ["FluxOdeSimulation"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["model", "ts", "ys", "args"],
    meta_fields=["variable_names", "flux_names"],
)
@dataclass(slots=False)  # cached_property needs an instance __dict__
class FluxOdeSimulation:
    """Named simulation result for a :class:`~mxlpy.jax.models.FluxOde`.

    Bridges the raw ``jax.Array`` output of ``FluxOde.integrate*`` back to
    named, pandas-based results, mirroring :class:`mxlpy.simulation.Simulation`
    for the (much narrower) set of things a frozen JAX model can report.

    Registered as a JAX pytree (``model``/``ts``/``ys``/``args`` as dynamic
    data, ``variable_names``/``flux_names`` as static metadata) so it
    composes with JAX transformations -- in particular
    :func:`~mxlpy.jax.ensemble.batch_simulate`, which would otherwise treat
    an unregistered dataclass as a single opaque leaf and silently return
    leaked tracers instead of a properly batched result.

    Parameters
    ----------
    model : FluxOde
        Model the trajectory was produced from; used to recompute fluxes.
    ts : jax.Array
        Time points, shape ``(T,)``.
    ys : jax.Array
        State trajectory, shape ``(T, n_state)``.
    args : jax.Array
        Per-row external arguments (excluding trainable ``pars``, which are
        added back in when computing fluxes), shape ``(T, n_args)``.
    variable_names : tuple[str, ...]
        Column names for :attr:`variables`, in ``ys``'s column order.
    flux_names : tuple[str, ...]
        Column names for :attr:`fluxes`, in ``model.fluxes``'s output order.
    """

    model: FluxOde
    ts: jax.Array
    ys: jax.Array
    args: jax.Array
    variable_names: tuple[str, ...]
    flux_names: tuple[str, ...]

    @cached_property
    def variables(self) -> pd.DataFrame:
        """State trajectory as a time-indexed DataFrame."""
        return pd.DataFrame(
            np.asarray(self.ys),
            index=np.asarray(self.ts),
            columns=list(self.variable_names),
        )

    @cached_property
    def fluxes(self) -> pd.DataFrame:
        """Flux trajectory as a time-indexed DataFrame.

        Recomputed from ``model.fluxes``, since ``integrate*`` only ever
        saves state, not fluxes.
        """
        flux_ys = jax.vmap(
            lambda t, y, a: self.model.fluxes(t, y, jnp.concat((a, self.model.pars)))
        )(self.ts, self.ys, self.args)
        return pd.DataFrame(
            np.asarray(flux_ys),
            index=np.asarray(self.ts),
            columns=list(self.flux_names),
        )

    def get_combined(self) -> pd.DataFrame:
        """Variables and fluxes as a single time-indexed DataFrame."""
        return pd.concat((self.variables, self.fluxes), axis=1)
