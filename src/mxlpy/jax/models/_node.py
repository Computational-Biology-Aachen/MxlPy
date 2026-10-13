# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Neural ODEs: :class:`Node` and :class:`FluxNode`.

Purely data-driven models without a mechanistic prior, each a single MLP
on the observed state (plus optional external ``args``):

* :class:`Node` -- ``dy/dt = out_scale * mlp(concat(y, args))``.
* :class:`FluxNode` -- the MLP predicts reaction fluxes,
  ``dy/dt = nv(flux_scale * mlp(concat(y, args)))``, so the learned
  dynamics respect a known stoichiometry. :meth:`FluxNode.from_flux_ode`
  takes ``nv``, names and readouts from a :class:`~._ode.FluxOde`.

The output scales default to 0.1 so an untrained network starts with small
dynamics. Being autonomous ODEs on the observed variables, these models
can't represent trajectories that cross in state space; the augmented
variants in :mod:`._aug_node` lift that restriction. Used on their own or
as the neural component of :class:`~._ude.Ude`/:class:`~._ude.FluxUde`.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from mxlpy.jax.simulation import Simulation

from ._base import (
    Base,
    FluxBase,
    Nv,
    ReadoutFn,
    _readouts_with_pars,
    _resolve_readouts,
)
from ._ode import FluxOde


class Node(Base[Simulation]):
    """Neural ODE: an MLP maps state (and args) to its derivative.

    Output is scaled by ``out_scale`` (default 0.1) to keep initial
    dynamics small and aid early training stability.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the state variables; their count is the MLP's state size.
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    out_scale : jax.Array or None
        Multiplier for the MLP output.  Defaults to 0.1.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`; ``readout_fn`` gets ``(t, y, args)``.
    """

    out_scale: jax.Array
    nn: eqx.nn.MLP
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> None:
        n_obs = len(variable_names)
        self.variable_names = variable_names
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=n_obs + n_args,
            out_size=n_obs,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def __call__(
        self,
        time: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """``out_scale * nn(concat(y, args))``."""
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, args)``."""
        return self.readout_scale * self.readout_fn(t, y, args)


class FluxNode(FluxBase):
    """Neural ODE with a flux-based formulation and stoichiometric map.

    An MLP predicts reaction fluxes; ``nv`` maps them to state derivatives.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the state variables (MLP state input).
    flux_names : tuple[str, ...]
        Names of the fluxes (MLP output).
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    flux_scale : jax.Array or None
        Multiplier for the MLP's fluxes.  Defaults to 0.1.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`.
    pars : jax.Array or None
        Parameters appended to ``args`` in :meth:`readouts` only (the MLP
        never sees them); :meth:`from_flux_ode` sets them to the source
        ``FluxOde``'s ``pars``, which its ``readout_fn`` expects.
    """

    flux_scale: jax.Array
    flux_nn: eqx.nn.MLP
    nv: Nv
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    pars: jax.Array | None
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(  # noqa: PLR0917
        self,
        variable_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        width: int,
        depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        pars: jax.Array | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.pars = pars
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.flux_nn = eqx.nn.MLP(
            in_size=len(variable_names) + n_args,
            out_size=len(flux_names),
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.nv = nv
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    @classmethod
    def from_flux_ode(
        cls,
        flux_ode: FluxOde,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> Self:
        """Construct a :class:`FluxNode` sharing ``flux_ode``'s names, ``nv`` and readouts."""
        return cls(
            variable_names=flux_ode.variable_names,
            flux_names=flux_ode.flux_names,
            width=width,
            depth=depth,
            key=key,
            nv=flux_ode.nv,
            n_args=n_args,
            flux_scale=flux_scale,
            readout_fn=flux_ode.readout_fn,
            readout_names=flux_ode.readout_names,
            readout_scale=readout_scale,
            pars=flux_ode.pars,
        )

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:  # noqa: ARG002
        """``flux_scale * flux_nn(concat(y, args))``."""
        return self.flux_scale * self.flux_nn(jnp.concat((y, args)))

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``nv(fluxes(t, y, args))``."""
        return self.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return _readouts_with_pars(self, t, y, args)
