# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Augmented universal differential equations: :class:`AugUde` and :class:`FluxAugUde`.

UDEs whose neural component is augmented (:class:`~._aug_node.Anode`/
:class:`~._aug_node.FluxAnode`) and therefore integrates in a latent space
``z = (y, h)`` of ``n_obs + n_hidden`` dimensions:

* :class:`AugUde` -- with ``d = anode(t, z, args)``:
  ``dy/dt = op(ode(t, y, args), d[:n_obs])`` and ``dh/dt = d[n_obs:]``.
* :class:`FluxAugUde` --
  ``v = op(flux_ode.fluxes(t, y, args), flux_nn.fluxes(t, z, args))``,
  ``dy/dt = nv(v)`` and ``dh/dt`` from ``flux_nn``'s Markov MLP.

The mechanistic component only ever sees the observed slice
``y = z[:n_obs]``, so the hidden dimensions act on the observed dynamics
exclusively through the neural correction -- matching augmented NODEs and
physics-plus-correction hybrids, which keep latent states out of the
mechanistic term. Unobserved variables with known mechanisms belong in the
mechanistic model itself instead.

This requires a latent mapper that keeps the observation in ``z[:n_obs]``:
:class:`~._latent.FullLatentMapper` is rejected,
:class:`~._latent.HardLatentMapper` is the interpretable choice (decoded
observations equal the mechanistic state), and the annealed mappers
converge to it as ``alpha -> 0``.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from mxlpy.jax.simulation import Simulation

from ._aug_node import Anode, FluxAnode
from ._base import (
    Base,
    CombineOp,
    FluxBase,
)
from ._latent import FullLatentMapper
from ._ode import FluxOde, Ode
from ._ude import _resolve_op


def _check_latent_mapper(nn: Anode | FluxAnode) -> None:
    """Reject mappers whose latent state doesn't carry the observations in ``z[:n_obs]``."""
    if isinstance(nn.latent_mapper, FullLatentMapper):
        msg = (
            "FullLatentMapper is not supported: the mechanistic component acts "
            "on z[:n_obs], which it doesn't keep as the observation"
        )
        raise TypeError(msg)


class AugUde(Base[Simulation]):
    """Augmented Universal Differential Equation: an ODE combined with an :class:`Anode`.

    On the latent state ``z = concat(y, h)``, with ``d = nn(t, z, args)``::

        dy/dt = op(ode(t, y, args), d[:n_obs])
        dh/dt = d[n_obs:]

    Encoding and decoding go through ``nn.latent_mapper``.
    :class:`HardLatentMapper` keeps the decoded observations equal to the
    state the mechanistic component sees; :class:`MixedLatentMapper`/
    :class:`TrainableZ0LatentMapper` anneal towards it.
    :class:`FullLatentMapper` is rejected.

    Parameters
    ----------
    ode : Ode
        Mechanistic component, acting on the observed slice.
    nn : Anode
        Neural component; must have the same ``variable_names``.
    op : CombineOp or Callable
        See :class:`Ude`.
    """

    ode: Ode
    nn: Anode
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        ode: Ode,
        nn: Anode,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        if nn.variable_names != ode.variable_names:
            msg = (
                f"nn.variable_names {nn.variable_names} differ from "
                f"ode.variable_names {ode.variable_names}"
            )
            raise ValueError(msg)
        _check_latent_mapper(nn)
        self.ode = ode
        self.nn = nn
        self.op = _resolve_op(op)
        self.variable_names = ode.variable_names
        self.readout_names = ode.readout_names

    def _encode(self, y0: jax.Array) -> jax.Array:
        return self.nn._encode(y0)  # noqa: SLF001

    def _decode(self, zs: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.nn._decode(zs, alpha=alpha)  # noqa: SLF001

    def _decode_point(self, z: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.nn._decode_point(z, alpha=alpha)  # noqa: SLF001

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``concat(op(ode(t, z[:n_obs], args), d[:n_obs]), d[n_obs:])``."""
        n = self.nn.n_obs
        d = self.nn(time, y, args)
        return jnp.concat((self.op(self.ode(time, y[:n], args), d[:n]), d[n:]))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts on observations ``y``."""
        return self.ode.readouts(t, y, args)


class FluxAugUde(FluxBase):
    """Flux-based Augmented Universal Differential Equation.

    On the latent state ``z = concat(y, h)``::

        fluxes = op(flux_ode.fluxes(t, y, args), flux_nn.fluxes(t, z, args))
        dy/dt  = flux_ode.nv(fluxes)
        dh/dt  = flux_nn.markov_scale * flux_nn.markov_nn(concat(z, args))

    See :class:`AugUde` for the latent mapper.

    Parameters
    ----------
    flux_ode : FluxOde
        Mechanistic component, acting on the observed slice.
    flux_nn : FluxAnode
        Neural component; must have the same ``variable_names`` and
        ``flux_names`` (e.g. built via :meth:`FluxAnode.from_flux_ode`).
    op : CombineOp or Callable
        See :class:`Ude`.
    """

    flux_ode: FluxOde
    flux_nn: FluxAnode
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux_ode: FluxOde,
        flux_nn: FluxAnode,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        for attr in ("variable_names", "flux_names"):
            if getattr(flux_nn, attr) != getattr(flux_ode, attr):
                msg = (
                    f"flux_nn.{attr} {getattr(flux_nn, attr)} differ from "
                    f"flux_ode.{attr} {getattr(flux_ode, attr)}"
                )
                raise ValueError(msg)
        _check_latent_mapper(flux_nn)
        self.flux_ode = flux_ode
        self.flux_nn = flux_nn
        self.op = _resolve_op(op)
        self.variable_names = flux_ode.variable_names
        self.flux_names = flux_ode.flux_names
        self.readout_names = flux_ode.readout_names

    def _encode(self, y0: jax.Array) -> jax.Array:
        return self.flux_nn._encode(y0)  # noqa: SLF001

    def _decode(self, zs: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.flux_nn._decode(zs, alpha=alpha)  # noqa: SLF001

    def _decode_point(self, z: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.flux_nn._decode_point(z, alpha=alpha)  # noqa: SLF001

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``op(flux_ode.fluxes(t, z[:n_obs], args), flux_nn.fluxes(t, z, args))``."""
        return self.op(
            self.flux_ode.fluxes(t, y[: self.flux_nn.n_obs], args),
            self.flux_nn.fluxes(t, y, args),
        )

    def _flux_parts(
        self, ts: jax.Array, zs: jax.Array, args: jax.Array
    ) -> dict[str, jax.Array]:
        """Report the mechanistic and neural fluxes that ``op`` combines."""
        return {
            "ode_fluxes": jax.vmap(self.flux_ode.fluxes)(
                ts, zs[:, : self.flux_nn.n_obs], args
            ),
            "nn_fluxes": jax.vmap(self.flux_nn.fluxes)(ts, zs, args),
        }

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``concat(flux_ode.nv(fluxes(t, z, args)), flux_nn._markov(t, z, args))``."""
        return jnp.concat(
            (
                self.flux_ode.nv(self.fluxes(time, y, args)),
                self.flux_nn._markov(time, y, args),  # noqa: SLF001
            )
        )

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts on observations ``y``."""
        return self.flux_ode.readouts(t, y, args)
