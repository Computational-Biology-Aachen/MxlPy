# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Augmented neural ODEs: :class:`Anode` and :class:`FluxAnode`.

Augmented neural ODEs (Dupont et al., 2019) integrate in a latent space
``z = (y, h)`` of ``n_obs + n_hidden`` dimensions instead of the observed
state alone, so trajectories that would cross in observation space can be
separated along the hidden dimensions ``h``. A
:class:`~._latent.LatentMapper` encodes ``y0`` into ``z0`` and decodes the
latent trajectory; ``integrate*``/``simulate_*`` take and return
observations, the latent state stays internal.

* :class:`Anode` -- ``dz/dt = out_scale * mlp(concat(z, args))``; defaults
  to :class:`~._latent.SoftLatentMapper`.
* :class:`FluxAnode` -- a flux MLP drives the observed dimensions through
  ``nv`` and a separate Markov MLP drives the hidden ones, which ``nv``
  doesn't cover: ``dz/dt = concat(nv(fluxes(z)), markov(z))``; defaults to
  :class:`~._latent.HardLatentMapper`. :meth:`FluxAnode.from_flux_ode`
  takes ``nv``, names and readouts from a :class:`~._ode.FluxOde`.

Readouts are evaluated on the decoded observations. Both serve as the
neural component of the augmented UDEs in :mod:`._aug_ude`.
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
from ._latent import (
    HardLatentMapper,
    LatentMapper,
    LatentMapperFn,
    SoftLatentMapper,
)
from ._ode import FluxOde


class _LatentMixin:
    """Encode/decode through ``self.latent_mapper``; shared by the latent models."""

    latent_mapper: LatentMapper

    def _encode(self, y0: jax.Array) -> jax.Array:
        return self.latent_mapper.encode(y0)

    def _decode(self, zs: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.latent_mapper.decode(zs, alpha=alpha)

    def _decode_point(self, z: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.latent_mapper.decode(z[None, :], alpha=alpha)[0]


class Anode(_LatentMixin, Base[Simulation]):
    """Augmented Neural ODE: integrates in latent space, decodes to observations.

    The initial observation is encoded into an ``n_obs + n_hidden``
    dimensional latent space via ``latent_mapper``; the ODE runs there and
    the trajectory is decoded back to observation space.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the observed variables.
    n_hidden : int
        Number of additional latent dimensions.
    width : int
        MLP hidden layer width.
    depth : int
        Number of MLP hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    out_scale : jax.Array or None
        Multiplier for the MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.  Defaults to
        :class:`SoftLatentMapper`.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`; ``readout_fn`` gets decoded observations.
    """

    n_obs: int
    n_args: int
    out_scale: jax.Array
    nn: eqx.nn.MLP
    latent_mapper: LatentMapper
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        n_hidden: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.n_obs = len(variable_names)
        self.n_args = n_args
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=self.n_obs + n_hidden + n_args,
            out_size=self.n_obs + n_hidden,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        mapper = SoftLatentMapper if latent_mapper is None else latent_mapper
        self.latent_mapper = mapper(self.n_obs, n_hidden, key)
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
        """``out_scale * nn(concat(z, args))`` on the latent state ``z``."""
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, args)`` on observations ``y``."""
        return self.readout_scale * self.readout_fn(t, y, args)


class FluxAnode(_LatentMixin, FluxBase):
    """Augmented flux Neural ODE with a Markov term for the hidden dimensions.

    Integrates in latent space using two MLPs: ``flux_nn`` predicts the
    reaction fluxes driving the observed dimensions through ``nv``,
    ``markov_nn`` the dynamics of the hidden dimensions.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the observed variables.
    flux_names : tuple[str, ...]
        Names of the fluxes (``flux_nn``'s output).
    n_hidden : int
        Number of additional latent dimensions.
    flux_width, flux_depth : int
        Hidden layer width and count of the flux MLP.
    markov_width, markov_depth : int
        Hidden layer width and count of the Markov MLP.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    n_args : int
        Number of external arguments both MLPs additionally consume.
    flux_scale : jax.Array or None
        Multiplier for the fluxes.  Defaults to 0.1.
    markov_scale : jax.Array or None
        Multiplier for the Markov MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.  Defaults to
        :class:`HardLatentMapper`.
    readout_fn, readout_names, readout_scale, pars
        See :class:`FluxNode`; ``readout_fn`` gets decoded observations.
    """

    n_obs: int
    n_args: int
    nv: Nv
    flux_nn: eqx.nn.MLP
    markov_nn: eqx.nn.MLP
    flux_scale: jax.Array
    markov_scale: jax.Array
    latent_mapper: LatentMapper
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
        n_hidden: int,
        flux_width: int,
        flux_depth: int,
        markov_width: int,
        markov_depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        pars: jax.Array | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.n_obs = len(variable_names)
        self.n_args = n_args
        self.nv = nv
        self.pars = pars
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.markov_scale = jnp.array([0.1]) if markov_scale is None else markov_scale
        n_latent = self.n_obs + n_hidden
        self.flux_nn = eqx.nn.MLP(
            in_size=n_latent + n_args,
            out_size=len(flux_names),
            width_size=flux_width,
            depth=flux_depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.markov_nn = eqx.nn.MLP(
            in_size=n_latent + n_args,
            out_size=n_hidden,
            width_size=markov_width,
            depth=markov_depth,
            activation=jax.nn.softplus,
            key=key,
        )
        mapper = HardLatentMapper if latent_mapper is None else latent_mapper
        self.latent_mapper = mapper(self.n_obs, n_hidden, key)
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``concat(nv(fluxes), markov_scale * markov_nn(concat(z, args)))``."""
        return jnp.concat(
            (
                self.nv(self.fluxes(time, y, args)),
                self._markov(time, y, args),
            )
        )

    def _markov(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:  # noqa: ARG002
        """``markov_scale * markov_nn(concat(z, args))``: the hidden dimensions' dynamics."""
        return self.markov_scale * self.markov_nn(jnp.concat((y, args)))

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:  # noqa: ARG002
        """``flux_scale * flux_nn(concat(z, args))`` on the latent state ``z``."""
        return self.flux_scale * self.flux_nn(jnp.concat((y, args)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))`` on observations ``y``."""
        return _readouts_with_pars(self, t, y, args)

    @classmethod
    def from_flux_ode(
        cls,
        flux_ode: FluxOde,
        n_hidden: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        flux_width: int | None = None,
        flux_depth: int = 2,
        markov_width: int | None = None,
        markov_depth: int = 2,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        readout_scale: jax.Array | float | None = None,
        latent_mapper: LatentMapperFn | None = None,
    ) -> Self:
        """Construct a :class:`FluxAnode` sharing ``flux_ode``'s names, ``nv`` and readouts.

        ``flux_width`` defaults to ``n_flux + n_args``, ``markov_width`` to
        ``n_obs + n_hidden + n_args``.
        """
        n_obs, n_flux = len(flux_ode.variable_names), len(flux_ode.flux_names)
        return cls(
            variable_names=flux_ode.variable_names,
            flux_names=flux_ode.flux_names,
            n_hidden=n_hidden,
            flux_width=n_flux + n_args if flux_width is None else flux_width,
            flux_depth=flux_depth,
            markov_width=(
                n_obs + n_hidden + n_args if markov_width is None else markov_width
            ),
            markov_depth=markov_depth,
            key=key,
            nv=flux_ode.nv,
            n_args=n_args,
            flux_scale=flux_scale,
            markov_scale=markov_scale,
            latent_mapper=latent_mapper,
            readout_fn=flux_ode.readout_fn,
            readout_names=flux_ode.readout_names,
            readout_scale=readout_scale,
            pars=flux_ode.pars,
        )
