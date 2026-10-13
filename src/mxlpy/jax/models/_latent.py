# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Latent mappers between observation space and an augmented latent space.

Latent models (:mod:`._aug_node`, :mod:`._aug_ude`) integrate a state
``z`` of ``n_obs + n_hidden`` dimensions. A :class:`LatentMapper` encodes
the initial observation ``y0`` into ``z0`` and decodes latent trajectories
back to observations; models take a :data:`LatentMapperFn` factory
``(n_obs, n_hidden, key) -> LatentMapper`` and construct it themselves.

================================  ====================  ===============================================
Mapper                            Encode ``y0 -> z0``   Decode ``z -> y``
================================  ====================  ===============================================
:class:`SoftLatentMapper`         ``(y0, linear(y0))``  ``linear(z)``
:class:`HardLatentMapper`         ``(y0, linear(y0))``  ``z[:n_obs]``
:class:`FullLatentMapper`         ``linear(y0)``        ``linear(z)``
:class:`MixedLatentMapper`        ``(y0, linear(y0))``  ``(1 - alpha) * z[:n_obs] + alpha * linear(z)``
:class:`TrainableZ0LatentMapper`  ``(y0, z0_learned)``  as :class:`MixedLatentMapper`
================================  ====================  ===============================================

Every mapper except :class:`FullLatentMapper` keeps ``y0`` verbatim in
``z0[:n_obs]``. The two annealed mappers require ``alpha``, threaded
through the models' ``integrate*``/``simulate_*`` methods (and
:class:`~mxlpy.jax.train.IntegrationSettings`); all others ignore it.
"""

from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


class LatentMapper(Protocol):
    """Protocol for encode/decode mappings between observation and latent space."""

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None: ...

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode an observation vector into the latent space."""
        ...

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories back to observation space.

        ``alpha`` is ignored by every implementation except
        :class:`MixedLatentMapper`/:class:`TrainableZ0LatentMapper`, whose
        decode is an annealed hard/learned blend; see their docstrings. Kept
        on the shared Protocol (rather than only on those two classes) so
        callers that thread an annealing schedule through
        :meth:`Anode.integrate`/:meth:`FluxAnode.integrate` don't need to
        know which concrete mapper is in use.
        """
        ...


type LatentMapperFn = Callable[[int, int, PRNGKeyArray], LatentMapper]


class SoftLatentMapper(eqx.Module):
    """Maps observed states to and from a soft latent representation.

    The encoder appends learned hidden features to the raw observation;
    the decoder maps the full latent vector back to observation space.

    z0 <- concat((y0, linear(y0)))
    ys <- linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories to observation space.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Decoded observations, shape ``(T, n_obs)``.
        """
        return jax.vmap(self.decoder)(zs)


class FullLatentMapper(eqx.Module):
    """Maps observed states to and from a fully learned latent representation.

    No part of the raw observation is preserved verbatim; both encoder
    and decoder are learned linear maps.

    z0 <- linear(y0)
    ys <- linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional latent dimensions (n_latent = n_obs + n_hidden).
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_latent,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space via a learned linear map.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return self.encoder(y0)

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories to observation space.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Decoded observations, shape ``(T, n_obs)``.
        """
        return jax.vmap(self.decoder)(zs)


class HardLatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with a hard (slice) decoder.

    The encoder appends learned features; the decoder is a fixed slice
    that returns the first ``n_obs`` latent dimensions without learning.

    z0 <- concat((y0, linear(y0)))
    ys <- zs[:, :n_obs]

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories by slicing the first ``n_obs`` dimensions.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Observed states, shape ``(T, n_obs)``.
        """
        return zs[:, : self.n_obs]


def _annealed_decode(
    zs: jax.Array,
    decoder: eqx.nn.Linear,
    n_obs: int,
    alpha: float | None,
) -> jax.Array:
    """Shared decode body for :class:`MixedLatentMapper`/:class:`TrainableZ0LatentMapper`.

    ``(1 - alpha) * hard + alpha * learned``, where ``hard`` is a plain slice
    of the first ``n_obs`` latent dimensions and ``learned`` is
    ``decoder`` applied to the full latent vector.

    Parameters
    ----------
    zs : jax.Array
        Latent trajectories, shape ``(T, n_latent)``.
    decoder : eqx.nn.Linear
        Learned latent-to-observation map.
    n_obs : int
        Number of observed state dimensions.
    alpha : float or None
        Mixing coefficient. ``0`` returns pure hard (slice) decoding; ``1``
        returns pure learned decoding. Required (not actually optional)
        despite the ``| None`` -- kept nullable only so the signature
        matches :class:`LatentMapper`'s shared Protocol, which defaults
        ``alpha`` to ``None`` for mappers that ignore it entirely; a caller
        using an annealed mapper without threading a real ``alpha`` through
        is a bug, not a case to silently paper over with a guessed default.

    Returns
    -------
    jax.Array
        Decoded observations, shape ``(T, n_obs)``.
    """
    if alpha is None:
        msg = (
            "alpha is required for an annealed hard/learned decode "
            "(MixedLatentMapper/TrainableZ0LatentMapper) -- thread it "
            "through the model's integrate*(alpha=...) "
            "parameter."
        )
        raise ValueError(msg)
    hard = zs[:, :n_obs]
    learned = jax.vmap(decoder)(zs)
    return (1 - alpha) * hard + alpha * learned


class MixedLatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with an annealed decoder.

    The decoder interpolates between a hard projection and a learned map
    via ``alpha``; annealing ``alpha`` toward zero recovers the hard decoder.

    z0 <- concat((y0, linear(y0)))
    ys <- (1 - alpha) * zs[:, :n_obs] + alpha * linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories using an annealed mixture of hard and learned maps.

        See :func:`_annealed_decode`.
        """
        return _annealed_decode(zs, self.decoder, self.n_obs, alpha)


class TrainableZ0LatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with a free, trainable hidden init.

    Unlike every other :class:`LatentMapper`, the hidden dimensions of the
    initial latent state are not a function of ``y0`` at all: they're a
    single trainable vector shared across every trajectory the model is
    ever trained or evaluated on. The observed dimensions are still carried
    through ``y0`` verbatim, and the decoder is the same annealed
    hard/learned blend as :class:`MixedLatentMapper` (see
    :func:`_annealed_decode`) -- the two share that decode logic via a
    module-level helper rather than inheritance, since inheriting from
    MixedLatentMapper would carry along its (here unused) ``encoder`` field.

    z0 <- concat((y0, self.z0))
    ys <- (1 - alpha) * zs[:, :n_obs] + alpha * linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional trainable latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    z0_minval : float
        Lower bound for ``z0``'s uniform initialisation. Defaults to 0.0.
    z0_maxval : float
        Upper bound for ``z0``'s uniform initialisation. Defaults to 1.0.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    z0: jax.Array
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
        z0_minval: float = 0.0,
        z0_maxval: float = 1.0,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        z0_key, decoder_key = jax.random.split(key)
        self.z0 = jax.random.uniform(
            key=z0_key,
            shape=(n_hidden,),
            minval=z0_minval,
            maxval=z0_maxval,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=decoder_key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space, appending the trainable ``z0``.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.z0))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories using an annealed mixture of hard and learned maps.

        See :func:`_annealed_decode`.
        """
        return _annealed_decode(zs, self.decoder, self.n_obs, alpha)
