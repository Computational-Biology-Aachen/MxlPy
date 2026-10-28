# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""JAX model classes for mechanistic and neural ODE components.

Conventions shared by every model here:

* Every model integrates in its own *ODE state*. For most models that's
  simply the observed variables; :class:`Anode`/:class:`FluxAnode` and
  :class:`AugUde`/:class:`FluxAugUde` integrate in a latent space of
  ``n_obs + n_hidden`` dimensions instead, mapped to and from observations
  by a :class:`LatentMapper` (see :meth:`Base._encode`/:meth:`Base._decode`). ``integrate*`` always takes and returns
  observations; ``alpha`` is forwarded to the latent decode and ignored by
  models without a latent space.
* ``__call__`` (the RHS) and ``fluxes`` take the ODE state; ``readouts``
  take observations.
* ``variable_names`` (and ``flux_names`` for flux models) define the model's
  dimensions. ``readout_names`` label :meth:`Base.readouts`'s output.
* Solver options after the leading data arguments are keyword-only.
"""

from ._aug_node import Anode, FluxAnode
from ._aug_ude import AugUde, FluxAugUde
from ._base import (
    DEFAULT_ADJOINT,
    Base,
    CombineOp,
    FluxBase,
    JaxModel,
    Method,
    Nv,
    ReadoutFn,
    Rhs,
    boundaries_to_ts,
)
from ._latent import (
    FullLatentMapper,
    HardLatentMapper,
    LatentMapper,
    LatentMapperFn,
    MixedLatentMapper,
    SoftLatentMapper,
    TrainableZ0LatentMapper,
)
from ._node import FluxNode, Node
from ._ode import FluxOde, Ode
from ._ude import FluxUde, Ude

__all__ = [
    "DEFAULT_ADJOINT",
    "Anode",
    "AugUde",
    "Base",
    "CombineOp",
    "FluxAnode",
    "FluxAugUde",
    "FluxBase",
    "FluxNode",
    "FluxOde",
    "FluxUde",
    "FullLatentMapper",
    "HardLatentMapper",
    "JaxModel",
    "LatentMapper",
    "LatentMapperFn",
    "Method",
    "MixedLatentMapper",
    "Node",
    "Nv",
    "Ode",
    "ReadoutFn",
    "Rhs",
    "SoftLatentMapper",
    "TrainableZ0LatentMapper",
    "Ude",
    "boundaries_to_ts",
]
