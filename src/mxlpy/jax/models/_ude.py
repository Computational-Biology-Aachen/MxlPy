# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Universal differential equations: :class:`Ude` and :class:`FluxUde`.

A UDE combines a mechanistic model with a neural one on the same state,
element-wise via ``op``:

* :class:`Ude` -- ``dy/dt = op(ode(t, y, args), node(t, y, args))``.
* :class:`FluxUde` -- combines *fluxes* before the stoichiometric map,
  ``dy/dt = nv(op(flux_ode.fluxes(...), flux_nn.fluxes(...)))``, so the
  correction acts per reaction; simulations report both parts as
  ``ode_fluxes``/``nn_fluxes``.

``op`` is one of the :data:`~._base.CombineOp` names -- ``"+"``, ``"-"``,
``"*"``, ``"/"``, ``"rel"`` (``ode * (1 + nn)``), ``"exp"``
(``ode * exp(nn)``), ``"pow"`` (``ode ** nn``) -- or any
``(ode, nn) -> combined`` callable. Readouts are always the mechanistic
component's. For a neural component with hidden dimensions see
:mod:`._aug_ude`.
"""

import operator
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from mxlpy.jax.simulation import Simulation

from ._base import (
    Base,
    CombineOp,
    FluxBase,
)
from ._node import FluxNode, Node
from ._ode import FluxOde, Ode


def _rel(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode * (1 + nn)


def _exp(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode * jnp.exp(nn)


def _pow(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode**nn


_OPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "rel": _rel,
    "exp": _exp,
    "pow": _pow,
}


def _resolve_op(
    op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if not isinstance(op, str):
        return op
    if op not in _OPS:
        msg = f"unknown op {op!r}; expected one of {sorted(_OPS)} or a callable"
        raise ValueError(msg)
    return _OPS[op]


class Ude(Base[Simulation]):
    """Universal Differential Equation: an ODE combined with a neural ODE.

    ``dy/dt = op(ode(t, y, args), nn(t, y, args))``, element-wise.

    Parameters
    ----------
    ode : Ode
        Mechanistic component.
    nn : Node
        Neural component; must have the same ``variable_names``.
    op : CombineOp or Callable
        ``"+"``, ``"-"``, ``"*"``, ``"/"``; ``"rel"``: ``ode * (1 + nn)``;
        ``"exp"``: ``ode * exp(nn)``; ``"pow"``: ``ode ** nn``; or any
        ``(ode, nn) -> combined`` function.
    """

    ode: Ode
    nn: Node
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        ode: Ode,
        nn: Node,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        if nn.variable_names != ode.variable_names:
            msg = (
                f"nn.variable_names {nn.variable_names} differ from "
                f"ode.variable_names {ode.variable_names}"
            )
            raise ValueError(msg)
        self.ode = ode
        self.nn = nn
        self.op = _resolve_op(op)
        self.variable_names = ode.variable_names
        self.readout_names = ode.readout_names

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``op(ode(t, y, args), nn(t, y, args))``."""
        return self.op(self.ode(time, y, args), self.nn(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts."""
        return self.ode.readouts(t, y, args)


class FluxUde(FluxBase):
    """Flux-based Universal Differential Equation.

    Combines the mechanistic and the neural *fluxes*:
    ``fluxes = op(flux_ode.fluxes, flux_nn.fluxes)`` and
    ``dy/dt = flux_ode.nv(fluxes)``.

    Parameters
    ----------
    flux_ode : FluxOde
        Mechanistic component.
    flux_nn : FluxNode
        Neural component; must have the same ``variable_names`` and
        ``flux_names`` (e.g. built via :meth:`FluxNode.from_flux_ode`).
    op : CombineOp or Callable
        See :class:`Ude`.
    """

    flux_ode: FluxOde
    flux_nn: FluxNode
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux_ode: FluxOde,
        flux_nn: FluxNode,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        for attr in ("variable_names", "flux_names"):
            if getattr(flux_nn, attr) != getattr(flux_ode, attr):
                msg = (
                    f"flux_nn.{attr} {getattr(flux_nn, attr)} differ from "
                    f"flux_ode.{attr} {getattr(flux_ode, attr)}"
                )
                raise ValueError(msg)
        self.flux_ode = flux_ode
        self.flux_nn = flux_nn
        self.op = _resolve_op(op)
        self.variable_names = flux_ode.variable_names
        self.flux_names = flux_ode.flux_names
        self.readout_names = flux_ode.readout_names

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``op(flux_ode.fluxes(t, y, args), flux_nn.fluxes(t, y, args))``."""
        return self.op(
            self.flux_ode.fluxes(t, y, args), self.flux_nn.fluxes(t, y, args)
        )

    def _flux_parts(
        self, ts: jax.Array, zs: jax.Array, args: jax.Array
    ) -> dict[str, jax.Array]:
        """Report the mechanistic and neural fluxes that ``op`` combines."""
        return {
            "ode_fluxes": jax.vmap(self.flux_ode.fluxes)(ts, zs, args),
            "nn_fluxes": jax.vmap(self.flux_nn.fluxes)(ts, zs, args),
        }

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``flux_ode.nv(fluxes(t, y, args))``."""
        return self.flux_ode.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts."""
        return self.flux_ode.readouts(t, y, args)
