# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Mechanistic models: :class:`Ode` and :class:`FluxOde`.

These wrap JAX right-hand sides generated from an mxlpy model via
:func:`~mxlpy.meta._via_sym_repr.generate_model_code_jax`, with a vector of
trainable parameters appended to the runtime ``args``:

* :class:`Ode` -- ``dy/dt = rhs(t, y, concat(args, pars))``. Built from a
  ``Model`` or ``OdeModelBuilder`` via :meth:`Ode.from_mxlpy`.
* :class:`FluxOde` -- ``dy/dt = nv(flux_fn(t, y, concat(args, pars)))``,
  keeping reaction fluxes and stoichiometry separate. Built from a
  ``Model`` via :meth:`FluxOde.from_mxlpy`; required by the flux-based
  neural models, which reuse its ``nv``, names and readouts.

``from_mxlpy`` splits the model's parameters into ``parameters_to_fit``
(trainable, stored in ``pars``), ``free_parameters`` (passed as ``args`` at
call time) and everything else (baked into the generated code as
constants). Both classes are the mechanistic component of the UDEs in
:mod:`._ude` and :mod:`._aug_ude`.
"""

from typing import Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy.jax.simulation import Simulation
from mxlpy.meta._via_sym_repr import generate_model_code_jax

from ._base import (
    Base,
    FluxBase,
    Nv,
    ReadoutFn,
    Rhs,
    _readout_names,
    _resolve_readouts,
)


class Ode(Base[Simulation]):
    """ODE model with trainable parameter vector.

    Trainable parameters are appended to ``args`` at each evaluation, so
    ``rhs`` receives ``concat(args, pars)`` as its third argument -- the
    argument layout :func:`generate_model_code_jax` emits (free/runtime
    parameters first, trainable parameters last).

    Parameters
    ----------
    rhs : Rhs
        Right-hand-side function ``(t, y, args) -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector.
    variable_names : tuple[str, ...]
        Names of the state variables, in ``rhs``'s order.
    readout_fn : ReadoutFn or None
        Readout function ``(t, y, concat(args, pars)) -> readouts``.
    readout_names : tuple[str, ...] or None
        Names of ``readout_fn``'s outputs; required with ``readout_fn``.
    readout_scale : jax.Array or float or None
        Multiplier for the readouts. A plain float (the default, 1.0) is
        not trained; pass a ``jax.Array`` to learn it.
    out_scale : jax.Array or None
        Multiplier for ``dy/dt``.  Defaults to 1.0.
    """

    rhs: Rhs
    pars: jax.Array
    out_scale: jax.Array
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        rhs: Rhs,
        pars: jax.Array,
        variable_names: tuple[str, ...],
        *,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        out_scale: jax.Array | None = None,
    ) -> None:
        self.rhs = rhs
        self.pars = pars
        self.variable_names = variable_names
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale
        self.out_scale = jnp.array([1.0]) if out_scale is None else out_scale

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``out_scale * rhs(t, y, concat(args, pars))``."""
        return self.out_scale * self.rhs(time, y, jnp.concat((args, self.pars)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return self.readout_scale * self.readout_fn(t, y, jnp.concat((args, self.pars)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder | OdeModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
        readouts_to_calculate: list[str] | None = None,
    ) -> Self:
        """Construct an :class:`Ode` from an mxlpy ``Model`` or ``OdeModelBuilder``.

        Generates JAX code for the RHS and the readouts via symbolic code
        generation and initialises ``pars`` from the model's current
        parameter values.

        Parameters
        ----------
        mxlpy_model : Model or OdeModelBuilder
            The mechanistic model to convert.
        parameters_to_fit : list[str] or None
            Parameters that become trainable (stored in ``pars``).
        free_parameters : list[str] or None
            Parameters passed as external ``args`` at call time rather than
            baked into the RHS.
        readouts_to_calculate : list[str] or None
            Subset of the model's derived parameters, derived variables,
            reactions, surrogate outputs and readouts that :meth:`readouts`
            returns. ``None`` returns all of them.

        Returns
        -------
        Self
            A new :class:`Ode`.
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
            derived_to_calculate=readouts_to_calculate,
        )

        # A single shared namespace, so names bound by ``codegen.imports``
        # (e.g. ``jnp``, ``functools``) are visible to the generated code
        # regardless of what this module happens to import at the top level.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.model, generated)  # noqa: S102
        exec(codegen.derived, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        return cls(
            rhs=cast(Rhs, generated["model"]),
            pars=jnp.array([model_pars[k] for k in parameters_to_fit]),
            variable_names=tuple(mxlpy_model.get_initial_conditions()),
            readout_fn=cast(ReadoutFn, generated["derived"]),
            readout_names=_readout_names(mxlpy_model, readouts_to_calculate),
        )


class FluxOde(FluxBase):
    """ODE model defined via reaction fluxes and a stoichiometric map.

    ``dy/dt = nv(fluxes(t, y, args))``, with
    ``fluxes(t, y, args) = flux_scale * flux_fn(t, y, concat(args, pars))``.

    Parameters
    ----------
    flux_fn : Rhs
        Raw flux function ``(t, y, concat(args, pars)) -> flux_vector``.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector, appended to ``args``.
    variable_names : tuple[str, ...]
        Names of the state variables, in the order ``flux_fn``/``nv`` expect
        and return them.
    flux_names : tuple[str, ...]
        Names of the fluxes, in ``flux_fn``'s output order.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`.
    flux_scale : jax.Array or None
        Multiplier for the fluxes.  Defaults to 1.0.
    """

    flux_fn: Rhs
    nv: Nv
    pars: jax.Array
    flux_scale: jax.Array
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux_fn: Rhs,
        nv: Nv,
        pars: jax.Array,
        variable_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        *,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        flux_scale: jax.Array | None = None,
    ) -> None:
        self.flux_fn = flux_fn
        self.nv = nv
        self.pars = pars
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.flux_scale = jnp.array([1.0]) if flux_scale is None else flux_scale
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``flux_scale * flux_fn(t, y, concat(args, pars))``."""
        return self.flux_scale * self.flux_fn(t, y, jnp.concat((args, self.pars)))

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``nv(fluxes(t, y, args))``."""
        return self.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return self.readout_scale * self.readout_fn(t, y, jnp.concat((args, self.pars)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
        readouts_to_calculate: list[str] | None = None,
    ) -> Self:
        """Construct a :class:`FluxOde` from an mxlpy ``Model``.

        See :meth:`Ode.from_mxlpy` for the parameters; additionally
        generates the flux function and the stoichiometric map. By default
        the readouts exclude the fluxes, which :meth:`fluxes` already
        reports (scaled by ``flux_scale``).
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        flux_names = tuple(
            mxlpy_model.get_arg_names(
                include_time=False,
                include_variables=False,
                include_parameters=False,
                include_derived_parameters=False,
                include_derived_variables=False,
                include_reactions=True,
                include_surrogate_variables=False,
                include_surrogate_fluxes=True,
                include_readouts=False,
            )
        )
        if readouts_to_calculate is None:
            readouts_to_calculate = [
                name
                for name in _readout_names(mxlpy_model, None)
                if name not in flux_names
            ]
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
            derived_to_calculate=readouts_to_calculate,
        )

        # See Ode.from_mxlpy for the shared namespace.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.fluxes, generated)  # noqa: S102
        exec(codegen.nv, generated)  # noqa: S102
        exec(codegen.derived, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        return cls(
            flux_fn=cast(Rhs, generated["fluxes"]),
            nv=cast(Nv, generated["nv"]),
            pars=jnp.array([model_pars[k] for k in parameters_to_fit]),
            variable_names=tuple(mxlpy_model.get_initial_conditions()),
            flux_names=flux_names,
            readout_fn=cast(ReadoutFn, generated["derived"]),
            readout_names=_readout_names(mxlpy_model, readouts_to_calculate),
        )
