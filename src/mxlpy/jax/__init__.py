"""JAX export utilities for mxlpy models.

Provides :class:`JaxExport` and :func:`to_jax_export` which convert an
mxlpy :class:`~mxlpy.Model` into a JAX-compatible ODE function suitable
for use with Diffrax, Equinox, and other JAX-based tools.

Also provides the :class:`UDE` base class for Universal Differential
Equations and the :mod:`~mxlpy.jax.fit` submodule for training UDEs against
experimental data.

Requires the ``jax`` optional dependency group::

    uv sync --extra jax

Examples
--------
>>> from mxlpy import Model, fns
>>> from mxlpy.jax import to_jax_export, UDE
>>> from mxlpy.jax import fit as fit_jax
>>> import diffrax
>>>
>>> model = (
...     Model()
...     .add_variables({"x": 1.0})
...     .add_parameters({"k": 0.1})
...     .add_reaction("v", fns.mass_action_1s, stoichiometry={"x": -1}, args=["x", "k"])
... )
>>> export = model.to_jax()
>>> term = diffrax.ODETerm(export)

"""

from __future__ import annotations

from mxlpy.jax import fit
from mxlpy.jax._codegen import JaxExport, to_jax_export
from mxlpy.jax._ude import UDE, JaxFit

__all__ = [
    "JaxExport",
    "JaxFit",
    "UDE",
    "fit",
    "to_jax_export",
]
