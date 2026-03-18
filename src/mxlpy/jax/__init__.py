"""JAX export utilities for mxlpy models.

Provides :class:`JaxExport` and :func:`to_jax_export` which convert an
mxlpy :class:`~mxlpy.Model` into a JAX-compatible ODE function suitable
for use with Diffrax, Equinox, and other JAX-based tools.

Requires the ``jax`` optional dependency group::

    uv sync --extra jax

Examples
--------
>>> from mxlpy import Model, fns
>>> from mxlpy.jax import to_jax_export
>>> import diffrax
>>>
>>> model = (
...     Model()
...     .add_variables({"x": 1.0})
...     .add_parameters({"k": 0.1})
...     .add_reaction("v", fns.mass_action_1s, stoichiometry={"x": -1}, args=["x", "k"])
... )
>>> export = to_jax_export(model)
>>> term = diffrax.ODETerm(export.rhs)

"""

from __future__ import annotations

from mxlpy.jax._codegen import JaxExport, to_jax_export

__all__ = [
    "JaxExport",
    "to_jax_export",
]
