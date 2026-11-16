# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numeric Jacobian functions compiled from the symbolic model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sympy

from mxlpy.symbolic.symbolic_model import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy.types import Array

__all__ = ["jacobian_fn"]


def jacobian_fn(model: KineticModelBuilder) -> Callable[[float, Array], Array]:
    """Compile the analytic Jacobian of `model` into a numeric function.

    Parameters are function arguments rather than substituted constants, so
    the returned function stays valid across parameter updates; it reads the
    current values whenever the model cache changed. Only the non-zero
    entries are evaluated, with common subexpressions eliminated.

    Parameters
    ----------
    model
        Model to differentiate

    Returns
    -------
    Callable[[float, Array], Array]
        `jac(time, variables)` returning the dense (n, n) Jacobian

    Raises
    ------
    ValueError
        If the Jacobian depends on model data, which can't be compiled

    """
    sym = to_symbolic_model(model)
    var_names = model.get_variable_names()
    par_names = model.get_parameter_names()
    time = sympy.Symbol("time")
    xs = [sym.variables[k] for k in var_names]
    ps = [sym.parameters[k] for k in par_names]

    jac = sympy.Matrix(sym.eqs).jacobian(xs)
    n = len(xs)
    nonzero = [(i, j) for i in range(n) for j in range(n) if jac[i, j] != 0]
    exprs = [jac[i, j] for i, j in nonzero]

    unsupported = set().union(*(e.free_symbols for e in exprs)) - {time, *xs, *ps}
    if unsupported:
        msg = (
            f"Jacobian depends on unsupported symbols: {sorted(map(str, unsupported))}"
        )
        raise ValueError(msg)

    rows = np.array([i for i, _ in nonzero], dtype=np.intp)
    cols = np.array([j for _, j in nonzero], dtype=np.intp)
    # Symbols instead of names, as names needn't be valid identifiers
    fn = sympy.lambdify([time, xs, ps], exprs, modules="numpy", cse=True)

    # Every model change replaces its cache, so the cache's identity tells
    # whether the parameter vector needs rebuilding. float64 values make
    # e.g. 0.0 ** -1 evaluate to inf like the rhs instead of raising.
    cache: object = None
    params = np.empty(0)

    def jacobian(t: float, x: Array) -> Array:
        nonlocal cache, params
        if (current := model._cache) is None or current is not cache:  # noqa: SLF001
            values = model.get_parameter_values()
            cache = model._cache  # noqa: SLF001
            params = np.array([values[k] for k in par_names], dtype=float)
        out = np.zeros((n, n))
        out[rows, cols] = fn(t, x, params)
        return out

    # Fail here rather than mid-integration on expressions numpy can't evaluate
    y0 = model.get_initial_conditions()
    with np.errstate(all="ignore"):
        jacobian(0.0, np.array([y0[k] for k in var_names], dtype=float))
    return jacobian
