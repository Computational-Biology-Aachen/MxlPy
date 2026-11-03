# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Sympy stand-ins for the array nodes (mxl-schemas ADR 0014).

Sympy treats every symbol as a scalar, so a function over data arrays
(``simpson(absorption * light, x=wavelength)``) parses into an ordinary
expression whose element-wise parts are plain sympy arithmetic. Only the
operations that reduce or select from an array need their own (inert) sympy
function, one per array node, named like the node with an `Array` prefix.
Each prints (`sympy.pycode`, fully qualified) as the numpy/scipy call it
stands for, so generated Python functions parse back to the same expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.integrate
import sympy

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ARRAY_FNS",
    "ARRAY_NODE_TYPES",
    "ArrayAt",
    "ArrayIndex",
    "ArrayMaxOf",
    "ArrayMean",
    "ArrayMinOf",
    "ArraySimpson",
    "ArraySlice",
    "ArrayStd",
    "ArraySum",
    "array_fn_from_call",
]


class ArraySum(sympy.Function):
    """Sum of all elements of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.sum({printer.doprint(self.args[0])})"


class ArrayMean(sympy.Function):
    """Arithmetic mean of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.mean({printer.doprint(self.args[0])})"


class ArrayStd(sympy.Function):
    """Sample standard deviation (ddof = 1) of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.std({printer.doprint(self.args[0])}, ddof=1)"


class ArrayMaxOf(sympy.Function):
    """Largest element of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.max({printer.doprint(self.args[0])})"


class ArrayMinOf(sympy.Function):
    """Smallest element of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.min({printer.doprint(self.args[0])})"


class ArrayIndex(sympy.Function):
    """Element ``i`` of an array (0-based, negative counts from the end)."""

    nargs = 2

    def _pythoncode(self, printer: Any) -> str:
        a, i = (printer.doprint(x) for x in self.args)
        return f"numpy.asarray({a})[int({i})]"


class ArraySlice(sympy.Function):
    """Elements ``start`` up to (excluding) ``stop`` of an array."""

    nargs = 3

    def _pythoncode(self, printer: Any) -> str:
        a, start, stop = (printer.doprint(x) for x in self.args)
        return f"numpy.asarray({a})[int({start}):int({stop})]"


class ArrayAt(sympy.Function):
    """Element of ``a`` at the sample of ``times`` nearest ``t``."""

    nargs = 3

    def _pythoncode(self, printer: Any) -> str:
        a, t, times = (printer.doprint(x) for x in self.args)
        return (
            f"numpy.asarray({a})[numpy.argmin(numpy.abs(numpy.asarray({times}) - {t}))]"
        )


class ArraySimpson(sympy.Function):
    """Integral of ``y`` over ``x`` by Simpson's rule (``scipy.integrate.simpson``)."""

    nargs = 2

    def _pythoncode(self, printer: Any) -> str:
        y, x = (printer.doprint(a) for a in self.args)
        return f"scipy.integrate.simpson({y}, x={x})"


# sympy stand-in -> array node name. The stand-ins carry an `Array` prefix:
# sympy's printers dispatch on the class name, and e.g. `Sum` is taken.
ARRAY_NODE_TYPES: dict[type[sympy.Function], str] = {
    cls: cls.__name__.removeprefix("Array")
    for cls in (
        ArraySum,
        ArrayMean,
        ArrayStd,
        ArrayMaxOf,
        ArrayMinOf,
        ArrayIndex,
        ArraySlice,
        ArrayAt,
        ArraySimpson,
    )
}


def _reduction(
    cls: type[sympy.Function],
) -> Callable[[list[sympy.Expr], dict[str, sympy.Expr]], sympy.Expr | None]:
    def handle(
        args: list[sympy.Expr], kwargs: dict[str, sympy.Expr]
    ) -> sympy.Expr | None:
        if len(args) != 1 or kwargs:
            return None
        return cls(args[0])

    return handle


def _std(args: list[sympy.Expr], kwargs: dict[str, sympy.Expr]) -> sympy.Expr | None:
    # The node is the sample standard deviation; numpy defaults to ddof=0.
    if len(args) != 1 or set(kwargs) != {"ddof"} or float(kwargs["ddof"]) != 1:
        return None
    return ArrayStd(args[0])


def _simpson(
    args: list[sympy.Expr], kwargs: dict[str, sympy.Expr]
) -> sympy.Expr | None:
    # Only the explicit-axis form: `dx` / the default unit spacing would hide
    # the sample positions the node needs.
    if len(args) != 1 or set(kwargs) != {"x"}:
        return None
    return ArraySimpson(args[0], kwargs["x"])


ARRAY_FNS: dict[
    Callable, Callable[[list[sympy.Expr], dict[str, sympy.Expr]], sympy.Expr | None]
] = {
    np.sum: _reduction(ArraySum),
    np.mean: _reduction(ArrayMean),
    np.std: _std,
    np.max: _reduction(ArrayMaxOf),
    np.amax: _reduction(ArrayMaxOf),
    np.min: _reduction(ArrayMinOf),
    np.amin: _reduction(ArrayMinOf),
    scipy.integrate.simpson: _simpson,
}


def array_fn_from_call(
    py_fn: Callable, args: list[sympy.Expr], kwargs: dict[str, sympy.Expr]
) -> sympy.Expr | None:
    """The array-node expression for a call of `py_fn`, or None if it isn't one (or its arguments aren't supported)."""
    if (handle := ARRAY_FNS.get(py_fn)) is None:
        return None
    return handle(args, kwargs)
