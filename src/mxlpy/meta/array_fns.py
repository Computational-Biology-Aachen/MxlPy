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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.integrate
import sympy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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
    "fold_data",
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


# sympy stand-in -> its numpy/scipy kernel, matching the `_pythoncode` printers above
_ARRAY_KERNELS: dict[type, Callable[..., Any]] = {
    ArraySum: np.sum,
    ArrayMean: np.mean,
    ArrayStd: lambda a: np.std(a, ddof=1),
    ArrayMaxOf: np.max,
    ArrayMinOf: np.min,
    ArrayIndex: lambda a, i: np.asarray(a)[int(i)],
    ArraySlice: lambda a, start, stop: np.asarray(a)[int(start) : int(stop)],
    ArrayAt: lambda a, t, times: np.asarray(a)[
        np.argmin(np.abs(np.asarray(times) - t))
    ],
    ArraySimpson: lambda y, x: scipy.integrate.simpson(y, x=x),
}


_NODE_NAMES: dict[type, str] = dict(ARRAY_NODE_TYPES)


def _outermost_array_nodes(expr: sympy.Basic) -> list[sympy.Basic]:
    if type(expr) in ARRAY_NODE_TYPES:
        return [expr]
    return [n for arg in expr.args for n in _outermost_array_nodes(arg)]


def _evaluate(expr: sympy.Basic, data: Mapping[str, Any]) -> Any:
    """Evaluate an expression over data arrays and numbers only."""
    if (kernel := _ARRAY_KERNELS.get(type(expr))) is not None:
        return kernel(*(_evaluate(arg, data) for arg in expr.args))
    values: dict[sympy.Basic, Any] = {}
    replacements: dict[sympy.Basic, sympy.Symbol] = {}
    for node in _outermost_array_nodes(expr):
        replacements[node] = dummy = sympy.Dummy()
        values[dummy] = _evaluate(node, data)
    elementwise = expr.xreplace(replacements)
    symbols = list(elementwise.free_symbols)
    for symbol in symbols:
        if symbol not in values:
            values[symbol] = np.asarray(data[str(symbol)], dtype=float)
    fn = sympy.lambdify(symbols, elementwise, modules="numpy")
    return fn(*(values[s] for s in symbols))


def fold_data(expr: sympy.Expr, data: Mapping[str, Any], where: str) -> sympy.Expr:
    """`expr` with every array subtree evaluated against `data` and replaced by its number.

    An array subtree may reference only data and numbers (mxl-schemas ADR 0014),
    so it is a constant, and every scalar backend can treat the model as an
    ordinary one. Changing the data means folding again.

    Raises
    ------
    ValueError
        Naming `where`, if an array subtree references anything but data, if it
        evaluates to an array, or if data is used outside one.

    """
    if not data:
        return expr

    def touches_data(node: sympy.Basic) -> bool:
        return bool(node.atoms(*ARRAY_NODE_TYPES)) or any(
            s.name in data for s in node.free_symbols if isinstance(s, sympy.Symbol)
        )

    def fold(node: sympy.Basic) -> sympy.Basic:
        if not touches_data(node):
            return node
        if (node_type := _NODE_NAMES.get(type(node))) is not None:
            others = sorted(str(s) for s in node.free_symbols if str(s) not in data)
            if others:
                msg = (
                    f"{where}: {node_type} may only reference data and numbers, "
                    f"not {', '.join(repr(s) for s in others)}"
                )
                raise ValueError(msg)
            value = _evaluate(node, data)
            if np.ndim(value) != 0:
                msg = (
                    f"{where}: {node_type} gives an array; reduce it to a number "
                    "(e.g. with Sum or Simpson)"
                )
                raise ValueError(msg)
            return sympy.Float(float(value))
        if isinstance(node, sympy.Symbol):
            msg = (
                f"{where}: data {node.name!r} is an array; reduce it to a number "
                "(e.g. with Sum or Simpson) before using it"
            )
            raise ValueError(msg)  # noqa: TRY004, same model error as the others
        return node.func(*(fold(arg) for arg in node.args))

    return cast(sympy.Expr, fold(expr))
