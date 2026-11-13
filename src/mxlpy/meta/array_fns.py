# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Sympy stand-ins for the array nodes (mxl-schemas ADR 0014, ADR 0015).

Sympy treats every symbol as a scalar, so a function over data arrays
(``simpson(absorption * light, x=wavelength)``) parses into an ordinary
expression whose element-wise parts are plain sympy arithmetic. Only the
operations that reduce or select from an array need their own (inert) sympy
function, one per array node, named like the node with an `Array` prefix.
Each prints (`sympy.pycode`, fully qualified) as the numpy/scipy call it
stands for, so generated Python functions parse back to the same expression.

Arrays may also depend on variables and parameters (ADR 0015): a derived
quantity can be an array, as long as everything the right-hand side needs is a
number. `array_lengths` infers which quantities are arrays and checks those
rules. Derivatives stay exact: each stand-in differentiates through its array
operand (`d Sum(a) = Sum(d a)`), so `sympy.diff` works on array expressions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.integrate
import sympy

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

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
    "array_lengths",
    "constant_value",
    "expr_length",
    "fold_data",
    "simpson_weights",
]


class _ArrayFunction(sympy.Function):
    """An array node; numpy printers (element-wise code) print it as Python does.

    Linear in its array operand (the first argument) unless it overrides
    `_eval_derivative`: the derivative is the node applied to the operand's.
    """

    def _pythoncode(self, printer: Any) -> str:
        raise NotImplementedError

    def _numpycode(self, printer: Any) -> str:
        return self._pythoncode(printer)

    def operand(self) -> sympy.Expr:
        """The array operand."""
        return cast(sympy.Expr, self.args[0])

    def _eval_derivative(self, s: sympy.Symbol) -> sympy.Expr:
        return cast(sympy.Expr, self.func(self.operand().diff(s), *self.args[1:]))


def _extreme_derivative(node: _ArrayFunction, s: sympy.Symbol) -> sympy.Expr:
    """The derivative at the extreme element, averaged over ties."""
    a = node.operand()
    at = sympy.Eq(a, node)
    total = cast(sympy.Expr, ArraySum(sympy.Piecewise((a.diff(s), at), (0, True))))
    ties = cast(sympy.Expr, ArraySum(sympy.Piecewise((1, at), (0, True))))
    return total / ties


class ArraySum(_ArrayFunction):
    """Sum of all elements of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.sum({printer.doprint(self.args[0])})"


class ArrayMean(_ArrayFunction):
    """Arithmetic mean of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.mean({printer.doprint(self.args[0])})"


class ArrayStd(_ArrayFunction):
    """Sample standard deviation (ddof = 1) of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.std({printer.doprint(self.args[0])}, ddof=1)"

    def _eval_derivative(self, s: sympy.Symbol) -> sympy.Expr:
        # sum((a - m) da) / ((n - 1) std), with n - 1 = sum((a - m)^2) / std^2
        a = self.operand()
        dev = a - cast(sympy.Expr, ArrayMean(a))
        num = cast(sympy.Expr, ArraySum(dev * a.diff(s)))
        return self * num / cast(sympy.Expr, ArraySum(dev**2))


class ArrayMaxOf(_ArrayFunction):
    """Largest element of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.max({printer.doprint(self.args[0])})"

    def _eval_derivative(self, s: sympy.Symbol) -> sympy.Expr:
        return _extreme_derivative(self, s)


class ArrayMinOf(_ArrayFunction):
    """Smallest element of an array."""

    nargs = 1

    def _pythoncode(self, printer: Any) -> str:
        return f"numpy.min({printer.doprint(self.args[0])})"

    def _eval_derivative(self, s: sympy.Symbol) -> sympy.Expr:
        return _extreme_derivative(self, s)


class ArrayIndex(_ArrayFunction):
    """Element ``i`` of an array (0-based, negative counts from the end)."""

    nargs = 2

    def _pythoncode(self, printer: Any) -> str:
        a, i = (printer.doprint(x) for x in self.args)
        return f"numpy.asarray({a})[int({i})]"


class ArraySlice(_ArrayFunction):
    """Elements ``start`` up to (excluding) ``stop`` of an array."""

    nargs = 3

    def _pythoncode(self, printer: Any) -> str:
        a, start, stop = (printer.doprint(x) for x in self.args)
        return f"numpy.asarray({a})[int({start}):int({stop})]"


class ArrayAt(_ArrayFunction):
    """Element of ``a`` at the sample of ``times`` nearest ``t``."""

    nargs = 3

    def _pythoncode(self, printer: Any) -> str:
        a, t, times = (printer.doprint(x) for x in self.args)
        return (
            f"numpy.asarray({a})[numpy.argmin(numpy.abs(numpy.asarray({times}) - {t}))]"
        )


class ArraySimpson(_ArrayFunction):
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


def simpson_weights(x: Any) -> np.ndarray:
    """Weights w with `simpson(y, x=x) == w @ y` for every y (Simpson's rule is linear in y)."""
    x = np.asarray(x, dtype=float)
    return np.asarray(scipy.integrate.simpson(np.eye(len(x)), x=x), dtype=float)


def constant_value(expr: sympy.Basic, data: Mapping[str, Any], where: str) -> Any:
    """The value of `expr`, which may reference only data and numbers.

    Raises
    ------
    ValueError
        Naming `where`, if `expr` references anything but data.

    """
    others = sorted(str(s) for s in expr.free_symbols if str(s) not in data)
    if others:
        msg = (
            f"{where}: {expr} may only reference data and numbers, "
            f"not {', '.join(repr(s) for s in others)}"
        )
        raise ValueError(msg)
    return _evaluate(expr, data)


def _index(value: Any, n: int, node: str, where: str) -> int:
    i = int(value)
    if i != value or not -n <= i < n:
        msg = f"{where}: {node} index {value} is not an element of a length-{n} array"
        raise ValueError(msg)
    return i % n


def expr_length(
    expr: sympy.Basic,
    lengths: Mapping[str, int],
    data: Mapping[str, Any],
    where: str,
) -> int | None:
    """The length of `expr` if it is an array, None if it is a number (ADR 0015).

    `lengths` holds every array-valued name (data and array-valued derived
    quantities); `data` the data values, which the operands that have to be
    constants (`Simpson`'s x, `Index`'s i, `Slice`'s bounds, `At`'s t and times)
    are evaluated against.

    Raises
    ------
    ValueError
        Naming `where`, if arrays of unequal lengths combine, a reduction gets a
        number, or an operand that has to be constant isn't.

    """

    def length(e: sympy.Basic) -> int | None:
        if isinstance(e, sympy.Symbol):
            if e.name in data and np.ndim(data[e.name]) != 1:
                msg = f"{where}: data {e.name!r} is not a 1-D array"
                raise ValueError(msg)
            return lengths.get(e.name)
        node = _NODE_NAMES.get(type(e))
        if node is None:
            found = {n for arg in e.args if (n := length(arg)) is not None}
            if len(found) > 1:
                msg = f"{where}: arrays of lengths {sorted(found)} combine in {e}"
                raise ValueError(msg)
            return found.pop() if found else None
        a, *rest = e.args
        if (n := length(a)) is None:
            msg = f"{where}: {node} needs an array, got the number {a}"
            raise ValueError(msg)
        for r in rest:
            length(r)
        const = [constant_value(r, data, f"{where}: {node}") for r in rest]
        if node == "Simpson":
            if np.shape(const[0]) != (n,):
                msg = f"{where}: Simpson's x must be an array of length {n}"
                raise ValueError(msg)
        elif node == "Index":
            _index(const[0], n, node, where)
        elif node == "At":
            if np.ndim(const[0]) != 0 or np.shape(const[1]) != (n,):
                msg = f"{where}: At needs a number t and times of length {n}"
                raise ValueError(msg)
        elif node == "Slice":
            start, stop = (int(c) for c in const)
            if not (m := len(range(n)[start:stop])):
                msg = f"{where}: Slice {start}:{stop} of a length-{n} array is empty"
                raise ValueError(msg)
            return m
        return None

    return length(expr)


def array_lengths(
    assignments: list[tuple[str, sympy.Expr]],
    data: Mapping[str, Any],
    scalars: Collection[str] = (),
) -> dict[str, int]:
    """The lengths of the array-valued `assignments`, given in any order (ADR 0015).

    Every array is 1-D and its length comes from data. Names in `scalars`
    (reaction rates, readouts, ...) must be numbers. An assignment is
    classified once the assignments it references are.

    Raises
    ------
    ValueError
        If a name in `scalars` is an array, assignments depend on each other
        cyclically, or `expr_length` rejects an expression (e.g. one using data
        that isn't 1-D).

    """
    lengths = {k: len(v) for k, v in data.items() if np.ndim(v) == 1}
    arrays: dict[str, int] = {}
    pending = dict(assignments)
    while pending:
        waiting = set(pending)
        ready = [
            k
            for k, e in pending.items()
            if {str(s) for s in e.free_symbols}.isdisjoint(waiting - {k})
        ]
        if not ready:
            msg = f"cyclic dependency between {sorted(pending)}"
            raise ValueError(msg)
        for name in ready:
            if (n := expr_length(pending.pop(name), lengths, data, name)) is None:
                continue
            if name in scalars:
                msg = (
                    f"{name}: is an array of length {n}, but must be a number; "
                    "reduce it (e.g. with Sum or Simpson)"
                )
                raise ValueError(msg)
            lengths[name] = arrays[name] = n
    return arrays


def fold_data(expr: sympy.Expr, data: Mapping[str, Any], where: str) -> sympy.Expr:
    """`expr` with every array subtree over data and numbers replaced by its number.

    Such a subtree is a constant (mxl-schemas ADR 0014), so it is computed once
    instead of on every evaluation. Array subtrees that reference anything else
    are left for array-aware code (ADR 0015); `array_lengths` checks them.

    Raises
    ------
    ValueError
        Naming `where`, if a constant subtree can't be evaluated.

    """
    if not data:
        return expr

    def touches_data(node: sympy.Basic) -> bool:
        return bool(node.atoms(*ARRAY_NODE_TYPES)) or any(
            s.name in data for s in node.free_symbols if isinstance(s, sympy.Symbol)
        )

    def fold(node: sympy.Basic) -> sympy.Basic:
        if not node.args or not touches_data(node):
            return node
        if type(node) in _NODE_NAMES and all(str(s) in data for s in node.free_symbols):
            try:
                value = _evaluate(node, data)
            except Exception as e:
                msg = f"{where}: can't evaluate {node}: {e}"
                raise ValueError(msg) from e
            if np.ndim(value) == 0:
                return sympy.Float(float(value))
        return node.func(*(fold(arg) for arg in node.args))

    return cast(sympy.Expr, fold(expr))
