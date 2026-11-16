# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numeric Jacobian functions compiled from the symbolic model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import sympy
from sympy.printing.numpy import NumPyPrinter

from mxlpy.meta._via_sym_repr import (
    SymbolicDerived,
    SymbolicReaction,
    _get_dependencies_and_leaves,
    _get_order,
    model_to_symbolic_repr,
)
from mxlpy.symbolic.symbolic_model import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy.types import Array

__all__ = ["jacobian_fn"]

type _RawJacobian = Callable[[float, Array, Array], tuple[float, ...]]


def jacobian_fn(
    model: KineticModelBuilder,
    *,
    method: Literal["chain_rule", "expanded"] = "chain_rule",
) -> Callable[[float, Array], Array]:
    """Compile the analytic Jacobian of `model` into a numeric function.

    Parameters are function arguments rather than substituted constants, so
    the returned function stays valid across parameter updates; it reads the
    current values whenever the model cache changed. Only the non-zero
    entries are evaluated.

    Parameters
    ----------
    model
        Model to differentiate
    method
        `"chain_rule"` differentiates every derived quantity and reaction
        w.r.t. its direct arguments only and propagates the derivatives
        through the model graph, so its cost grows linearly with the model.
        `"expanded"` differentiates the fully expanded diff eqs, which
        duplicates shared subexpressions and can grow exponentially, e.g.
        for nested min/max/if.

    Returns
    -------
    Callable[[float, Array], Array]
        `jac(time, variables)` returning the dense (n, n) Jacobian

    Raises
    ------
    ValueError
        If the Jacobian depends on model data, which can't be compiled

    """
    if method == "chain_rule":
        fn, nonzero = _chain_rule(model)
    else:
        fn, nonzero = _expanded(model)

    var_names = model.get_variable_names()
    par_names = model.get_parameter_names()
    n = len(var_names)
    rows = np.array([i for i, _ in nonzero], dtype=np.intp)
    cols = np.array([j for _, j in nonzero], dtype=np.intp)

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
        if len(rows) > 0:
            out[rows, cols] = fn(t, x, params)
        return out

    # Fail here rather than mid-integration on expressions numpy can't evaluate
    y0 = model.get_initial_conditions()
    with np.errstate(all="ignore"):
        jacobian(0.0, np.array([y0[k] for k in var_names], dtype=float))
    return jacobian


def _expanded(model: KineticModelBuilder) -> tuple[_RawJacobian, list[tuple[int, int]]]:
    sym = to_symbolic_model(model)
    time = sympy.Symbol("time")
    xs = [sym.variables[k] for k in model.get_variable_names()]
    ps = [sym.parameters[k] for k in model.get_parameter_names()]

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

    # Symbols instead of names, as names needn't be valid identifiers
    fn = sympy.lambdify(
        [time, xs, ps],
        exprs,
        # builtin min/max: numpy's reduce over an axis given as second arg
        modules=[{"min": min, "max": max}, "numpy"],
        printer=_ScalarNumpyPrinter({"fully_qualified_modules": False}),
        cse=True,
    )
    return fn, nonzero


class _ScalarNumpyPrinter(NumPyPrinter):
    """numpy functions (inf/nan instead of exceptions) with scalar control flow."""

    def _print_Piecewise(self, expr: sympy.Piecewise) -> str:  # noqa: N802
        out = self._module_format("numpy.nan")
        pairs = cast(tuple[tuple[sympy.Expr, sympy.Basic], ...], expr.args)
        for e, c in reversed(pairs):
            out = f"(({self._print(e)}) if ({self._print(c)}) else ({out}))"
        return out

    def _print_Min(self, expr: sympy.Min) -> str:  # noqa: N802
        return f"min({', '.join(self._print(a) for a in expr.args)})"

    def _print_Max(self, expr: sympy.Max) -> str:  # noqa: N802
        return f"max({', '.join(self._print(a) for a in expr.args)})"

    def _print_Heaviside(self, expr: sympy.Heaviside) -> str:  # noqa: N802
        return f"(1.0 if ({self._print(expr.args[0])}) > 0 else 0.0)"

    def _print_DiracDelta(self, expr: sympy.DiracDelta) -> str:  # noqa: ARG002, N802
        return "0.0"


def _chain_rule(
    model: KineticModelBuilder,
) -> tuple[_RawJacobian, list[tuple[int, int]]]:
    """Generate forward-mode code propagating derivatives through the graph.

    Each node (derived quantity, reaction, diff eq) gets its value and, for
    every variable it depends on, the derivative
    d node / d x_k = sum_a (d node / d a) * (d a / d x_k) over its arguments a.
    """
    sr = model_to_symbolic_repr(model, only_warn=False, custom_fns={})
    rsr = sr.reduce()
    var_names = model.get_variable_names()
    printer = _ScalarNumpyPrinter({"fully_qualified_modules": True})

    lines = ["def _jacobian(time, x, p):"]
    # model name -> identifier in the generated code (names needn't be valid)
    ident: dict[str, str] = {"time": "time"}
    # name -> {variable index: identifier of the derivative}, "1" for x_k itself
    grad: dict[str, dict[int, str]] = {"time": {}}
    for i, name in enumerate(var_names):
        ident[name] = f"x{i}"
        grad[name] = {i: "1"}
        lines.append(f"    x{i} = x[{i}]")
    for i, name in enumerate(model.get_parameter_names()):
        ident[name] = f"p{i}"
        grad[name] = {}
        lines.append(f"    p{i} = p[{i}]")

    def emit(key: str, expr: sympy.Expr) -> None:
        idx = len(lines)
        names = {s.name for s in expr.free_symbols if isinstance(s, sympy.Symbol)}
        if unsupported := names - ident.keys():
            msg = f"Jacobian depends on unsupported symbols: {sorted(unsupported)}"
            raise ValueError(msg)
        active = [a for a in sorted(names) if grad[a]]
        rename = {sympy.Symbol(a): sympy.Symbol(ident[a]) for a in names}
        exprs = [expr, *(expr.diff(sympy.Symbol(a)) for a in active)]
        repl, (value, *partials) = sympy.cse(
            [e.xreplace(rename) for e in exprs],
            symbols=sympy.numbered_symbols(f"c{idx}_"),
        )
        lines.extend(f"    {s} = {printer.doprint(e)}" for s, e in repl)
        ident[key] = f"n{idx}"
        lines.append(f"    n{idx} = {printer.doprint(value)}")

        terms: dict[int, list[str]] = {}
        for i, (arg, partial) in enumerate(zip(active, partials, strict=True)):
            if partial == 0:
                continue
            lines.append(f"    d{idx}_{i} = {printer.doprint(partial)}")
            for k, dk in grad[arg].items():
                terms.setdefault(k, []).append(
                    f"d{idx}_{i}" if dk == "1" else f"d{idx}_{i} * {dk}"
                )
        grad[key] = {}
        for k, ts in terms.items():
            lines.append(f"    g{idx}_{k} = {' + '.join(ts)}")
            grad[key][k] = f"g{idx}_{k}"

    diff_eq_symbols = {
        s.name
        for expr in rsr.diffeqs.values()
        for s in expr.free_symbols
        if isinstance(s, sympy.Symbol)
    }
    deps = _get_dependencies_and_leaves(model, diff_eq_symbols)

    def emit_component(name: str) -> None:
        if name in grad or name not in deps:
            return
        fn = rsr.derived.get(name) or rsr.reactions.get(name)
        if fn is None:
            return
        if isinstance(fn, (SymbolicDerived, SymbolicReaction)):
            fn = fn.fn
        emit(name, fn.expr)

    for name in _get_order(model):
        if (srg := sr.surrogates.get(name)) is not None:
            for output in srg.outputs:
                emit_component(output)
        else:
            emit_component(name)

    nonzero: list[tuple[int, int]] = []
    values: list[str] = []
    for i, name in enumerate(var_names):
        key = f"__rhs_{name}"
        emit(key, rsr.diffeqs[name])
        for k, code in grad[key].items():
            nonzero.append((i, k))
            values.append(code)
    lines.append(f"    return ({''.join(f'{v}, ' for v in values)})")

    namespace: dict[str, object] = {"numpy": np}
    exec(compile("\n".join(lines), "<mxlpy-jacobian>", "exec"), namespace)  # noqa: S102
    return namespace["_jacobian"], nonzero  # type: ignore[return-value]
