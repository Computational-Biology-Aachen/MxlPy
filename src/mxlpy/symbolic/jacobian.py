# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Numeric Jacobian functions compiled from the symbolic model."""

from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import sympy
from sympy.printing.numpy import NumPyPrinter

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.meta._via_sym_repr import (
    SymbolicDerived,
    SymbolicReaction,
    SymbolicReadout,
    _get_dependencies_and_leaves,
    _get_order,
    model_to_symbolic_repr,
)
from mxlpy.symbolic.symbolic_model import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.types import Array

__all__ = ["jacobian_fn"]

# Code generated from a model's structure, per structure token (see
# `mxlpy.types.StructureToken`). Parameter values, data constants and initial
# values are runtime inputs, so entries stay valid across value updates, and
# they're dropped with the token, i.e. once the structure changes or the
# model is garbage collected.
_GENERATED: weakref.WeakKeyDictionary[object, dict[tuple, Any]] = (
    weakref.WeakKeyDictionary()
)


def _generated[T](
    model: KineticModelBuilder | OdeModelBuilder, key: tuple, build: Callable[[], T]
) -> T:
    """`build()` for the model's current structure, generated only once."""
    entries = _GENERATED.setdefault(model._structure, {})  # noqa: SLF001
    if key not in entries:
        entries[key] = build()
    return entries[key]


type _RawJacobian = Callable[[float, Array, Array], tuple[float, ...]]
type _RawDerivatives = Callable[[float, Array, Array, Array], tuple[float, ...]]


def jacobian_fn(
    model: KineticModelBuilder | OdeModelBuilder,
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
        for nested min/max/if. Only supports `KineticModelBuilder`.

    Returns
    -------
    Callable[[float, Array], Array]
        `jac(time, variables)` returning the dense (n, n) Jacobian

    Raises
    ------
    ValueError
        If the Jacobian depends on model data, which can't be compiled

    """
    constant_names: list[str] = []
    if method == "chain_rule":
        compiled = compile_derivatives(model)
        nonzero = compiled.rhs_nonzero
        fn = compiled.fn
        constant_names = compiled.constant_names
    elif isinstance(model, KineticModelBuilder):
        expanded, nonzero = _generated(
            model, ("expanded_jacobian",), lambda: _expanded(model)
        )
        fn = functools.partial(_without_constants, expanded)
    else:
        msg = "The expanded method only supports KineticModelBuilder"
        raise TypeError(msg)

    var_names = model.get_variable_names()
    par_names = model.get_parameter_names()
    n = len(var_names)
    rows = np.array([i for i, _ in nonzero], dtype=np.intp)
    cols = np.array([j for _, j in nonzero], dtype=np.intp)

    # Every model change (incl. data updates) replaces its cache, so the
    # cache's identity tells whether parameters and constants need
    # rebuilding. float64 values make e.g. 0.0 ** -1 evaluate to inf like
    # the rhs instead of raising.
    cache: object = None
    params = np.empty(0)
    consts = np.empty(0)

    def jacobian(t: float, x: Array) -> Array:
        nonlocal cache, params, consts
        if (current := model._cache) is None or current is not cache:  # noqa: SLF001
            values = all_parameter_values(model)
            cache = model._cache  # noqa: SLF001
            params = np.array([values[k] for k in par_names], dtype=float)
            consts = constant_values(model, constant_names)
        out = np.zeros((n, n))
        if len(rows) > 0:
            out[rows, cols] = fn(t, x, params, consts)
        return out

    # Fail here rather than mid-integration on expressions numpy can't evaluate
    y0 = model.get_initial_conditions()
    with np.errstate(all="ignore"):
        jacobian(0.0, np.array([y0[k] for k in var_names], dtype=float))
    return jacobian


def _without_constants(
    fn: _RawJacobian, t: float, x: Array, p: Array, _: Array
) -> tuple[float, ...]:
    return fn(t, x, p)


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


def all_parameter_values(
    model: KineticModelBuilder | OdeModelBuilder,
) -> dict[str, float]:
    """Values of all parameters, including those set by initial assignments."""
    if (cache := model._cache) is None:  # noqa: SLF001
        cache = model._create_cache()  # noqa: SLF001
    return cache.all_parameter_values


@dataclass(slots=True)
class CompiledDerivatives:
    """Forward-mode derivatives of a model's rhs and outputs.

    Columns `0..n-1` are the variables, `n..n+m-1` the parameters in `wrt`.

    Attributes
    ----------
    fn
        `fn(time, x, p, c)` returning the non-zero entries, first those of
        `rhs_nonzero`, then those of `out_nonzero`. `p` holds the values of
        all model parameters (`model.get_parameter_names()` order), `c` those
        of `constant_names`
    rhs_nonzero
        (row, column) of each non-zero entry of d rhs / d (x, wrt)
    out_nonzero
        (output index, column) of each non-zero entry of d outputs / d (x, wrt)
    wrt
        Parameters with a derivative column
    outputs
        Derived quantities and reactions with a derivative row
    constant_names
        Derived quantities computed only from model data. They can't be
        differentiated, but are constant, so their values are runtime inputs
    constants
        Values of `constant_names` when first compiled for this structure,
        see `constant_values` for the current ones

    """

    fn: _RawDerivatives
    rhs_nonzero: list[tuple[int, int]]
    out_nonzero: list[tuple[int, int]]
    wrt: list[str]
    outputs: list[str]
    constant_names: list[str]
    constants: Array


def constant_values(
    model: KineticModelBuilder | OdeModelBuilder, names: list[str]
) -> Array:
    """Current values of the data-derived constants `names`.

    Raises
    ------
    ValueError
        If a value isn't a scalar

    """
    if not names:
        return np.empty(0)
    args = model.get_args(time=0.0)
    try:
        return np.array([args[k] for k in names], dtype=float)
    except (TypeError, ValueError) as e:
        msg = "Only scalar values derived from model data can be compiled"
        raise ValueError(msg) from e


def _data_constants(model: KineticModelBuilder | OdeModelBuilder) -> list[str]:
    """Names of derived quantities depending on nothing but model data."""
    const = set(getattr(model, "get_raw_data", dict)())
    if not const:
        return []
    derived = model.get_raw_derived(as_copy=False)
    found: list[str] = []
    changed = True
    while changed:
        changed = False
        for k, v in derived.items():
            if k not in const and v.args and set(v.args) <= const:
                const.add(k)
                found.append(k)
                changed = True
    return found


def compile_derivatives(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    wrt: list[str] | None = None,
    outputs: list[str] | None = None,
) -> CompiledDerivatives:
    """Generate forward-mode code propagating derivatives through the graph.

    Generated once per model structure and `wrt` / `outputs`, so repeated
    calls (e.g. one per Jacobian of a fit) after only parameter or variable
    values changed reuse the code. See `_compile_derivatives`.
    """
    key = (
        "derivatives",
        None if wrt is None else tuple(wrt),
        None if outputs is None else tuple(outputs),
    )
    return _generated(
        model, key, lambda: _compile_derivatives(model, wrt=wrt, outputs=outputs)
    )


def _compile_derivatives(
    model: KineticModelBuilder | OdeModelBuilder,
    *,
    wrt: list[str] | None = None,
    outputs: list[str] | None = None,
) -> CompiledDerivatives:
    """Generate forward-mode code propagating derivatives through the graph.

    Each node (derived quantity, reaction, diff eq) gets its value and, for
    every variable (and parameter in `wrt`) it depends on, the derivative
    d node / d x_k = sum_a (d node / d a) * (d a / d x_k) over its arguments a.

    Parameters
    ----------
    model
        Model to differentiate
    wrt
        Parameters to differentiate with respect to, next to the variables
    outputs
        Derived quantities and reactions whose derivatives are returned
        additionally to those of the rhs

    Returns
    -------
    CompiledDerivatives
        The compiled function and the layout of its return value

    Raises
    ------
    ValueError
        If the derivatives depend on something that can't be compiled, e.g.
        model data used together with variables or parameters

    """
    wrt = [] if wrt is None else list(wrt)
    outputs = [] if outputs is None else list(outputs)
    var_names = model.get_variable_names()
    par_names = model.get_parameter_names()
    if unknown := set(wrt) - set(par_names):
        msg = f"Can only differentiate w.r.t. parameters, not {sorted(unknown)}"
        raise ValueError(msg)

    constant_names = _data_constants(model)
    sr = model_to_symbolic_repr(
        model,
        only_warn=False,
        custom_fns=dict.fromkeys(constant_names, sympy.Float(0)),
    )
    rsr = sr.reduce()
    printer = _ScalarNumpyPrinter({"fully_qualified_modules": True})
    n = len(var_names)

    lines = ["def _derivatives(time, x, p, c):"]
    # model name -> identifier in the generated code (names needn't be valid)
    ident: dict[str, str] = {"time": "time"}
    # name -> {column: identifier of the derivative}, "1" for x_k / p_j itself
    grad: dict[str, dict[int, str]] = {"time": {}}
    for i, name in enumerate(var_names):
        ident[name] = f"x{i}"
        grad[name] = {i: "1"}
        lines.append(f"    x{i} = x[{i}]")
    wrt_index = {name: n + j for j, name in enumerate(wrt)}
    for i, name in enumerate(par_names):
        ident[name] = f"p{i}"
        grad[name] = {wrt_index[name]: "1"} if name in wrt_index else {}
        lines.append(f"    p{i} = p[{i}]")
    for i, name in enumerate(constant_names):
        ident[name] = f"q{i}"
        grad[name] = {}
        lines.append(f"    q{i} = c[{i}]")

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
    deps = _get_dependencies_and_leaves(model, diff_eq_symbols | set(outputs))

    def emit_component(name: str) -> None:
        if name in grad or name not in deps:
            return
        fn = rsr.derived.get(name) or rsr.reactions.get(name)
        if fn is None:
            return
        if isinstance(fn, (SymbolicDerived, SymbolicReaction, SymbolicReadout)):
            fn = fn.fn
        emit(name, fn.expr)

    for name in _get_order(model):
        if (srg := sr.surrogates.get(name)) is not None:
            for output in srg.outputs:
                emit_component(output)
        else:
            emit_component(name)

    rhs_nonzero: list[tuple[int, int]] = []
    values: list[str] = []
    for i, name in enumerate(var_names):
        key = f"__rhs_{name}"
        emit(key, rsr.diffeqs[name])
        for k, code in grad[key].items():
            rhs_nonzero.append((i, k))
            values.append(code)

    out_nonzero: list[tuple[int, int]] = []
    for o, name in enumerate(outputs):
        if name not in grad:
            msg = f"Can't differentiate output '{name}'"
            raise ValueError(msg)
        for k, code in grad[name].items():
            out_nonzero.append((o, k))
            values.append(code)
    lines.append(f"    return ({''.join(f'{v}, ' for v in values)})")

    namespace: dict[str, object] = {"numpy": np}
    exec(compile("\n".join(lines), "<mxlpy-derivatives>", "exec"), namespace)  # noqa: S102

    constants = constant_values(model, constant_names)

    return CompiledDerivatives(
        fn=namespace["_derivatives"],  # type: ignore[arg-type]
        rhs_nonzero=rhs_nonzero,
        out_nonzero=out_nonzero,
        wrt=wrt,
        outputs=outputs,
        constant_names=constant_names,
        constants=constants,
    )
