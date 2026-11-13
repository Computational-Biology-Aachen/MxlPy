# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Flatten a model into one symbolic form that every backend emits from.

All symbols are renamed to positional ids (`y0`, `p0`, `a0`, `d0`), so no model name
can collide with a libm function or a backend's own argument.

Arrays (mxl-schemas ADR 0015) stay symbolic: an array-valued assignment is an
element-wise expression the backends evaluate in a loop, and the data it reads are
constants (`Lowered.data`). Lowering rewrites the array nodes into the few forms the
emitters handle: `Simpson` becomes `Sum` with precomputed weights, `At` an `Index`,
and every `Index`/`Slice` bound a non-negative integer.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import pickle
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sympy

from mxlpy.meta.array_fns import (
    ARRAY_NODE_TYPES,
    ArrayAt,
    ArrayIndex,
    ArraySimpson,
    ArraySlice,
    ArraySum,
    array_lengths,
    constant_value,
    expr_length,
    fold_data,
    simpson_weights,
)
from mxlpy.meta.source_tools import replace_symbols

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import pandas as pd

__all__ = [
    "Lowered",
    "NativeLoweringError",
    "content_key",
    "fingerprint",
    "lower",
    "lower_outputs",
]


class NativeLoweringError(ValueError):
    """The model can't be compiled, so the Native integrator can't run it."""


@dataclass(frozen=True)
class Lowered:
    """A model as flat symbolic assignments.

    `arrays` maps each array-valued assignment to its length, `data` each constant
    array the expressions read (`d0`, ...) to its values.
    """

    n: int
    par_names: list[str]
    assignments: list[tuple[str, Any]]
    dydt: list[Any]
    arrays: dict[str, int] = field(default_factory=dict)
    data: dict[str, np.ndarray] = field(default_factory=dict)


def _needed(
    assignments: list[tuple[str, Any]],
    outputs: list[Any],
) -> list[tuple[str, Any]]:
    needed = {str(s) for e in outputs for s in e.free_symbols}
    kept = []
    for name, expr in reversed(assignments):
        if name in needed:
            kept.append((name, expr))
            needed |= {str(s) for s in expr.free_symbols}
    return kept[::-1]


def _check_lowerable(model: Any) -> None:
    if not hasattr(model, "get_parameter_values"):
        msg = f"Native needs a model, got {type(model).__name__}"
        raise NativeLoweringError(msg)


def _data(model: Any) -> dict[str, np.ndarray]:
    return {
        k: np.asarray(v, dtype=float).squeeze()
        for k, v in getattr(model, "get_raw_data", dict)().items()
    }


def _fold(model: Any, assignments: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Replace the array subtrees over model data by their values, as mxlweb-core does."""
    data = _data(model)
    try:
        return [(k, fold_data(sympy.sympify(e), data, k)) for k, e in assignments]
    except Exception as e:
        msg = f"Native can't lower this model's use of data: {e}"
        raise NativeLoweringError(msg) from e


class _Arrays:
    """Rewrites array nodes into the forms the emitters handle, collecting their data."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data
        self.used: dict[str, np.ndarray] = {}
        self.weights: dict[bytes, sympy.Symbol] = {}

    def _weights(self, x: np.ndarray) -> sympy.Symbol:
        key = x.tobytes()
        if key not in self.weights:
            self.weights[key] = sym = sympy.Symbol(f"_w{len(self.weights)}")
            self.used[sym.name] = simpson_weights(x)
        return self.weights[key]

    def rewrite(self, expr: sympy.Basic, lengths: dict[str, int], where: str) -> Any:
        def length(e: sympy.Basic) -> int:
            return cast(int, expr_length(e, lengths, self.data, where))

        def const(e: sympy.Basic) -> Any:
            return constant_value(e, self.data, where)

        def walk(e: sympy.Basic) -> Any:
            if isinstance(e, sympy.Symbol):
                if e.name in self.data:
                    self.used[e.name] = self.data[e.name]
                return e
            if not e.args:
                return e
            if isinstance(e, ArraySimpson):
                return ArraySum(self._weights(const(e.args[1])) * walk(e.args[0]))
            if isinstance(e, ArrayAt):
                times = np.asarray(const(e.args[2]), dtype=float)
                i = int(np.argmin(np.abs(times - float(const(e.args[1])))))
                return ArrayIndex(walk(e.args[0]), sympy.Integer(i))
            if isinstance(e, ArrayIndex):
                n = length(e.args[0])
                return ArrayIndex(
                    walk(e.args[0]), sympy.Integer(int(const(e.args[1])) % n)
                )
            if isinstance(e, ArraySlice):
                r = range(length(e.args[0]))[
                    int(const(e.args[1])) : int(const(e.args[2]))
                ]
                if r.step != 1:
                    msg = f"{where}: Native can't compile the slice {e}"
                    raise NativeLoweringError(msg)
                return ArraySlice(
                    walk(e.args[0]), sympy.Integer(r.start), sympy.Integer(r.stop)
                )
            return e.func(*(walk(a) for a in e.args))

        return walk(expr)


def _lower_arrays(
    data: dict[str, np.ndarray],
    assignments: list[tuple[str, Any]],
    outputs: list[tuple[str, Any]],
    scalars: set[str],
) -> tuple[list[tuple[str, Any]], list[Any], dict[str, int], dict[str, np.ndarray]]:
    """Check the array rules (ADR 0015) and rewrite the array nodes for the emitters.

    `outputs` and `scalars` (e.g. reaction rates) must be numbers.
    """
    exprs = [e for _, e in assignments + outputs]
    if not any(
        s.name in data for e in exprs for s in e.free_symbols if s.is_Symbol
    ) and not any(e.atoms(*ARRAY_NODE_TYPES) for e in exprs):
        # every use of data was folded into a constant
        return assignments, [e for _, e in outputs], {}, {}
    try:
        arrays = array_lengths(
            assignments + outputs, data, scalars | {k for k, _ in outputs}
        )
        lengths = {k: len(v) for k, v in data.items() if np.ndim(v) == 1} | arrays
        rw = _Arrays(data)
        body = [(k, rw.rewrite(e, lengths, k)) for k, e in assignments]
        out = [rw.rewrite(e, lengths, k) for k, e in outputs]
    except NativeLoweringError:
        raise
    except Exception as e:
        msg = f"Native can't lower this model's arrays: {e}"
        raise NativeLoweringError(msg) from e
    return body, out, arrays, rw.used


def _rates(model: Any) -> set[str]:
    return set(getattr(model, "get_reaction_names", list)())


def _renamed_data(
    used: dict[str, np.ndarray],
) -> tuple[dict[sympy.Symbol, sympy.Symbol], dict[str, np.ndarray]]:
    rename = {sympy.Symbol(k): sympy.Symbol(f"d{i}") for i, k in enumerate(used)}
    return rename, {rename[sympy.Symbol(k)].name: v for k, v in used.items()}


def lower(model: Any) -> Lowered:
    """Lower a model, raising NativeLoweringError if it isn't symbolically representable."""
    from mxlpy.meta._via_sym_repr import _normalized_symbolic_model

    _check_lowerable(model)
    var_names = list(model.get_variable_names())
    par_names = list(model.get_parameter_names())
    # Readouts don't affect the derivatives, so one that can't be lowered mustn't block this.
    without_readouts = copy.copy(model)
    without_readouts._readouts = {}  # noqa: SLF001
    without_readouts._cache = None  # noqa: SLF001
    try:
        nsm = _normalized_symbolic_model(without_readouts, par_names, None, {})
    except Exception as e:
        msg = f"Native can't lower this model to symbolic form: {e!r}"
        raise NativeLoweringError(msg) from e

    folded = _fold(model, nsm.body + [(v, nsm.diff_eqs[v]) for v in var_names])
    diff_eqs = dict(folded[len(nsm.body) :])
    body = _needed(folded[: len(nsm.body)], [diff_eqs[v] for v in var_names])
    arrays: dict[str, int] = {}
    data: dict[str, np.ndarray] = {}
    if model_data := _data(model):
        body, dydt, arrays, data = _lower_arrays(
            model_data,
            body,
            [(f"d{v}/dt", diff_eqs[v]) for v in var_names],
            _rates(model),
        )
        diff_eqs = dict(zip(var_names, dydt, strict=True))
    rename: dict[sympy.Symbol, sympy.Symbol] = {sympy.Symbol("time"): sympy.Symbol("t")}
    rename |= {sympy.Symbol(v): sympy.Symbol(f"y{i}") for i, v in enumerate(var_names)}
    rename |= {sympy.Symbol(p): sympy.Symbol(f"p{i}") for i, p in enumerate(par_names)}
    rename |= {sympy.Symbol(a): sympy.Symbol(f"a{i}") for i, (a, _) in enumerate(body)}
    data_rename, data = _renamed_data(data)
    rename |= data_rename
    return Lowered(
        n=len(var_names),
        par_names=par_names,
        assignments=[
            (rename[sympy.Symbol(a)].name, replace_symbols(e, rename)) for a, e in body
        ],
        dydt=[replace_symbols(diff_eqs[v], rename) for v in var_names],
        arrays={rename[sympy.Symbol(k)].name: n for k, n in arrays.items()},
        data=data,
    )


def lower_outputs(model: Any, names: list[str]) -> Lowered:
    """Lower everything a simulation can report, with `names` as the outputs.

    Results can be selected freely after a simulation, so all of them are computed,
    unlike the right-hand side, which only needs what the derivatives depend on.
    """
    from mxlpy.meta._via_sym_repr import _normalized_symbolic_model

    _check_lowerable(model)
    var_names = list(model.get_variable_names())
    par_names = list(model.get_parameter_names())
    try:
        nsm = _normalized_symbolic_model(model, par_names, None, {})
    except Exception as e:
        msg = f"Native can't lower this model to symbolic form: {e!r}"
        raise NativeLoweringError(msg) from e
    rename: dict[sympy.Symbol, sympy.Symbol] = {sympy.Symbol("time"): sympy.Symbol("t")}
    rename |= {sympy.Symbol(v): sympy.Symbol(f"y{i}") for i, v in enumerate(var_names)}
    rename |= {sympy.Symbol(p): sympy.Symbol(f"p{i}") for i, p in enumerate(par_names)}
    extended = _fold(model, nsm.extended)
    known = {a for a, _ in extended} | set(var_names) | set(par_names) | {"time"}
    if missing := [n for n in names if n not in known]:
        msg = f"Native has no symbolic form for: {missing}"
        raise NativeLoweringError(msg)
    outputs: list[Any] = [sympy.Symbol(n) for n in names]
    arrays: dict[str, int] = {}
    data: dict[str, np.ndarray] = {}
    if model_data := _data(model):
        extended, outputs, arrays, data = _lower_arrays(
            model_data, extended, [(n, sympy.Symbol(n)) for n in names], _rates(model)
        )
    rename |= {
        sympy.Symbol(a): sympy.Symbol(f"a{i}") for i, (a, _) in enumerate(extended)
    }
    data_rename, data = _renamed_data(data)
    rename |= data_rename
    return Lowered(
        n=len(var_names),
        par_names=par_names,
        assignments=[
            (rename[sympy.Symbol(a)].name, replace_symbols(e, rename))
            for a, e in extended
        ],
        dydt=[replace_symbols(e, rename) for e in outputs],
        arrays={rename[sympy.Symbol(k)].name: n for k, n in arrays.items()},
        data=data,
    )


class _Identity:
    """Compares by identity, and keeps its object alive so the id stays unique."""

    def __init__(self, item: object) -> None:
        self.item = item

    def __hash__(self) -> int:
        return id(self.item)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and other.item is self.item


def _component(item: Any, key: Callable[[Any], Hashable]) -> Hashable:
    stoich = getattr(item, "stoichiometry", None) or {}
    return (
        key(item.fn),
        tuple(item.args),
        tuple(
            (k, v if isinstance(v, int | float) else _component(v, key))
            for k, v in stoich.items()
        ),
    )


def _digest(values: pd.Series | pd.DataFrame) -> str:
    return hashlib.sha256(pickle.dumps(values.to_numpy())).hexdigest()


def _structure(
    model: Any,
    fn_key: Callable[[Any], Hashable],
    surrogate_key: Callable[[Any], Hashable],
) -> Hashable:
    _check_lowerable(model)
    return (
        tuple(model.get_variable_names()),
        tuple(model.get_parameter_names()),
        *(
            tuple(
                (k, _component(v, fn_key)) for k, v in getattr(model, attr, {}).items()
            )
            for attr in ("_derived", "_reactions", "_diff_eqs", "_readouts")
        ),
        tuple(
            (k, surrogate_key(v)) for k, v in getattr(model, "_surrogates", {}).items()
        ),
        # Data is folded into the compiled code as constants
        tuple((k, _digest(v)) for k, v in getattr(model, "_data", {}).items()),
    )


def fingerprint(model: Any) -> Hashable:
    """A cheap key for what the compiled code depends on, not parameter values.

    It keeps references to the rate functions, so their identity stays unique.
    """
    return _structure(model, _Identity, _Identity)


def _ref(obj: Any) -> tuple[str, str, str]:
    return ("ref", getattr(obj, "__module__", ""), getattr(obj, "__qualname__", ""))


def _code(code: types.CodeType) -> tuple:
    return (
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        tuple(
            _code(c) if isinstance(c, types.CodeType) else (type(c).__name__, repr(c))
            for c in code.co_consts
        ),
    )


def _global_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            names |= _global_names(c)
    return names


def _cell(cell: types.CellType, seen: set[int]) -> Hashable:
    try:
        return _canon(cell.cell_contents, seen)
    except ValueError:  # an empty cell
        return ("empty",)


def _canon(obj: Any, seen: set[int]) -> Hashable:
    """What `obj` computes, comparable across processes and pickling."""
    if obj is None or isinstance(obj, bool | int | float | complex | str | bytes):
        return (type(obj).__name__, repr(obj))
    if isinstance(obj, tuple | list):
        return (type(obj).__name__, tuple(_canon(i, seen) for i in obj))
    if isinstance(obj, dict):
        return (
            "dict",
            tuple((_canon(k, seen), _canon(v, seen)) for k, v in obj.items()),
        )
    if isinstance(obj, types.FunctionType):
        if id(obj) in seen:
            return ("recursion", obj.__qualname__)
        seen.add(id(obj))
        globs = obj.__globals__
        return (
            "fn",
            obj.__module__,
            obj.__qualname__,
            _code(obj.__code__),
            _canon(obj.__defaults__, seen),
            _canon(obj.__kwdefaults__, seen),
            tuple(_cell(c, seen) for c in obj.__closure__ or ()),
            tuple(
                (n, _canon(globs[n], seen))
                for n in sorted(_global_names(obj.__code__))
                if n in globs
            ),
        )
    if isinstance(obj, functools.partial):
        return (
            "partial",
            _canon(obj.func, seen),
            _canon(obj.args, seen),
            _canon(obj.keywords, seen),
        )
    if isinstance(obj, types.ModuleType):
        return ("module", obj.__name__)
    if isinstance(obj, type | types.BuiltinFunctionType | np.ufunc):
        return _ref(obj)
    if isinstance(obj, np.ndarray):
        return (
            "ndarray",
            str(obj.dtype),
            obj.shape,
            hashlib.sha256(obj.tobytes()).hexdigest(),
        )
    if dataclasses.is_dataclass(obj):
        return (
            *_ref(type(obj)),
            tuple(
                (f.name, _canon(getattr(obj, f.name), seen))
                for f in dataclasses.fields(obj)
            ),
        )
    return (*_ref(type(obj)), hashlib.sha256(pickle.dumps(obj)).hexdigest())


def content_key(model: Any) -> str | None:
    """A key for what the compiled code depends on, which survives pickling.

    Unlike `fingerprint`, which compares rate functions by identity, this hashes
    their code, defaults, closures and globals, so a model sent to another process
    keeps its key. Equal keys make equal code very likely but don't guarantee it
    (e.g. a function calling a global function by an attribute), so callers have to
    check the compiled code. None if some part of the model can't be keyed.
    """
    try:
        structure = _structure(
            model,
            lambda fn: _canon(fn, set()),
            lambda surrogate: _canon(surrogate, set()),
        )
    except Exception:  # noqa: BLE001  anything unkeyable just isn't shared
        return None
    return hashlib.sha256(repr(structure).encode()).hexdigest()
