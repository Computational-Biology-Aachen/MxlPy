# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Flatten a model into one symbolic form that every backend emits from.

All symbols are renamed to positional ids (`y0`, `p0`, `a0`), so no model name can
collide with a libm function or a backend's own argument.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import pickle
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy

from mxlpy.meta.array_fns import fold_data
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
    """A model as flat symbolic assignments."""

    n: int
    par_names: list[str]
    assignments: list[tuple[str, Any]]
    dydt: list[Any]


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


def _fold(model: Any, assignments: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Replace the array subtrees over model data by their values, as mxlweb-core does."""
    data = getattr(model, "get_raw_data", dict)()
    try:
        return [(k, fold_data(sympy.sympify(e), data, k)) for k, e in assignments]
    except Exception as e:
        msg = f"Native can't lower this model's use of data: {e}"
        raise NativeLoweringError(msg) from e


def lower(model: Any) -> Lowered:
    """Lower a model, raising NativeLoweringError if it isn't symbolically representable."""
    from mxlpy.meta._via_sym_repr import _normalized_symbolic_model  # noqa: PLC0415

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
    rename: dict[sympy.Symbol, sympy.Symbol] = {sympy.Symbol("time"): sympy.Symbol("t")}
    rename |= {sympy.Symbol(v): sympy.Symbol(f"y{i}") for i, v in enumerate(var_names)}
    rename |= {sympy.Symbol(p): sympy.Symbol(f"p{i}") for i, p in enumerate(par_names)}
    rename |= {sympy.Symbol(a): sympy.Symbol(f"a{i}") for i, (a, _) in enumerate(body)}
    return Lowered(
        n=len(var_names),
        par_names=par_names,
        assignments=[
            (rename[sympy.Symbol(a)].name, replace_symbols(e, rename)) for a, e in body
        ],
        dydt=[replace_symbols(diff_eqs[v], rename) for v in var_names],
    )


def lower_outputs(model: Any, names: list[str]) -> Lowered:
    """Lower everything a simulation can report, with `names` as the outputs.

    Results can be selected freely after a simulation, so all of them are computed,
    unlike the right-hand side, which only needs what the derivatives depend on.
    """
    from mxlpy.meta._via_sym_repr import _normalized_symbolic_model  # noqa: PLC0415

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
    rename |= {
        sympy.Symbol(a): sympy.Symbol(f"a{i}") for i, (a, _) in enumerate(extended)
    }
    missing = [n for n in names if sympy.Symbol(n) not in rename]
    if missing:
        msg = f"Native has no symbolic form for: {missing}"
        raise NativeLoweringError(msg)
    return Lowered(
        n=len(var_names),
        par_names=par_names,
        assignments=[
            (rename[sympy.Symbol(a)].name, replace_symbols(e, rename))
            for a, e in extended
        ],
        dydt=[rename[sympy.Symbol(n)] for n in names],
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
