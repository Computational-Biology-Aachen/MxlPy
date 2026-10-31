# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Flatten a model into one symbolic form that every backend emits from.

All symbols are renamed to positional ids (`y0`, `p0`, `a0`), so no model name can
collide with a libm function or a backend's own argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy

if TYPE_CHECKING:
    from collections.abc import Hashable

__all__ = ["Lowered", "NativeLoweringError", "fingerprint", "lower"]


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


def lower(model: Any) -> Lowered:
    """Lower a model, raising NativeLoweringError if it isn't symbolically representable."""
    from mxlpy.meta._via_sym_repr import _normalized_symbolic_model  # noqa: PLC0415

    _check_lowerable(model)
    var_names = list(model.get_variable_names())
    par_names = list(model.get_parameter_names())
    try:
        nsm = _normalized_symbolic_model(model, par_names, None, {})
    except Exception as e:
        msg = f"Native can't lower this model to symbolic form: {e!r}"
        raise NativeLoweringError(msg) from e

    body = _needed(nsm.body, [nsm.diff_eqs[v] for v in var_names])
    rename: dict[sympy.Symbol, sympy.Symbol] = {sympy.Symbol("time"): sympy.Symbol("t")}
    rename |= {sympy.Symbol(v): sympy.Symbol(f"y{i}") for i, v in enumerate(var_names)}
    rename |= {sympy.Symbol(p): sympy.Symbol(f"p{i}") for i, p in enumerate(par_names)}
    rename |= {sympy.Symbol(a): sympy.Symbol(f"a{i}") for i, (a, _) in enumerate(body)}
    return Lowered(
        n=len(var_names),
        par_names=par_names,
        assignments=[
            (rename[sympy.Symbol(a)].name, e.xreplace(rename)) for a, e in body
        ],
        dydt=[sympy.sympify(nsm.diff_eqs[v]).xreplace(rename) for v in var_names],
    )


class _Identity:
    """Compares by identity, and keeps its object alive so the id stays unique."""

    def __init__(self, item: object) -> None:
        self.item = item

    def __hash__(self) -> int:
        return id(self.item)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and other.item is self.item


def _component(item: Any) -> Hashable:
    stoich = getattr(item, "stoichiometry", None) or {}
    return (
        _Identity(item.fn),
        tuple(item.args),
        tuple(
            (k, v if isinstance(v, int | float) else _component(v))
            for k, v in stoich.items()
        ),
    )


def fingerprint(model: Any) -> Hashable:
    """A cheap key for what the compiled code depends on, not parameter values.

    It keeps references to the rate functions, so their identity stays unique.
    """
    _check_lowerable(model)
    return (
        tuple(model.get_variable_names()),
        tuple(model.get_parameter_names()),
        tuple((k, _component(v)) for k, v in getattr(model, "_derived", {}).items()),
        tuple((k, _component(v)) for k, v in getattr(model, "_reactions", {}).items()),
        tuple((k, _Identity(v)) for k, v in getattr(model, "_surrogates", {}).items()),
    )
