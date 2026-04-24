"""Generate mxlpy code from a model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import sympy
from wadler_lindig import pformat

from mxlpy.meta.source_tools import fn_to_sympy_expr, fn_to_sympy_exprs
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
)
from mxlpy.surrogates import qss
from mxlpy.types import Derived, InitialAssignment
from mxlpy.units import Quantity

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model

__all__ = [
    "Err",
    "SymbolicFn",
    "SymbolicParameter",
    "SymbolicReaction",
    "SymbolicRepr",
    "SymbolicSurrogate",
    "SymbolicVariable",
    "model_to_symbolic_repr",
]

_LOGGER = logging.getLogger()


@dataclass
class SymbolicFn:
    """Container for symbolic fn."""

    fn_name: str
    expr: sympy.Expr
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicVariable:
    """Container for symbolic variable."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicParameter:
    """Container for symbolic par."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicReaction:
    """Container for symbolic rxn."""

    fn: SymbolicFn
    stoichiometry: dict[str, sympy.Float | SymbolicFn]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicSurrogate:
    """Container for symbolic rxn."""

    fns: list[SymbolicFn]
    outputs: list[str]
    stoichiometry: dict[str, dict[str, sympy.Float | SymbolicFn]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicRepr:
    """Container for symbolic model."""

    variables: dict[str, SymbolicVariable] = field(default_factory=dict)
    parameters: dict[str, SymbolicParameter] = field(default_factory=dict)
    derived: dict[str, SymbolicFn] = field(default_factory=dict)
    readouts: dict[str, SymbolicFn] = field(default_factory=dict)
    reactions: dict[str, SymbolicReaction] = field(default_factory=dict)
    surrogates: dict[str, SymbolicSurrogate] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


class Err(sympy.Function):
    @classmethod
    def eval(cls, *_: tuple[Any, ...]) -> sympy.Float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return sympy.Float(1.0)


def _fn_to_symbolic_repr(
    k: str,
    fn: Callable,
    model_args: list[str],
    *,
    only_warn: bool,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> SymbolicFn:
    fn_name = fn.__name__
    args = cast(list, list_of_symbols(model_args))
    if (expr := custom_fns.get(k)) is not None:
        if isinstance(expr, list):
            msg = f"Expected a single expr for '{k}' but got multiple"
            if only_warn:
                expr = sympy.Float("nan")
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

        return SymbolicFn(fn_name=fn_name, expr=expr, args=model_args)

    if (expr := fn_to_sympy_expr(fn, origin=k, model_args=args)) is None:
        msg = f"Unable to parse fn for '{k}'"
        if only_warn:
            expr = sympy.Float("nan")
            _LOGGER.warning(msg)
        else:
            raise ValueError(msg)
    return SymbolicFn(fn_name=fn_name, expr=expr, args=model_args)


def _fns_to_symbolic_reprs(
    k: str,
    fn: Callable,
    model_args: list[str],
    outputs: list[str],
    *,
    only_warn: bool,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> list[SymbolicFn]:
    fn_name = fn.__name__
    args = cast(list, list_of_symbols(model_args))
    exprs = fn_to_sympy_exprs(fn, origin=k, model_args=args)
    if (exprs := custom_fns.get(k)) is not None:
        if not isinstance(exprs, list):
            msg = f"Expected multiple exprs for '{k}' but got a single expr"
            if only_warn:
                exprs = [sympy.Float("nan") for i in outputs]
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

        return [
            SymbolicFn(fn_name=f"{fn_name}_{i}", expr=expr, args=model_args)
            for i, expr in zip(outputs, exprs, strict=True)
        ]

    if exprs is None:
        msg = f"Unable to parse fns for '{k}'"
        if only_warn:
            exprs = [sympy.Float("nan") for _ in outputs]
            _LOGGER.warning(msg)
        else:
            raise ValueError(msg)

    return [
        SymbolicFn(fn_name=f"{fn_name}_{i}", expr=expr, args=model_args)
        for i, expr in zip(outputs, exprs, strict=True)
    ]


def model_to_symbolic_repr(
    model: Model,
    *,
    only_warn: bool = False,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> SymbolicRepr:
    sym = SymbolicRepr()

    for k, variable in model.get_raw_variables().items():
        sym.variables[k] = SymbolicVariable(
            value=_fn_to_symbolic_repr(
                k,
                val.fn,
                val.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )
            if isinstance(val := variable.initial_value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, variable.unit),
        )

    for k, parameter in model.get_raw_parameters().items():
        sym.parameters[k] = SymbolicParameter(
            value=_fn_to_symbolic_repr(
                k,
                val.fn,
                val.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )
            if isinstance(val := parameter.value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, parameter.unit),
        )

    for k, der in model.get_raw_derived().items():
        sym.derived[k] = _fn_to_symbolic_repr(
            k,
            der.fn,
            der.args,
            only_warn=only_warn,
            custom_fns=custom_fns,
        )

    for k, rxn in model.get_raw_reactions().items():
        sym.reactions[k] = SymbolicReaction(
            fn=_fn_to_symbolic_repr(
                k,
                rxn.fn,
                rxn.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            ),
            stoichiometry={
                k: _fn_to_symbolic_repr(
                    k,
                    v.fn,
                    v.args,
                    only_warn=only_warn,
                    custom_fns=custom_fns,
                )
                if isinstance(v, Derived)
                else sympy.Float(v)
                for k, v in rxn.stoichiometry.items()
            },
        )

    for k, srg in model.get_raw_surrogates().items():
        if isinstance(srg, qss.Surrogate):
            fns = _fns_to_symbolic_reprs(
                k,
                srg.model,
                srg.args,
                srg.outputs,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )

            sym.surrogates[k] = SymbolicSurrogate(
                fns=fns,
                outputs=srg.outputs,
                stoichiometry={
                    out: {
                        k: _fn_to_symbolic_repr(
                            k,
                            v.fn,
                            v.args,
                            only_warn=only_warn,
                            custom_fns=custom_fns,
                        )
                        if isinstance(v, Derived)
                        else sympy.Float(v)
                        for k, v in stoich.items()
                    }
                    for out, stoich in srg.stoichiometries.items()
                },
            )
        else:
            msg = f"Unsupported surrogate type {type(srg)}."
            if only_warn:
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

    for k, der in model.get_raw_readouts().items():
        sym.readouts[k] = _fn_to_symbolic_repr(
            k,
            der.fn,
            der.args,
            only_warn=only_warn,
            custom_fns=custom_fns,
        )

    return sym
