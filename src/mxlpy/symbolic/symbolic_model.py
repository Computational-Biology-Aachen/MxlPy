"""Conversions into symbolic representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import sympy
from wadler_lindig import pformat

from mxlpy.meta._via_sym_repr import (
    _get_dependencies_and_leaves,
    _get_order,
    model_to_symbolic_repr,
)
from mxlpy.meta.sympy_tools import list_of_symbols

if TYPE_CHECKING:
    from mxlpy._kinetic_builder import KineticModelBuilder

__all__ = [
    "SymbolicModel",
    "to_symbolic_model",
]


@dataclass
class SymbolicModel:
    """Symbolic model."""

    variables: dict[str, sympy.Symbol]
    parameters: dict[str, sympy.Symbol]
    external: dict[str, sympy.Symbol]
    eqs: list[sympy.Expr]
    initial_conditions: dict[str, float]
    parameter_values: dict[str, float]

    def __repr__(self) -> str:
        return pformat(self)

    def jacobian(self) -> sympy.Matrix:
        """Get jacobian of symbolic model."""
        # FIXME: don't rely on ordering of variables
        return sympy.Matrix(self.eqs).jacobian(
            sympy.Matrix(list(self.variables.values()))
        )


def to_symbolic_model(
    model: KineticModelBuilder,
    *,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> SymbolicModel:
    """Convert model into symbolic representation.

    Parameters
    ----------
    model
        Model to convert
    custom_fns
        Sympy expressions to substitute for functions that cannot be parsed
        from source.

    Returns
    -------
    SymbolicModel
        Symbolic representation with sympy variables, parameters and equations

    """
    sr = model_to_symbolic_repr(
        model,
        only_warn=False,
        custom_fns={} if custom_fns is None else custom_fns,
    )
    rsr = sr.reduce()

    initial_conditions = model.get_initial_conditions()
    parameter_values = model.get_parameter_values()

    variables: dict[str, sympy.Symbol] = dict(
        zip(
            initial_conditions,
            cast(list[sympy.Symbol], list_of_symbols(initial_conditions)),
            strict=True,
        )
    )
    parameters: dict[str, sympy.Symbol] = dict(
        zip(
            parameter_values,
            cast(list[sympy.Symbol], list_of_symbols(parameter_values)),
            strict=True,
        )
    )
    data: dict[str, sympy.Symbol] = dict(
        zip(
            model._data,  # noqa: SLF001
            cast(list[sympy.Symbol], list_of_symbols(model._data)),  # noqa: SLF001
            strict=True,
        )
    )

    # Every derived/reaction/surrogate expression only refers to its own
    # immediate arguments (by name), which is exactly what per-language,
    # per-statement codegen wants. A Jacobian instead needs one closed-form
    # expression per diff eq, so we walk components in dependency order,
    # substituting each one's already-fully-expanded value into its
    # dependents, until only variables/parameters/data/time remain.
    subs_map: dict[str, sympy.Expr] = {
        **variables,
        **parameters,
        **data,
        "time": sympy.Symbol("time"),
    }

    diff_eq_symbols = {
        s.name
        for expr in rsr.diffeqs.values()
        for s in expr.free_symbols
        if isinstance(s, sympy.Symbol)
    }
    deps = _get_dependencies_and_leaves(model, diff_eq_symbols)

    def _expand(name: str) -> None:
        fn = rsr.derived.get(name) or rsr.reactions.get(name)
        if fn is None:
            return
        subs_map[name] = cast(
            sympy.Expr,
            fn.expr.subs({sympy.Symbol(a): subs_map[a] for a in fn.args}),
        )

    for name in _get_order(model):
        if name in variables or name in parameters:
            continue
        if (srg := sr.surrogates.get(name)) is not None:
            for output in srg.outputs:
                if output in deps:
                    _expand(output)
            continue
        if name in deps:
            _expand(name)

    full_subs = {sympy.Symbol(k): v for k, v in subs_map.items()}
    eqs = [
        cast(sympy.Expr, rsr.diffeqs[name].subs(full_subs))
        for name in model.get_variable_names()
    ]

    return SymbolicModel(
        variables=variables,
        parameters=parameters,
        eqs=eqs,
        initial_conditions=initial_conditions,
        parameter_values=parameter_values,
        external=data,
    )
