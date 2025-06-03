# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107, D200, D203, D400, D401


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy

from mxlpy.meta.sympy_tools import fn_to_sympy

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mxlpy.model import Model

__all__ = [
    "SymbolicModel",
    "to_symbolic_model",
]


@dataclass
class SymbolicModel:
    variables: dict[str, sympy.Symbol]
    parameters: dict[str, sympy.Symbol]
    eqs: list[sympy.Expr]
    initial_conditions: dict[str, float]
    parameter_values: dict[str, float]

    def jacobian(self) -> sympy.Matrix:
        # FIXME: don't rely on ordering of variables
        return sympy.Matrix(self.eqs).jacobian(
            sympy.Matrix(list(self.variables.values()))
        )


def _list_of_symbols(args: Iterable[str]) -> list[sympy.Symbol]:
    return [sympy.Symbol(arg) for arg in args]


def to_symbolic_model(model: Model) -> SymbolicModel:
    cache = model._create_cache()  # noqa: SLF001

    variables: dict[str, sympy.Symbol] = dict(
        zip(model.variables, _list_of_symbols(model.variables), strict=True)
    )
    parameters: dict[str, sympy.Symbol] = dict(
        zip(model.parameters, _list_of_symbols(model.parameters), strict=True)
    )
    symbols: dict[str, sympy.Symbol | sympy.Expr] = variables | parameters  # type: ignore

    # Insert derived into symbols
    for k, v in model.derived.items():
        symbols[k] = fn_to_sympy(v.fn, [symbols[i] for i in v.args])

    # Insert derived into reaction via args
    rxns = {
        k: fn_to_sympy(v.fn, [symbols[i] for i in v.args])
        for k, v in model.reactions.items()
    }

    eqs: dict[str, sympy.Expr] = {}
    for cpd, stoich in cache.stoich_by_cpds.items():
        for rxn, stoich_value in stoich.items():
            eqs[cpd] = (
                eqs.get(cpd, sympy.Float(0.0)) + sympy.Float(stoich_value) * rxns[rxn]  # type: ignore
            )

    for cpd, dstoich in cache.dyn_stoich_by_cpds.items():
        for rxn, der in dstoich.items():
            eqs[cpd] = eqs.get(cpd, sympy.Float(0.0)) + fn_to_sympy(
                der.fn,
                [symbols[i] for i in der.args] * rxns[rxn],  # type: ignore
            )  # type: ignore

    return SymbolicModel(
        variables=variables,
        parameters=parameters,
        eqs=[eqs[i] for i in cache.var_names],
        initial_conditions=model.get_initial_conditions(),
        parameter_values=model.parameters,
    )
