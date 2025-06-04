# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107, D200, D203, D400, D401


from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import sympy

from mxlpy.meta.sympy_tools import fn_to_sympy, list_of_symbols

if TYPE_CHECKING:
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


def to_symbolic_model(model: Model) -> SymbolicModel:
    cache = model._create_cache()  # noqa: SLF001
    initial_conditions = model.get_initial_conditions()

    variables: dict[str, sympy.Symbol] = dict(
        zip(
            initial_conditions,
            cast(list[sympy.Symbol], list_of_symbols(initial_conditions)),
            strict=True,
        )
    )
    parameters: dict[str, sympy.Symbol] = dict(
        zip(
            model.get_parameter_values(),
            cast(list[sympy.Symbol], list_of_symbols(model.get_parameter_values())),
            strict=True,
        )
    )
    symbols: dict[str, sympy.Symbol | sympy.Expr] = variables | parameters  # type: ignore

    # Insert derived into symbols
    for k, v in model.get_raw_derived().items():
        if (
            expr := fn_to_sympy(v.fn, origin=k, model_args=[symbols[i] for i in v.args])
        ) is None:
            msg = f"Unable to parse derived value '{k}'"
            raise ValueError(msg)
        symbols[k] = expr

    # Insert derived into reaction via args
    rxns: dict[str, sympy.Expr] = {}
    for k, v in model.get_raw_reactions().items():
        if (
            expr := fn_to_sympy(v.fn, origin=k, model_args=[symbols[i] for i in v.args])
        ) is None:
            msg = f"Unable to parse reaction '{k}'"
            raise ValueError(msg)
        rxns[k] = expr

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
        parameter_values=model.get_parameter_values(),
    )
