from __future__ import annotations

import re
import sys
import unicodedata
from importlib import util
from typing import TYPE_CHECKING

import pysbml
import sympy

from mxlpy.meta.codegen_mxlpy import (
    SymbolicFn,
    SymbolicParameter,
    SymbolicReaction,
    SymbolicRepr,
    SymbolicVariable,
    generate_mxlpy_code_from_symbolic_repr,
)
from mxlpy.paths import default_tmp_dir
from mxlpy.types import unwrap

__all__ = ["free_symbols", "import_from_path", "read", "valid_filename"]

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mxlpy.model import Model


def free_symbols(expr: sympy.Expr) -> list[str]:
    return [i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol)]


def _transform_stoichiometry(
    k: str,
    v: pysbml.transform.data.Expr,
) -> SymbolicFn | str | sympy.Float:
    if isinstance(v, sympy.Float):
        return v
    if isinstance(v, sympy.Symbol):
        return v.name

    return SymbolicFn(k, expr=v, args=free_symbols(v))


def _codegen(name: str, model: pysbml.transform.data.Model) -> Path:
    sym = SymbolicRepr()
    for key, var in model.variables.items():
        sym.variables[key] = SymbolicVariable(
            value=var.value,
            unit=var.unit,
        )

    for key, par in model.parameters.items():
        sym.parameters[key] = SymbolicParameter(value=par.value, unit=par.unit)

    for key, der in model.derived.items():
        sym.derived[key] = SymbolicFn(fn_name=key, expr=der, args=free_symbols(der))

    for key, rxn in model.reactions.items():
        sym.reactions[key] = SymbolicReaction(
            fn=SymbolicFn(fn_name=key, expr=rxn.expr, args=free_symbols(rxn.expr)),
            stoichiometry={
                k: _transform_stoichiometry(k, v) for k, v in rxn.stoichiometry.items()
            },
        )
    for key, der in model.initial_assignments.items():
        if key in model.parameters:
            sym.parameters[key].value = SymbolicFn(
                fn_name=key, expr=der, args=free_symbols(der)
            )
        elif key in model.variables:
            sym.variables[key].value = SymbolicFn(
                fn_name=key, expr=der, args=free_symbols(der)
            )

    path = default_tmp_dir(None, remove_old_cache=False) / f"{name}.py"
    with path.open("w+") as f:
        f.write(
            generate_mxlpy_code_from_symbolic_repr(
                sym,
                imports=[
                    "import math",
                    "import scipy",
                ],
            )
        )
    return path


def import_from_path(module_name: str, file_path: Path) -> Callable[[], Model]:
    spec = unwrap(util.spec_from_file_location(module_name, file_path))
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    unwrap(spec.loader).exec_module(module)
    return module.create_model


def valid_filename(value: str) -> str:
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "_", value).strip("-_")
    return f"mb_{value}"


def read(file: Path) -> Model:
    """Import a metabolic model from an SBML file.

    Args:
        file: Path to the SBML file to import.

    Returns:
        Model: Imported model instance.

    """
    model = pysbml.load_and_transform_model(file)
    out_name = valid_filename(file.stem)
    model_fn = import_from_path(out_name, _codegen(out_name, model))
    return model_fn()
