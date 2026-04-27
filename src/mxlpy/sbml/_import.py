from __future__ import annotations

import re
import sys
import unicodedata
from importlib import util
from typing import TYPE_CHECKING

import pysbml
import sympy

from mxlpy.meta._via_sym_repr import (
    SymbolicFn,
    SymbolicParameter,
    SymbolicReaction,
    SymbolicRepr,
    SymbolicVariable,
)
from mxlpy.paths import default_tmp_dir

__all__ = ["free_symbols", "import_from_path", "read", "valid_filename"]

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mxlpy.model import Model


def free_symbols(expr: sympy.Expr) -> list[str]:
    """Extract free symbol names from a sympy expression.

    Parameters
    ----------
    expr
        Sympy expression to extract symbols from

    Returns
    -------
    list[str]
        Names of the free symbols in the expression

    """
    return [i.name for i in expr.free_symbols if isinstance(i, sympy.Symbol)]


def _transform_stoichiometry(
    k: str,
    v: pysbml.transform.data.Expr,
) -> SymbolicFn | sympy.Float:
    if isinstance(v, sympy.Float):
        return v
    if isinstance(v, sympy.Symbol):
        return SymbolicFn(v.name, expr=v, args=[v.name])

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
        f.write(sym.generate_mxlpy())
    return path


def import_from_path(module_name: str, file_path: Path) -> Callable[[], Model]:
    """Import a model factory function from a generated Python file.

    Parameters
    ----------
    module_name
        Name to register the module under in sys.modules
    file_path
        Path to the Python file to import

    Returns
    -------
    Callable[[], Model]
        The ``create_model`` function from the imported module

    """
    spec = util.spec_from_file_location(module_name, file_path)
    assert spec is not None  # noqa: S101
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader = spec.loader
    assert loader is not None  # noqa: S101
    loader.exec_module(module)
    return module.create_model


def valid_filename(value: str) -> str:
    """Sanitise a string into a valid Python module filename.

    Parameters
    ----------
    value
        Raw string to sanitise

    Returns
    -------
    str
        Sanitised filename prefixed with ``mb_``

    """
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "_", value).strip("-_")
    return f"mb_{value}"


def read(file: Path) -> Model:
    """Import a metabolic model from an SBML file.

    Parameters
    ----------
    file
        Path to the SBML file to import.

    Returns
    -------
    Model
        Imported model instance.

    """
    model = pysbml.load_and_transform_model(file)
    out_name = valid_filename(file.stem)
    model_fn = import_from_path(out_name, _codegen(out_name, model))
    return model_fn()
