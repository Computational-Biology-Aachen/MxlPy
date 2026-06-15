from __future__ import annotations

import re
import sys
import types
import unicodedata
from importlib import util
from typing import TYPE_CHECKING

import libsbml
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
from mxlpy.sbml._export import BQB_QUALIFIERS, BQM_QUALIFIERS
from mxlpy.types import Annotation

__all__ = [
    "free_symbols",
    "import_from_path",
    "read",
    "valid_filename",
]

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


def _codegen(model: pysbml.transform.data.Model) -> str:
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

    return sym.generate_mxlpy()


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


_BQB_REVERSE = {v: k for k, v in BQB_QUALIFIERS.items()}
_BQM_REVERSE = {v: k for k, v in BQM_QUALIFIERS.items()}


def _cv_terms_to_annotations(sbase: libsbml.SBase) -> list[Annotation]:
    """Extract MIRIAM annotations from an SBML element's CVTerms."""
    annotations: list[Annotation] = []
    for i in range(sbase.getNumCVTerms()):
        cv = sbase.getCVTerm(i)
        qualifier_type = cv.getQualifierType()
        if qualifier_type == libsbml.BIOLOGICAL_QUALIFIER:
            predicate = _BQB_REVERSE.get(cv.getBiologicalQualifierType())
        elif qualifier_type == libsbml.MODEL_QUALIFIER:
            predicate = _BQM_REVERSE.get(cv.getModelQualifierType())
        else:
            predicate = None
        if predicate is None:
            continue
        annotations.extend(
            Annotation(uri=cv.getResourceURI(j), predicate=predicate)
            for j in range(cv.getNumResources())
        )
    return annotations


def _attach_annotations(model: Model, file: Path) -> None:
    """Parse annotations from the SBML file and attach them to the model.

    ``pysbml`` discards ``<annotation>`` blocks during transformation, so the
    annotations are read directly from the libsbml document and matched back to
    the model components by name.
    """
    doc = libsbml.readSBMLFromFile(str(file))
    if (sbml_model := doc.getModel()) is None:
        return

    if model_annotations := _cv_terms_to_annotations(sbml_model):
        model.annotate_model(model_annotations)

    variables = model.get_raw_variables(as_copy=False)
    parameters = model.get_raw_parameters(as_copy=False)
    derived = model.get_raw_derived(as_copy=False)
    reactions = model.get_raw_reactions(as_copy=False)

    for species in sbml_model.getListOfSpecies():
        if (annotations := _cv_terms_to_annotations(species)) and (
            name := species.getId()
        ) in variables:
            variables[name].annotations = annotations

    for parameter in sbml_model.getListOfParameters():
        if annotations := _cv_terms_to_annotations(parameter):
            name = parameter.getId()
            if name in parameters:
                parameters[name].annotations = annotations
            elif name in derived:
                derived[name].annotations = annotations

    for rule in sbml_model.getListOfRules():
        if annotations := _cv_terms_to_annotations(rule):
            name = rule.getVariable()
            if name in derived:
                derived[name].annotations = annotations
            elif name in variables:
                variables[name].annotations = annotations
            elif name in parameters:
                parameters[name].annotations = annotations

    for reaction in sbml_model.getListOfReactions():
        if (annotations := _cv_terms_to_annotations(reaction)) and (
            name := reaction.getId()
        ) in reactions:
            reactions[name].annotations = annotations


def read(file: Path, *, via_temp_file: bool = True) -> Model:
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
    model_code = _codegen(model)

    model_fn: Callable[[], Model]
    if via_temp_file:
        path = default_tmp_dir(None, remove_old_cache=False) / f"{out_name}.py"
        with path.open("w+") as f:
            f.write(model_code)
        model_fn = import_from_path(out_name, path)
    else:
        module = types.ModuleType(out_name)
        exec(model_code, module.__dict__)  # noqa: S102
        model_fn = module.create_model

    mxl_model = model_fn()
    _attach_annotations(mxl_model, file)
    return mxl_model
