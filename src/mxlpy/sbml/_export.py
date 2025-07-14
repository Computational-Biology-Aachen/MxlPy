from __future__ import annotations

import ast
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import libsbml
import numpy as np

from mxlpy.meta.source_tools import get_fn_ast
from mxlpy.sbml._data import AtomicUnit, Compartment
from mxlpy.types import Derived, InitialAssignment

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mxlpy.model import Model


__all__ = [
    "BINARY",
    "DocstringRemover",
    "IdentifierReplacer",
    "NARY",
    "RE_LAMBDA_ALGEBRAIC_MODULE_FUNC",
    "RE_LAMBDA_FUNC",
    "RE_LAMBDA_RATE_FUNC",
    "RE_TO_SBML",
    "SBML_DOT",
    "UNARY",
    "write",
]

RE_LAMBDA_FUNC = re.compile(r".*(lambda)(.+?):(.*?)")
RE_LAMBDA_RATE_FUNC = re.compile(r".*(lambda)(.+?):(.*?),")
RE_LAMBDA_ALGEBRAIC_MODULE_FUNC = re.compile(r".*(lambda)(.+?):(.*[\(\[].+[\)\]]),")
RE_TO_SBML = re.compile(r"([^0-9_a-zA-Z])")

SBML_DOT = "__SBML_DOT__"


UNARY = {
    "sqrt": libsbml.AST_FUNCTION_ROOT,
    "remainder": libsbml.AST_FUNCTION_REM,
    "abs": libsbml.AST_FUNCTION_ABS,
    "ceil": libsbml.AST_FUNCTION_CEILING,
    "sin": libsbml.AST_FUNCTION_SIN,
    "cos": libsbml.AST_FUNCTION_COS,
    "tan": libsbml.AST_FUNCTION_TAN,
    "arcsin": libsbml.AST_FUNCTION_ARCSIN,
    "arccos": libsbml.AST_FUNCTION_ARCCOS,
    "arctan": libsbml.AST_FUNCTION_ARCTAN,
    "sinh": libsbml.AST_FUNCTION_SINH,
    "cosh": libsbml.AST_FUNCTION_COSH,
    "tanh": libsbml.AST_FUNCTION_TANH,
    "arcsinh": libsbml.AST_FUNCTION_ARCSINH,
    "arccosh": libsbml.AST_FUNCTION_ARCCOSH,
    "arctanh": libsbml.AST_FUNCTION_ARCTANH,
    "log": libsbml.AST_FUNCTION_LN,
    "log10": libsbml.AST_FUNCTION_LOG,
}

BINARY = {
    "power": libsbml.AST_POWER,
}

NARY = {
    "max": libsbml.AST_FUNCTION_MAX,
    "min": libsbml.AST_FUNCTION_MIN,
}


class IdentifierReplacer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        return ast.Name(
            id=self.mapping.get(node.id, node.id),
            ctx=node.ctx,
        )


class DocstringRemover(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:  # noqa: N802
        if isinstance(const := node.value, ast.Constant) and isinstance(
            const.value, str
        ):
            return None
        return node


def _convert_unaryop(node: ast.UnaryOp) -> libsbml.ASTNode:
    operand = _convert_node(node.operand)

    match node.op:
        case ast.USub():
            op = libsbml.AST_MINUS
        case ast.Not():
            op = libsbml.AST_LOGICAL_NOT
        case _:
            raise NotImplementedError(type(node.op))

    sbml_node = libsbml.ASTNode(op)
    sbml_node.addChild(operand)
    return sbml_node


def _convert_binop(node: ast.BinOp) -> libsbml.ASTNode:
    left = _convert_node(node.left)
    right = _convert_node(node.right)

    match node.op:
        case ast.Mult():
            op = libsbml.AST_TIMES
        case ast.Add():
            op = libsbml.AST_PLUS
        case ast.Sub():
            op = libsbml.AST_MINUS
        case ast.Div():
            op = libsbml.AST_DIVIDE
        case ast.Pow():
            op = libsbml.AST_POWER
        case ast.FloorDiv():
            op = libsbml.AST_FUNCTION_QUOTIENT
        case _:
            raise NotImplementedError(type(node.op))

    sbml_node = libsbml.ASTNode(op)
    sbml_node.addChild(left)
    sbml_node.addChild(right)
    return sbml_node


def _convert_attribute(node: ast.Attribute) -> libsbml.ASTNode:
    parent = cast(ast.Name, node.value).id
    attr = node.attr

    if parent in ("math", "np", "numpy"):
        if attr == "e":
            return libsbml.ASTNode(libsbml.AST_CONSTANT_E)
        if attr == "pi":
            return libsbml.ASTNode(libsbml.AST_CONSTANT_PI)
        if attr == "inf":
            sbml_node = libsbml.ASTNode(libsbml.AST_REAL)
            sbml_node.setValue(np.inf)
            return sbml_node
        if attr == "nan":
            sbml_node = libsbml.ASTNode(libsbml.AST_REAL)
            sbml_node.setValue(np.nan)
            return sbml_node

    msg = f"{parent}.{attr}"
    raise NotImplementedError(msg)


def _convert_constant(node: ast.Constant) -> libsbml.ASTNode:
    value = node.value
    if isinstance(value, bool):
        if value:
            return libsbml.ASTNode(libsbml.AST_CONSTANT_TRUE)
        return libsbml.ASTNode(libsbml.AST_CONSTANT_FALSE)

    sbml_node = libsbml.ASTNode(libsbml.AST_REAL)
    sbml_node.setValue(value)
    return sbml_node


def _convert_ifexp(node: ast.IfExp) -> libsbml.ASTNode:
    condition = _convert_node(node.test)
    true = _convert_node(node.body)
    false = _convert_node(node.orelse)

    sbml_node = libsbml.ASTNode(libsbml.AST_FUNCTION_PIECEWISE)
    sbml_node.addChild(condition)
    sbml_node.addChild(true)
    sbml_node.addChild(false)
    return sbml_node


def _convert_direct_call(node: ast.Call) -> libsbml.ASTNode:
    func = cast(ast.Name, node.func).id

    if (typ := UNARY.get(func)) is not None:
        sbml_node = libsbml.ASTNode(typ)
        sbml_node.addChild(_convert_node(node.args[0]))
        return sbml_node
    if (typ := BINARY.get(func)) is not None:
        sbml_node = libsbml.ASTNode(typ)
        sbml_node.addChild(_convert_node(node.args[0]))
        sbml_node.addChild(_convert_node(node.args[1]))
        return sbml_node
    if (typ := NARY.get(func)) is not None:
        sbml_node = libsbml.ASTNode(typ)
        for arg in node.args:
            sbml_node.addChild(_convert_node(arg))
        return sbml_node

    # General function call
    sbml_node = libsbml.ASTNode(libsbml.AST_FUNCTION)
    for arg in node.args:
        sbml_node.addChild(_convert_node(arg))
    return sbml_node


def _convert_library_call(node: ast.Call) -> libsbml.ASTNode:
    func = cast(ast.Attribute, node.func)
    parent = cast(ast.Name, func.value).id
    attr = func.attr

    if parent in ("math", "np", "numpy"):
        if (typ := UNARY.get(attr)) is not None:
            sbml_node = libsbml.ASTNode(typ)
            sbml_node.addChild(_convert_node(node.args[0]))
            return sbml_node
        if (typ := BINARY.get(attr)) is not None:
            sbml_node = libsbml.ASTNode(typ)
            sbml_node.addChild(_convert_node(node.args[0]))
            sbml_node.addChild(_convert_node(node.args[1]))
            return sbml_node
        if (typ := NARY.get(attr)) is not None:
            sbml_node = libsbml.ASTNode(typ)
            for arg in node.args:
                sbml_node.addChild(_convert_node(arg))
            return sbml_node

    # General library call
    sbml_node = libsbml.ASTNode(libsbml.AST_FUNCTION)
    for arg in node.args:
        sbml_node.addChild(_convert_node(arg))
    return sbml_node


def _convert_call(node: ast.Call) -> libsbml.ASTNode:
    func = node.func
    if isinstance(func, ast.Name):
        return _convert_direct_call(node)
    if isinstance(func, ast.Attribute):
        return _convert_library_call(node)

    msg = f"Unknown call type: {type(func)}"
    raise NotImplementedError(msg)


def _convert_compare(node: ast.Compare) -> libsbml.ASTNode:
    # FIXME: handle cases such as x < y < z

    left = _convert_node(node.left)
    right = _convert_node(node.comparators[0])

    match node.ops[0]:
        case ast.Eq():
            op = libsbml.AST_RELATIONAL_EQ
        case ast.NotEq():
            op = libsbml.AST_RELATIONAL_NEQ
        case ast.Lt():
            op = libsbml.AST_RELATIONAL_LT
        case ast.LtE():
            op = libsbml.AST_RELATIONAL_LEQ
        case ast.Gt():
            op = libsbml.AST_RELATIONAL_GT
        case ast.GtE():
            op = libsbml.AST_RELATIONAL_GEQ
        case _:
            raise NotImplementedError(type(node.ops[0]))

    sbml_node = libsbml.ASTNode(op)
    sbml_node.addChild(left)
    sbml_node.addChild(right)
    return sbml_node


def _convert_node(node: ast.stmt | ast.expr) -> libsbml.ASTNode:
    match node:
        case ast.Return(value):
            if value is None:
                msg = "Model function cannot return `None`"
                raise ValueError(msg)
            return _convert_node(value)
        case ast.UnaryOp():
            return _convert_unaryop(node)
        case ast.BinOp():
            return _convert_binop(node)
        case ast.Name(id):
            sbml_node = libsbml.ASTNode(libsbml.AST_NAME)
            sbml_node.setName(id)
            return sbml_node
        case ast.Constant():
            return _convert_constant(node)
        case ast.Attribute():
            return _convert_attribute(node)
        case ast.IfExp():
            return _convert_ifexp(node)
        case ast.Call():
            return _convert_call(node)
        case ast.Compare():
            return _convert_compare(node)
        case _:
            raise NotImplementedError(type(node))


def _handle_body(stmts: list[ast.stmt]) -> libsbml.ASTNode:
    code = libsbml.ASTNode()
    for stmt in stmts:
        code = _convert_node(stmt)
    return code


def _tree_to_sbml(
    tree: ast.FunctionDef, args: list[str] | None = None
) -> libsbml.ASTNode:
    DocstringRemover().visit(tree)
    if args is not None:
        fn_args = [i.arg for i in tree.args.args]
        argmap = dict(zip(fn_args, args, strict=True))
        IdentifierReplacer(argmap).visit(tree)
    return _handle_body(tree.body)


def _sbmlify_fn(fn: Callable, user_args: list[str]) -> libsbml.ASTNode:
    return _tree_to_sbml(get_fn_ast(fn), args=user_args)


##########################################################################
# SBML functions
##########################################################################


def _escape_non_alphanumeric(re_sub: Any) -> str:
    """Convert a non-alphanumeric charactor to a string representation of its ascii number."""
    return f"__{ord(re_sub.group(0))}__"


def _convert_id_to_sbml(id_: str, prefix: str) -> str:
    """Add prefix if id startswith number."""
    new_id = RE_TO_SBML.sub(_escape_non_alphanumeric, id_).replace(".", SBML_DOT)
    if not new_id[0].isalpha():
        return f"{prefix}_{new_id}"
    return new_id


def _create_sbml_document() -> libsbml.SBMLDocument:
    """Create an sbml document, into which sbml information can be written.

    Returns:
        doc : libsbml.Document

    """
    # SBML namespaces
    sbml_ns = libsbml.SBMLNamespaces(3, 2)
    sbml_ns.addPackageNamespace("fbc", 2)
    # SBML document
    doc = libsbml.SBMLDocument(sbml_ns)
    doc.setPackageRequired("fbc", flag=False)
    doc.setSBOTerm("SBO:0000004")
    return doc


def _create_sbml_model(
    *,
    model_name: str,
    doc: libsbml.SBMLDocument,
    extent_units: str,
    substance_units: str,
    time_units: str,
) -> libsbml.Model:
    """Create an sbml model.

    Args:
        model_name: Name of the model.
        doc: libsbml.Document
        extent_units: Units for the extent of reactions.
        substance_units: Units for the amount of substances.
        time_units: Units for time.

    Returns:
        sbml_model : libsbml.Model

    """
    name = f"{model_name}_{datetime.now(UTC).date().strftime('%Y-%m-%d')}"
    sbml_model = doc.createModel()
    sbml_model.setId(_convert_id_to_sbml(id_=name, prefix="MODEL"))
    sbml_model.setName(_convert_id_to_sbml(id_=name, prefix="MODEL"))
    sbml_model.setTimeUnits(time_units)
    sbml_model.setExtentUnits(extent_units)
    sbml_model.setSubstanceUnits(substance_units)
    sbml_model_fbc = sbml_model.getPlugin("fbc")
    sbml_model_fbc.setStrict(True)
    return sbml_model


def _create_sbml_units(
    *,
    units: dict[str, AtomicUnit],
    sbml_model: libsbml.Model,
) -> None:
    """Create sbml units out of the meta_info.

    Args:
        units: Dictionary of units to use in the SBML file.
        sbml_model : libsbml Model

    """
    for unit_id, unit in units.items():
        sbml_definition = sbml_model.createUnitDefinition()
        sbml_definition.setId(unit_id)
        sbml_unit = sbml_definition.createUnit()
        sbml_unit.setKind(unit.kind)
        sbml_unit.setExponent(unit.exponent)
        sbml_unit.setScale(unit.scale)
        sbml_unit.setMultiplier(unit.multiplier)


def _create_sbml_compartments(
    *,
    compartments: dict[str, Compartment],
    sbml_model: libsbml.Model,
) -> None:
    for compartment_id, compartment in compartments.items():
        sbml_compartment = sbml_model.createCompartment()
        sbml_compartment.setId(compartment_id)
        sbml_compartment.setName(compartment.name)
        sbml_compartment.setConstant(compartment.is_constant)
        sbml_compartment.setSize(compartment.size)
        sbml_compartment.setSpatialDimensions(compartment.dimensions)
        sbml_compartment.setUnits(compartment.units)


def _create_sbml_variables(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the variables for the sbml model.

    Args:
        model: Model instance to export.
        sbml_model : libsbml.Model

    """
    for name, variable in model.get_raw_variables().items():
        cpd = sbml_model.createSpecies()
        cpd.setId(_convert_id_to_sbml(id_=name, prefix="CPD"))

        cpd.setConstant(False)
        cpd.setBoundaryCondition(False)
        cpd.setHasOnlySubstanceUnits(False)
        cpd.setCompartment("compartment")
        # cpd.setUnit() # FIXME: implement
        if isinstance((init := variable.initial_value), InitialAssignment):
            ar = sbml_model.createInitialAssignment()
            ar.setId(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setName(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setVariable(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setMath(_sbmlify_fn(init.fn, init.args))
        else:
            cpd.setInitialConcentration(float(init))


def _create_sbml_derived_variables(*, model: Model, sbml_model: libsbml.Model) -> None:
    for name, dv in model.get_derived_variables().items():
        sbml_ar = sbml_model.createAssignmentRule()
        sbml_ar.setId(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setName(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setVariable(_convert_id_to_sbml(id_=name, prefix="AR"))
        sbml_ar.setMath(_sbmlify_fn(dv.fn, dv.args))
        # cpd.setUnit() # FIXME: implement


def _create_derived_parameter(
    sbml_model: libsbml.Model,
    name: str,
    dp: Derived,
) -> None:
    """Create a derived parameter for the sbml model."""
    ar = sbml_model.createAssignmentRule()
    ar.setId(_convert_id_to_sbml(id_=name, prefix="AR"))
    ar.setName(_convert_id_to_sbml(id_=name, prefix="AR"))
    ar.setVariable(_convert_id_to_sbml(id_=name, prefix="AR"))
    ar.setMath(_sbmlify_fn(dp.fn, dp.args))
    # cpd.setUnit() # FIXME: implement


def _create_sbml_parameters(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the parameters for the sbml model.

    Args:
        model: Model instance to export.
        sbml_model : libsbml.Model

    """
    for name, value in model.get_raw_parameters().items():
        k = sbml_model.createParameter()
        k.setId(_convert_id_to_sbml(id_=name, prefix="PAR"))
        k.setConstant(True)

        if isinstance((init := value.value), InitialAssignment):
            ar = sbml_model.createInitialAssignment()
            ar.setId(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setName(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setVariable(_convert_id_to_sbml(id_=name, prefix="IA"))
            ar.setMath(_sbmlify_fn(init.fn, init.args))
        else:
            k.setValue(float(init))


def _create_sbml_derived_parameters(*, model: Model, sbml_model: libsbml.Model) -> None:
    for name, dp in model.get_derived_parameters().items():
        _create_derived_parameter(sbml_model, name, dp)


def _create_sbml_reactions(
    *,
    model: Model,
    sbml_model: libsbml.Model,
) -> None:
    """Create the reactions for the sbml model."""
    for name, rxn in model.get_raw_reactions().items():
        sbml_rxn = sbml_model.createReaction()
        sbml_rxn.setId(_convert_id_to_sbml(id_=name, prefix="RXN"))
        sbml_rxn.setName(name)
        sbml_rxn.setFast(False)

        for compound_id, factor in rxn.stoichiometry.items():
            match factor:
                case float() | int():
                    sref = (
                        sbml_rxn.createReactant()
                        if factor < 0
                        else sbml_rxn.createProduct()
                    )
                    sref.setSpecies(_convert_id_to_sbml(id_=compound_id, prefix="CPD"))
                    sref.setStoichiometry(abs(factor))
                    sref.setConstant(False)
                case Derived():
                    # SBML uses species references for derived stoichiometries
                    # So we need to create a assignment rule and then refer to it
                    reference = f"{compound_id}ref"
                    _create_derived_parameter(sbml_model, reference, factor)

                    sref = sbml_rxn.createReactant()
                    sref.setId(_convert_id_to_sbml(id_=reference, prefix="CPD"))
                    sref.setSpecies(_convert_id_to_sbml(id_=compound_id, prefix="CPD"))
                case _:
                    msg = f"Stoichiometry type {type(factor)} not supported"
                    raise NotImplementedError(msg)
        for compound_id in rxn.get_modifiers(model):
            sref = sbml_rxn.createModifier()
            sref.setSpecies(_convert_id_to_sbml(id_=compound_id, prefix="CPD"))

        sbml_rxn.createKineticLaw().setMath(_sbmlify_fn(rxn.fn, rxn.args))


def _model_to_sbml(
    model: Model,
    *,
    model_name: str,
    units: dict[str, AtomicUnit],
    extent_units: str,
    substance_units: str,
    time_units: str,
    compartments: dict[str, Compartment],
) -> libsbml.SBMLDocument:
    """Export model to sbml."""
    doc = _create_sbml_document()
    sbml_model = _create_sbml_model(
        model_name=model_name,
        doc=doc,
        extent_units=extent_units,
        substance_units=substance_units,
        time_units=time_units,
    )
    _create_sbml_units(units=units, sbml_model=sbml_model)
    _create_sbml_compartments(compartments=compartments, sbml_model=sbml_model)
    # Actual model components
    _create_sbml_parameters(model=model, sbml_model=sbml_model)
    _create_sbml_derived_parameters(model=model, sbml_model=sbml_model)
    _create_sbml_variables(model=model, sbml_model=sbml_model)
    _create_sbml_derived_variables(model=model, sbml_model=sbml_model)
    _create_sbml_reactions(model=model, sbml_model=sbml_model)
    return doc


def _default_compartments(
    compartments: dict[str, Compartment] | None,
) -> dict[str, Compartment]:
    if compartments is None:
        return {
            "compartment": Compartment(
                name="compartment",
                dimensions=3,
                size=1,
                units="litre",
                is_constant=True,
            )
        }
    return compartments


def _default_model_name(model_name: str | None) -> str:
    if model_name is None:
        return "model"
    return model_name


def _default_units(units: dict[str, AtomicUnit] | None) -> dict[str, AtomicUnit]:
    if units is None:
        return {
            "per_second": AtomicUnit(
                kind=libsbml.UNIT_KIND_SECOND,
                exponent=-1,
                scale=0,
                multiplier=1,
            )
        }
    return units


def write(
    model: Model,
    file: Path,
    *,
    model_name: str | None = None,
    units: dict[str, AtomicUnit] | None = None,
    compartments: dict[str, Compartment] | None = None,
    extent_units: str = "mole",
    substance_units: str = "mole",
    time_units: str = "second",
) -> Path:
    """Export a metabolic model to an SBML file.

    Args:
        model: Model instance to export.
        file: Name of the SBML file to create.
        model_name: Name of the model.
        units: Dictionary of units to use in the SBML file (default: None).
        compartments: Dictionary of compartments to use in the SBML file (default: None).
        extent_units: Units for the extent of reactions (default: "mole").
        substance_units: Units for the amount of substances (default: "mole").
        time_units: Units for time (default: "second").

    Returns:
        str | None: None if the export is successful.

    """
    doc = _model_to_sbml(
        model=model,
        model_name=_default_model_name(model_name),
        units=_default_units(units),
        extent_units=extent_units,
        substance_units=substance_units,
        time_units=time_units,
        compartments=_default_compartments(compartments),
    )

    libsbml.writeSBMLToFile(doc, str(file))
    return file
