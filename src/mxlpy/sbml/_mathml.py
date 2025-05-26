from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._name_conversion import _name_to_py
from ._unit_conversion import get_ast_types

if TYPE_CHECKING:
    from libsbml import ASTNode

__all__ = [
    "AST_TYPES",
    "handle_ast_constant_e",
    "handle_ast_constant_false",
    "handle_ast_constant_pi",
    "handle_ast_constant_true",
    "handle_ast_divide",
    "handle_ast_divide_int",
    "handle_ast_function",
    "handle_ast_function_abs",
    "handle_ast_function_ceiling",
    "handle_ast_function_delay",
    "handle_ast_function_exp",
    "handle_ast_function_factorial",
    "handle_ast_function_floor",
    "handle_ast_function_ln",
    "handle_ast_function_log",
    "handle_ast_function_max",
    "handle_ast_function_min",
    "handle_ast_function_piecewise",
    "handle_ast_function_power",
    "handle_ast_function_rate_of",
    "handle_ast_function_rem",
    "handle_ast_function_root",
    "handle_ast_integer",
    "handle_ast_lambda",
    "handle_ast_logical_and",
    "handle_ast_logical_implies",
    "handle_ast_logical_not",
    "handle_ast_logical_or",
    "handle_ast_logical_xor",
    "handle_ast_minus",
    "handle_ast_name",
    "handle_ast_name_avogadro",
    "handle_ast_name_time",
    "handle_ast_originates_in_package",
    "handle_ast_plus",
    "handle_ast_rational",
    "handle_ast_real",
    "handle_ast_relational_eq",
    "handle_ast_relational_geq",
    "handle_ast_relational_gt",
    "handle_ast_relational_leq",
    "handle_ast_relational_lt",
    "handle_ast_relational_neq",
    "handle_ast_times",
    "handle_ast_trigonometric_arc_cos",
    "handle_ast_trigonometric_arc_cosh",
    "handle_ast_trigonometric_arc_cot",
    "handle_ast_trigonometric_arc_coth",
    "handle_ast_trigonometric_arc_csc",
    "handle_ast_trigonometric_arc_csch",
    "handle_ast_trigonometric_arc_sec",
    "handle_ast_trigonometric_arc_sech",
    "handle_ast_trigonometric_arc_sin",
    "handle_ast_trigonometric_arc_sinh",
    "handle_ast_trigonometric_arc_tan",
    "handle_ast_trigonometric_arc_tanh",
    "handle_ast_trigonometric_cos",
    "handle_ast_trigonometric_cosh",
    "handle_ast_trigonometric_cot",
    "handle_ast_trigonometric_coth",
    "handle_ast_trigonometric_csc",
    "handle_ast_trigonometric_csch",
    "handle_ast_trigonometric_sec",
    "handle_ast_trigonometric_sech",
    "handle_ast_trigonometric_sin",
    "handle_ast_trigonometric_sinh",
    "handle_ast_trigonometric_tan",
    "handle_ast_trigonometric_tanh",
    "parse_sbml_math",
]


AST_TYPES = get_ast_types()


def handle_ast_constant_e(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return "math.e"


def handle_ast_constant_false(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return "False"


def handle_ast_constant_true(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return "True"


def handle_ast_constant_pi(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return "math.pi"


def handle_ast_divide(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    bracketed_children = []
    for child in children:
        if len(child.split("+")) > 1:
            bracketed_children.append(f"({child})")
        else:
            bracketed_children.append(child)
    return " / ".join(bracketed_children)


def handle_ast_divide_int(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    bracketed_children = []
    for child in children:
        if len(child.split("+")) > 1:
            bracketed_children.append(f"({child})")
        else:
            bracketed_children.append(child)
    return " // ".join(bracketed_children)


def handle_ast_function(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    node_name = node.getName()
    arguments = ", ".join(children)
    return f"{node_name}({arguments})"


def handle_ast_function_abs(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.abs({child})"


def handle_ast_function_ceiling(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.ceil({child})"


def handle_ast_function_delay(node: ASTNode, func_arguments: list[str]) -> str:
    raise NotImplementedError


def handle_ast_function_exp(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.exp({child})"


def handle_ast_function_factorial(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"scipy.special.factorial({child})"


def handle_ast_function_floor(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.floor({child})"


def handle_ast_function_ln(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.log({child})"


def handle_ast_function_log(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.log10({child})"


def handle_ast_function_max(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    args = ", ".join(children)
    return f"np.max([{args}])"


def handle_ast_function_min(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    args = ", ".join(children)
    return f"np.min({args})"


def handle_ast_function_piecewise(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) == 3:  # noqa: PLR2004
        condition = children[1]
        x = children[0]
        y = children[2]
        return f"np.where({condition}, {x}, {y})"
    return f"({children[0]} if {children[1]} else 0.0)"


def handle_ast_function_power(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    left_subchildren = node.getChild(0).getChild(0) is not None
    right_subchildren = node.getChild(1).getChild(0) is not None
    if left_subchildren and right_subchildren:
        return f"({children[0]}) ** ({children[1]})"
    if left_subchildren:
        return f"({children[0]}) ** {children[1]}"
    return f"{children[0]} ** ({children[1]})"


def handle_ast_function_rate_of(node: ASTNode, func_arguments: list[str]) -> str:
    raise NotImplementedError


def handle_ast_function_root(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.sqrt({child})"


def handle_ast_function_rem(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    args = ", ".join(children)
    return f"np.remainder({args})"


def handle_ast_integer(
    node: ASTNode,
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return str(int(node.getValue()))


def handle_ast_lambda(node: ASTNode, func_arguments: list[str]) -> str:
    num_b_vars = node.getNumBvars()
    num_children = node.getNumChildren()
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(num_b_vars, num_children)
    ]
    return ", ".join(children)


def handle_ast_logical_and(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) == 0:
        return "and"
    if len(children) == 1:
        return "and"
    args = " and ".join(children)
    return f"({args})"


def handle_ast_logical_implies(
    node: ASTNode,
    func_arguments: list[str],
) -> str:
    raise NotImplementedError


def handle_ast_logical_not(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) == 0:
        return "not"
    if len(children) == 1:
        return f"not({children[0]})"
    args = " not ".join(children)
    return f"({args})"


def handle_ast_logical_or(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) == 0:
        return "or"
    if len(children) == 1:
        return "or"
    args = " or ".join(children)
    return f"({args})"


def handle_ast_logical_xor(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) == 0:
        return "^"
    if len(children) == 1:
        return "^"
    children = [f"({i})" for i in children]
    args = " ^ ".join(children)
    return f"({args})"


def handle_ast_minus(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if node.getNumChildren() == 1:
        child = node.getChild(0)
        child_str = children[0]
        if child.getChild(0) is not None:
            return f"-({child_str})"
        return f"-{child_str}"
    return " - ".join(children)


def handle_ast_name(node: ASTNode, func_arguments: list[str]) -> str:
    name: str = _name_to_py(node.getName())
    func_arguments.append(name)
    return name


def handle_ast_name_avogadro(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return "6.02214179e+23"


def handle_ast_name_time(
    node: ASTNode,  # noqa: ARG001
    func_arguments: list[str],
) -> str:
    func_arguments.append("time")
    return "time"


def handle_ast_originates_in_package(node: ASTNode, func_arguments: list[str]) -> str:
    raise NotImplementedError


def handle_ast_plus(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    return " + ".join(children)


def handle_ast_rational(
    node: ASTNode,
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    return str(node.getValue())


def handle_ast_real(
    node: ASTNode,
    func_arguments: list[str],  # noqa: ARG001
) -> str:
    value = str(node.getValue())
    if value == "inf":
        return "np.inf"
    if value == "nan":
        return "np.nan"
    return value


def handle_ast_relational_eq(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} == {children[1]}"


def handle_ast_relational_geq(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} >= {children[1]}"


def handle_ast_relational_gt(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} > {children[1]}"


def handle_ast_relational_leq(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} <= {children[1]}"


def handle_ast_relational_lt(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} < {children[1]}"


def handle_ast_relational_neq(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    if len(children) > 2:  # noqa: PLR2004
        raise NotImplementedError
    return f"{children[0]} != {children[1]}"


def handle_ast_times(node: ASTNode, func_arguments: list[str]) -> str:
    children = [
        _handle_ast_node(node=node.getChild(i), func_arguments=func_arguments)
        for i in range(node.getNumChildren())
    ]
    bracketed_children = []
    for child in children:
        if len(child.split("+")) > 1:
            bracketed_children.append(f"({child})")
        else:
            bracketed_children.append(child)
    return " * ".join(bracketed_children)


###############################################################################
# Base
###############################################################################


def handle_ast_trigonometric_sin(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.sin({child})"


def handle_ast_trigonometric_cos(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.cos({child})"


def handle_ast_trigonometric_tan(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.tan({child})"


def handle_ast_trigonometric_sec(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.cos({child})"


def handle_ast_trigonometric_csc(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.sin({child})"


def handle_ast_trigonometric_cot(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.tan({child})"


###############################################################################
# Inverse
###############################################################################


def handle_ast_trigonometric_arc_sin(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arcsin({child})"


def handle_ast_trigonometric_arc_cos(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arccos({child})"


def handle_ast_trigonometric_arc_tan(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arctan({child})"


def handle_ast_trigonometric_arc_cot(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arctan(1 / ({child}))"


def handle_ast_trigonometric_arc_sec(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arccos(1 / ({child}))"


def handle_ast_trigonometric_arc_csc(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arcsin(1 / ({child}))"


###############################################################################
# Hyperbolic
###############################################################################


def handle_ast_trigonometric_sinh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.sinh({child})"


def handle_ast_trigonometric_cosh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.cosh({child})"


def handle_ast_trigonometric_tanh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.tanh({child})"


def handle_ast_trigonometric_sech(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.cosh({child})"


def handle_ast_trigonometric_csch(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.sinh({child})"


def handle_ast_trigonometric_coth(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"1 / np.tanh({child})"


###############################################################################
# Hyperbolic - inverse
###############################################################################


def handle_ast_trigonometric_arc_sinh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arcsinh({child})"


def handle_ast_trigonometric_arc_cosh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arccosh({child})"


def handle_ast_trigonometric_arc_tanh(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arctanh({child})"


def handle_ast_trigonometric_arc_csch(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arcsinh(1 / {child})"


def handle_ast_trigonometric_arc_sech(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arccosh(1 / {child})"


def handle_ast_trigonometric_arc_coth(node: ASTNode, func_arguments: list[str]) -> str:
    child = _handle_ast_node(node=node.getChild(0), func_arguments=func_arguments)
    return f"np.arctanh(1 / {child})"


def _handle_ast_node(node: ASTNode, func_arguments: list[str]) -> str:
    commands = {
        "AST_CONSTANT_E": handle_ast_constant_e,
        "AST_CONSTANT_FALSE": handle_ast_constant_false,
        "AST_CONSTANT_PI": handle_ast_constant_pi,
        "AST_CONSTANT_TRUE": handle_ast_constant_true,
        "AST_DIVIDE": handle_ast_divide,
        "AST_FUNCTION": handle_ast_function,
        "AST_FUNCTION_ABS": handle_ast_function_abs,
        "AST_FUNCTION_ARCCOS": handle_ast_trigonometric_arc_cos,
        "AST_FUNCTION_ARCCOSH": handle_ast_trigonometric_arc_cosh,
        "AST_FUNCTION_ARCCOT": handle_ast_trigonometric_arc_cot,
        "AST_FUNCTION_ARCCOTH": handle_ast_trigonometric_arc_coth,
        "AST_FUNCTION_ARCCSC": handle_ast_trigonometric_arc_csc,
        "AST_FUNCTION_ARCCSCH": handle_ast_trigonometric_arc_csch,
        "AST_FUNCTION_ARCSEC": handle_ast_trigonometric_arc_sec,
        "AST_FUNCTION_ARCSECH": handle_ast_trigonometric_arc_sech,
        "AST_FUNCTION_ARCSIN": handle_ast_trigonometric_arc_sin,
        "AST_FUNCTION_ARCSINH": handle_ast_trigonometric_arc_sinh,
        "AST_FUNCTION_ARCTAN": handle_ast_trigonometric_arc_tan,
        "AST_FUNCTION_ARCTANH": handle_ast_trigonometric_arc_tanh,
        "AST_FUNCTION_CEILING": handle_ast_function_ceiling,
        "AST_FUNCTION_COS": handle_ast_trigonometric_cos,
        "AST_FUNCTION_COSH": handle_ast_trigonometric_cosh,
        "AST_FUNCTION_COT": handle_ast_trigonometric_cot,
        "AST_FUNCTION_COTH": handle_ast_trigonometric_coth,
        "AST_FUNCTION_CSC": handle_ast_trigonometric_csc,
        "AST_FUNCTION_CSCH": handle_ast_trigonometric_csch,
        "AST_FUNCTION_DELAY": handle_ast_function_delay,
        "AST_FUNCTION_EXP": handle_ast_function_exp,
        "AST_FUNCTION_FACTORIAL": handle_ast_function_factorial,
        "AST_FUNCTION_FLOOR": handle_ast_function_floor,
        "AST_FUNCTION_LN": handle_ast_function_ln,
        "AST_FUNCTION_LOG": handle_ast_function_log,
        "AST_FUNCTION_MAX": handle_ast_function_max,
        "AST_FUNCTION_MIN": handle_ast_function_min,
        "AST_FUNCTION_PIECEWISE": handle_ast_function_piecewise,
        "AST_FUNCTION_POWER": handle_ast_function_power,
        "AST_FUNCTION_QUOTIENT": handle_ast_divide_int,
        "AST_FUNCTION_RATE_OF": handle_ast_function_rate_of,
        "AST_FUNCTION_ROOT": handle_ast_function_root,
        "AST_FUNCTION_REM": handle_ast_function_rem,
        "AST_FUNCTION_SEC": handle_ast_trigonometric_sec,
        "AST_FUNCTION_SECH": handle_ast_trigonometric_sech,
        "AST_FUNCTION_SIN": handle_ast_trigonometric_sin,
        "AST_FUNCTION_SINH": handle_ast_trigonometric_sinh,
        "AST_FUNCTION_TAN": handle_ast_trigonometric_tan,
        "AST_FUNCTION_TANH": handle_ast_trigonometric_tanh,
        "AST_INTEGER": handle_ast_integer,
        "AST_LAMBDA": handle_ast_lambda,
        "AST_LOGICAL_AND": handle_ast_logical_and,
        "AST_LOGICAL_IMPLIES": handle_ast_logical_implies,
        "AST_LOGICAL_NOT": handle_ast_logical_not,
        "AST_LOGICAL_OR": handle_ast_logical_or,
        "AST_LOGICAL_XOR": handle_ast_logical_xor,
        "AST_MINUS": handle_ast_minus,
        "AST_NAME": handle_ast_name,
        "AST_NAME_AVOGADRO": handle_ast_name_avogadro,
        "AST_NAME_TIME": handle_ast_name_time,
        "AST_ORIGINATES_IN_PACKAGE": handle_ast_originates_in_package,
        "AST_PLUS": handle_ast_plus,
        "AST_POWER": handle_ast_function_power,
        "AST_RATIONAL": handle_ast_rational,
        "AST_REAL": handle_ast_real,
        "AST_REAL_E": handle_ast_real,
        "AST_RELATIONAL_EQ": handle_ast_relational_eq,
        "AST_RELATIONAL_GEQ": handle_ast_relational_geq,
        "AST_RELATIONAL_GT": handle_ast_relational_gt,
        "AST_RELATIONAL_LEQ": handle_ast_relational_leq,
        "AST_RELATIONAL_LT": handle_ast_relational_lt,
        "AST_RELATIONAL_NEQ": handle_ast_relational_neq,
        "AST_TIMES": handle_ast_times,
    }
    return commands[AST_TYPES[node.getType()]](node=node, func_arguments=func_arguments)


def parse_sbml_math(node: ASTNode) -> tuple[str, list[str]]:
    func_arguments: list[Any] = []
    body = _handle_ast_node(node=node, func_arguments=func_arguments)
    return body, sorted(set(func_arguments))
