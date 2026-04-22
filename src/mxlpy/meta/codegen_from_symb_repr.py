import logging
from collections.abc import Callable
from typing import NamedTuple, Protocol, cast

import sympy

from mxlpy import _topo
from mxlpy.meta.symbolic_repr import (
    SymbolicFn,
    model_to_symbolic_repr,
)
from mxlpy.meta.sympy_tools import sympy_to_inline_py
from mxlpy.meta.utils import _to_valid_identifier
from mxlpy.model import Model
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import InitialAssignment

__all__ = [
    "Codegen",
    "ExprTemplate",
    "FnDeclTemplate",
    "ListTemplate",
    "ModelCode",
    "NormalizedSymbolicModel",
    "ReturnTemplate",
    "TupleTemplate",
    "VariableAssignmentTemplate",
    "VariableUnpackingTemplate",
    "generate_model_code_py",
]

_LOGGER = logging.getLogger()


class NormalizedSymbolicModel(NamedTuple):
    body: list[tuple[str, sympy.Expr]]
    extended: list[tuple[str, sympy.Expr]]
    diff_eqs: dict[str, sympy.Expr]
    inits: list[tuple[str, sympy.Expr]]
    free_pars: list[str]


class Codegen(NamedTuple):
    imports: str
    model: str
    derived: str
    inits: str


###############################################################################
# Utils
###############################################################################


def _get_dependencies_and_leaves(model: Model, requested: set[str]):
    leaves = (
        {
            k
            for k, v in model.get_raw_variables(as_copy=False).items()
            if not isinstance(v.initial_value, InitialAssignment)
        }
        | {
            k
            for k, v in model.get_raw_parameters(as_copy=False).items()
            if not isinstance(v.value, InitialAssignment)
        }
        | {"time"}
    )
    dependees = (
        {
            k: set(ia.args)
            for k, v in model.get_raw_variables(as_copy=False).items()
            if isinstance(ia := v.initial_value, InitialAssignment)
        }
        | {
            k: set(ia.args)
            for k, v in model.get_raw_parameters(as_copy=False).items()
            if isinstance(ia := v.value, InitialAssignment)
        }
        | {k: set(v.args) for k, v in model.get_raw_derived(as_copy=False).items()}
        | {k: set(v.args) for k, v in model.get_raw_reactions(as_copy=False).items()}
        | {k: set(v.args) for k, v in model.get_raw_readouts(as_copy=False).items()}
    )
    for v in model.get_raw_surrogates(as_copy=False).values():
        args = set(v.args)
        dependees.update(dict.fromkeys(v.outputs, args))

    return _topo.get_all_dependencies_of(requested, leaves, dependees)


def _get_order(self: Model):
    base_parameter_values: dict[str, float] = {
        k: val
        for k, v in self._parameters.items()
        if not isinstance(val := v.value, InitialAssignment)
    }
    base_variable_values: dict[str, float] = {
        k: init
        for k, v in self._variables.items()
        if not isinstance(init := v.initial_value, InitialAssignment)
    }
    initial_assignments: dict[str, InitialAssignment] = {
        k: init
        for k, v in self._variables.items()
        if isinstance(init := v.initial_value, InitialAssignment)
    } | {
        k: init
        for k, v in self._parameters.items()
        if isinstance(init := v.value, InitialAssignment)
    }

    # Sort derived & reactions
    available = (
        set(base_parameter_values)
        | set(base_variable_values)
        | set(self._data)
        | {"time"}
    )
    to_sort = (
        initial_assignments  # wrap this line
        | self._derived
        | self._reactions
        | self._surrogates
        | self._readouts
    )

    return _topo.sort_all(
        available=available,
        elements=[
            _topo.Dependency(name=k, required=set(v.args), provided={k})
            if not isinstance(v, AbstractSurrogate)
            else _topo.Dependency(name=k, required=set(v.args), provided=set(v.outputs))
            for k, v in to_sort.items()
        ],
    )


def _normalized_symbolic_model(
    model: Model,
    free_parameters: list[str],
    derived_to_calculate: list[str] | None,
) -> NormalizedSymbolicModel:
    sr = model_to_symbolic_repr(model)
    order = _get_order(model)

    # linearize
    inits = {}
    assignments = {}
    for k, v in sr.variables.items():
        if isinstance(val := v.value, SymbolicFn):
            inits[k] = val.expr
        else:
            inits[k] = val
    for k, v in sr.parameters.items():
        if k in free_parameters:
            continue
        if isinstance(val := v.value, SymbolicFn):
            inits[k] = val.expr
        else:
            assignments[k] = val
    for k, v in sr.derived.items():
        assignments[k] = v.expr
    for k, v in sr.reactions.items():
        assignments[k] = v.fn.expr
    for v in sr.surrogates.values():
        for output, fn in zip(v.outputs, v.fns, strict=True):
            assignments[output] = fn.expr
    for k, v in sr.readouts.items():
        assignments[k] = v.expr

    ##################################
    # Main fn body
    ##################################
    linear = []
    readouts = []  # readouts
    for i in order:
        if i in inits or i in free_parameters:
            continue
        if i in sr.readouts:
            readouts.append((i, assignments[i]))
            continue
        if i not in sr.surrogates:
            linear.append((i, assignments[i]))
        else:
            linear.extend([(j, assignments[j]) for j in sr.surrogates[i].outputs])

    if derived_to_calculate is None:
        derived = linear + readouts
    else:
        deps = _get_dependencies_and_leaves(model, set(derived_to_calculate))
        derived = [i for i in linear if i[0] in deps] + readouts

    ##################################
    # Diff eqs
    ##################################
    diff_eqs: dict[str, sympy.Expr] = {i: sympy.Integer(0) for i in sr.variables}
    for rate, rxn in sr.reactions.items():
        for cpd, factor in rxn.stoichiometry.items():
            if isinstance(factor, SymbolicFn):
                diff_eqs[cpd] += sympy.Symbol(rate) * factor.expr  # pyright: ignore[reportOperatorIssue]
            else:
                diff_eqs[cpd] += sympy.Symbol(rate) * factor  # pyright: ignore[reportOperatorIssue]
    for surrogate in sr.surrogates.values():
        for rate, rxn in surrogate.stoichiometry.items():
            for cpd, factor in rxn.items():
                if isinstance(factor, SymbolicFn):
                    diff_eqs[cpd] += sympy.Symbol(rate) * factor.expr  # pyright: ignore[reportOperatorIssue]
                else:
                    diff_eqs[cpd] += sympy.Symbol(rate) * factor  # pyright: ignore[reportOperatorIssue]

    ##################################
    # Initial variables and parameters
    ##################################
    _init_dependencies = (
        _get_dependencies_and_leaves(model, set(inits)) | set(sr.variables)
    ) - set(free_parameters)

    _init_order = [i for i in order if i in _init_dependencies]

    init_eqs: list[tuple[str, sympy.Expr]] = [
        (i, assignments.get(i, inits.get(i, sympy.Float("nan")))) for i in _init_order
    ]
    return NormalizedSymbolicModel(
        body=linear,
        extended=derived,
        diff_eqs=diff_eqs,
        inits=init_eqs,
        free_pars=sorted(
            free_parameters
            # {i for i in (_init_dependencies ^ set(inits)) if i in sr.parameters}
            # | set(free_parameters)
        ),
    )


###############################################################################
# Generate initial conditions
###############################################################################

###############################################################################
# Generate components (derived + readouts)
###############################################################################


###############################################################################
# Full codegen
###############################################################################


class FnDeclTemplate(Protocol):
    def __call__(
        self,
        name: str,
        args: list[tuple[str, str]],
        return_type: str,
    ) -> str: ...


class VariableUnpackingTemplate(Protocol):
    def __call__(self, variables: list[str]) -> str: ...


class VariableAssignmentTemplate(Protocol):
    def __call__(self, name: str, value: str) -> str: ...


class ExprTemplate(Protocol):
    def __call__(self, expr: sympy.Expr) -> str: ...


class ListTemplate(Protocol):
    def __call__(self, elements: list[str]) -> str: ...


class TupleTemplate(Protocol):
    def __call__(self, elements: list[str]) -> str: ...


class ReturnTemplate(Protocol):
    def __call__(self, variables: list[str]) -> str: ...


class ModelCode(NamedTuple):
    model: str
    initial_conditions: str
    derived: str


def _generate_model_code(
    model: Model,
    *,
    free_parameters: list[str],
    derived_to_calculate: list[str] | None,
    fn_template: FnDeclTemplate,
    assignment_template: VariableAssignmentTemplate,
    expr_template: ExprTemplate,
    variable_unpacking: VariableUnpackingTemplate,
    return_formatter: Callable[[list[str]], str],
    list_formatter: ListTemplate,
    imports: list[str],
    name_map: dict[str, str],
    custom_fns: dict[str, sympy.Expr],
) -> Codegen:
    extended_returns = (
        list(
            model.get_derived_parameter_names()
            + model.get_derived_variable_names()
            + model.get_reaction_names()
            + model.get_surrogate_output_names()
            + model.get_readout_names()
        )
        if derived_to_calculate is None
        else derived_to_calculate
    )

    nsm = _normalized_symbolic_model(
        model,
        free_parameters=free_parameters,
        derived_to_calculate=derived_to_calculate,
    )

    sympy_name_map = {sympy.Symbol(k): sympy.Symbol(v) for k, v in name_map.items()}

    variable_order = list(model.get_initial_conditions())

    body_src = "\n".join(
        [
            assignment_template(
                name_map[name],
                expr_template(cast(sympy.Expr, expr.subs(sympy_name_map))),
            )
            for name, expr in nsm.body
        ]
    )
    extended_src = "\n".join(
        [
            assignment_template(
                name_map[name],
                expr_template(cast(sympy.Expr, expr.subs(sympy_name_map))),
            )
            for name, expr in nsm.extended
        ]
    )
    diff_eq_src = "\n".join(
        [
            assignment_template(
                f"d{name_map[name]}dt",
                expr_template(
                    cast(sympy.Expr, nsm.diff_eqs[name].subs(sympy_name_map))
                ),
            )
            # Important: same ordering!
            for name in variable_order
        ]
    )
    init_src = "\n".join(
        [
            assignment_template(
                name_map[name],
                expr_template(cast(sympy.Expr, expr.subs(sympy_name_map))),
            )
            for name, expr in nsm.inits
        ]
    )

    _fn_args = [
        ("time", "float"),
        ("variables", "Iterable[float]"),
        *((name_map[i], "float") for i in nsm.free_pars),
    ]

    model_src = "\n".join(
        [
            fn_template(
                "model",
                _fn_args,
                "Iterable[float]",
            ),
            variable_unpacking([name_map[i] for i in variable_order]),
            body_src,
            diff_eq_src,
            return_formatter([f"d{name_map[x]}dt" for x in variable_order]),
        ]
    )
    derived_src = "\n".join(
        [
            fn_template(
                "derived",
                _fn_args,
                "Iterable[float]",
            ),
            variable_unpacking([name_map[i] for i in variable_order]),
            extended_src,
            return_formatter([name_map[name] for name in extended_returns]),
        ]
    )

    init_vars = list_formatter([name_map[name] for name in variable_order])
    init_pars = list_formatter(
        [name_map[i] for i, _ in nsm.inits if i not in variable_order]
    )

    init_src = "\n".join(
        [
            fn_template(
                "inits",
                [(name_map[k], "float") for k in nsm.free_pars],
                "tuple[Iterable[float], Iterable[float]]",
            ),
            init_src,
            return_formatter([init_vars, init_pars]),
        ]
    )
    return Codegen(
        imports="\n".join(imports),
        model=model_src,
        derived=derived_src,
        inits=init_src,
    )


def generate_model_code_py(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr] | None = None,
    typed: bool = False,
):
    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:
        return (
            f"def {name}({', '.join(f'{k}: {t}' for k, t in args)}) -> {return_type}:"
        )

    def variable_unpacking(variables: list[str]) -> str:
        return f"    {', '.join(variables)} = variables"

    def list_template(elements: list[str]) -> str:
        return f"[{', '.join(elements)}]"

    def assignment_template(name: str, value: str) -> str:
        if typed:
            return f"    {name}: float = {value}"
        return f"    {name} = {value}"

    def return_template(variables: list[str]) -> str:
        return f"    return {', '.join(variables) or '()'}"

    return _generate_model_code(
        model,
        free_parameters=[] if free_parameters is None else free_parameters,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_py,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=custom_fns if custom_fns is not None else {},
        imports=[
            "import math",
            "from collections.abc import Iterable",
        ],
        name_map={name: _to_valid_identifier(name) for name in model.ids},
    )
