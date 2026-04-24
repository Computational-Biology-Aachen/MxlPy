"""Generate multi-language model code from a symbolic model representation."""

import logging
from collections.abc import Callable
from typing import NamedTuple, Protocol, cast

import sympy

from mxlpy import _topo
from mxlpy.meta.symbolic_repr import (
    SymbolicFn,
    SymbolicParameter,
    SymbolicReaction,
    SymbolicSurrogate,
    SymbolicVariable,
    model_to_symbolic_repr,
)
from mxlpy.meta.sympy_tools import (
    sympy_to_inline_cxx,
    sympy_to_inline_js,
    sympy_to_inline_julia,
    sympy_to_inline_matlab,
    sympy_to_inline_mxlweb,
    sympy_to_inline_py,
    sympy_to_inline_rust,
)
from mxlpy.meta.utils import valid_identifier, valid_tex_identifier
from mxlpy.model import Model
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import InitialAssignment

__all__ = [
    "Codegen",
    "ExprTemplate",
    "FnDeclTemplate",
    "ListTemplate",
    "NormalizedSymbolicModel",
    "ReturnTemplate",
    "TupleTemplate",
    "VariableAssignmentTemplate",
    "VariableUnpackingTemplate",
    "generate_model_code_cpp",
    "generate_model_code_jl",
    "generate_model_code_matlab",
    "generate_model_code_py",
    "generate_model_code_rs",
    "generate_model_code_ts",
    "generate_mxlweb_code",
]

_LOGGER = logging.getLogger()


class NormalizedSymbolicModel(NamedTuple):
    """Symbolic model normalized into flat assignment lists for code generation."""

    body: list[tuple[str, sympy.Expr]]
    extended: list[tuple[str, sympy.Expr]]
    diff_eqs: dict[str, sympy.Expr]
    inits: list[tuple[str, sympy.Expr]]
    free_pars: list[str]


class Codegen(NamedTuple):
    """Generated code split into four sections ready for emission."""

    imports: str
    model: str
    derived: str
    inits: str


class FnDeclTemplate(Protocol):
    """Protocol for generating a language-specific function declaration string."""

    def __call__(
        self,
        name: str,
        args: list[tuple[str, str]],
        return_type: str,
    ) -> str:
        """Generate a function declaration line.

        Parameters
        ----------
        name
            Function name.
        args
            (parameter_name, type_string) pairs.
        return_type
            Return type as a Python-domain type string.

        Returns
        -------
        str
            Function declaration in the target language.

        """
        ...


class VariableUnpackingTemplate(Protocol):
    """Protocol for generating a statement that unpacks the variables array."""

    def __call__(self, variables: list[str]) -> str:
        """Return a statement that destructures *variables* into named locals."""
        ...


class VariableAssignmentTemplate(Protocol):
    """Protocol for generating a single variable assignment statement."""

    def __call__(self, name: str, value: str) -> str:
        """Return an assignment statement ``name = value`` in the target language."""
        ...


class ExprTemplate(Protocol):
    """Protocol for converting a SymPy expression to an inline code string."""

    def __call__(self, expr: sympy.Expr) -> str:
        """Render *expr* as an inline expression in the target language."""
        ...


class ListTemplate(Protocol):
    """Protocol for rendering a list/array literal in the target language."""

    def __call__(self, elements: list[str]) -> str:
        """Return an array literal containing *elements*."""
        ...


class TupleTemplate(Protocol):
    """Protocol for rendering a tuple literal in the target language."""

    def __call__(self, elements: list[str]) -> str:
        """Return a tuple literal containing *elements*."""
        ...


class ReturnTemplate(Protocol):
    """Protocol for generating a return statement in the target language."""

    def __call__(self, variables: list[str]) -> str:
        """Return a return statement yielding *variables*."""
        ...


###############################################################################
# Utils
###############################################################################


def _get_dependencies_and_leaves(model: Model, requested: set[str]) -> set[str]:
    """Return all names reachable from *requested* via the model's dependency graph."""
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


def _get_order(self: Model) -> list[str]:
    """Return topological sort order of all model components."""
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
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> NormalizedSymbolicModel:
    sr = model_to_symbolic_repr(model, custom_fns=custom_fns)
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


def _get_extended_returns(
    model: Model,
    derived_to_calculate: list[str] | None,
) -> list[str]:
    return (
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
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> Codegen:
    extended_returns = _get_extended_returns(model, derived_to_calculate)

    nsm = _normalized_symbolic_model(
        model,
        free_parameters=free_parameters,
        derived_to_calculate=derived_to_calculate,
        custom_fns=custom_fns,
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


def generate_mxlweb_code(
    model: Model,
    *,
    tex_names: dict[str, str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    sliders: dict[str, dict[str, str]] | None = None,
) -> str:
    """Generate TypeScript source for the mxlweb browser simulator.

    Parameters
    ----------
    model
        Model to generate code for.
    tex_names
        Optional mapping of component names to LaTeX display names.
    custom_fns
        Optional custom sympy expressions to substitute for functions.
    sliders
        Optional mapping of parameter names to slider config dicts with
        ``min``, ``max``, and ``step`` keys.

    Returns
    -------
    str
        TypeScript source that constructs a ``ModelBuilder`` for mxlweb.

    """
    sliders = {} if sliders is None else sliders
    custom_fns = {} if custom_fns is None else custom_fns
    tex_names = {} if tex_names is None else tex_names

    name_map = {name: valid_identifier(name) for name in model.ids}
    subs = {sympy.Symbol(k): sympy.Symbol(v) for k, v in name_map.items()}
    sr = model_to_symbolic_repr(model, custom_fns=custom_fns)
    used: set[str] = set()

    def _gen_slider(k: str) -> str:
        if (s := sliders.get(k)) is not None:
            return (
                f"        slider: {{\n"
                f'          min: "{s["min"]}",\n'
                f'          max: "{s["max"]}",\n'
                f'          step: "{s["step"]}",\n'
                f"        }},\n"
            )
        return ""

    def _gen_var(
        k: str,
        el: SymbolicVariable,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        texName = (
            valid_tex_identifier(k)
            if (texName := tex_names.get(k)) is None
            else texName
        )

        # FIXME: rewrite mxlweb to only take `Base`
        # then you can use new Num instead of float
        if isinstance(var := el.value, SymbolicFn):
            value = sympy_to_inline_mxlweb(var.expr, used, subs)
        elif isinstance(var, sympy.Float):
            value = str(float(var))
        else:
            value = sympy_to_inline_mxlweb(var, used, subs)

        return (
            f'      .addVariable("{name_map[k]}", {{\n'
            f"        value: {value},\n"
            f"        texName: {texName!r},\n{_gen_slider(k)}"
            f"      }})"
        )

    def _gen_par(
        k: str,
        el: SymbolicParameter,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        texName = (
            valid_tex_identifier(k)
            if (texName := tex_names.get(k)) is None
            else texName
        )

        # FIXME: rewrite mxlweb to only take `Base`
        # then you can use new Num instead of float
        if isinstance(var := el.value, SymbolicFn):
            value = sympy_to_inline_mxlweb(var.expr, used, subs)
        elif isinstance(var, sympy.Float):
            value = str(float(var))
        else:
            value = sympy_to_inline_mxlweb(var, used, subs)
        return (
            f'      .addParameter("{name_map[k]}", {{\n'
            f"        value: {value},\n"
            f"        texName: {texName!r},\n{_gen_slider(k)}"
            f"      }})"
        )

    def _gen_der(
        k: str,
        el: SymbolicFn,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        texName = (
            valid_tex_identifier(k)
            if (texName := tex_names.get(k)) is None
            else texName
        )
        value = sympy_to_inline_mxlweb(el.expr, used, subs)
        return (
            f'      .addAssignment("{name_map[k]}", {{\n'
            f"        fn: {value},\n"
            f"        texName: {texName!r},\n"
            f"      }})"
        )

    def _gen_rdo(
        k: str,
        el: SymbolicFn,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        # FIXME: add readout to mxlweb
        texName = (
            valid_tex_identifier(k)
            if (texName := tex_names.get(k)) is None
            else texName
        )
        value = sympy_to_inline_mxlweb(el.expr, used, subs)
        return (
            f'      .addAssignment("{name_map[k]}", {{\n'
            f"        fn: {value},\n"
            f"        texName: {texName!r},\n"
            f"      }})"
        )

    def _gen_stoich(
        k: str,
        el: sympy.Float | SymbolicFn,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        value = (
            sympy_to_inline_mxlweb(el.expr, used, subs)
            if isinstance(el, SymbolicFn)
            else sympy_to_inline_mxlweb(el, used, subs)
        )
        return f'{{ name: "{name_map[k]}", value: {value} }}'

    def _gen_rxn(
        k: str,
        el: SymbolicReaction,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        texName = (
            valid_tex_identifier(k)
            if (texName := tex_names.get(k)) is None
            else texName
        )
        fn = sympy_to_inline_mxlweb(el.fn.expr, used, subs)
        stoich = f"[{', '.join(_gen_stoich(k, v, used, subs) for k, v in el.stoichiometry.items())}]"

        return (
            f'      .addReaction("{name_map[k]}", {{\n'
            f"        fn: {fn},\n"
            f"        stoichiometry: {stoich},\n"
            f"        texName: {texName!r},\n"
            f"      }})"
        )

    def _gen_srg(
        el: SymbolicSurrogate,
        used: set[str],
        subs: dict[sympy.Symbol, sympy.Symbol],
    ) -> str:
        # FIXME: add surrogate to mxlweb

        out = []
        for o, sfn in zip(el.outputs, el.fns, strict=True):
            texName = (
                valid_tex_identifier(o)
                if (texName := tex_names.get(o)) is None
                else texName
            )
            fn = sympy_to_inline_mxlweb(sfn.expr, used, subs)

            if (ss := el.stoichiometry.get(o)) is not None:
                stoich = f"[{', '.join(_gen_stoich(k, v, used, subs) for k, v in ss.items())}]"
                out.append(
                    f'      .addReaction("{name_map[o]}", {{\n'
                    f"        fn: {fn},\n"
                    f"        stoichiometry: {stoich},\n"
                    f"        texName: {texName!r},\n"
                    f"      }})"
                )
            else:
                out.append(
                    f'      .addAssignment("{name_map[o]}", {{\n'
                    f"        fn: {fn},\n"
                    f"        texName: {texName!r},\n"
                    f"      }})"
                )
        return "\n".join(out)

    lines = []
    lines.extend(_gen_par(k, v, used, subs) for k, v in sr.parameters.items())
    lines.extend(_gen_var(k, v, used, subs) for k, v in sr.variables.items())
    lines.extend(_gen_der(k, v, used, subs) for k, v in sr.derived.items())
    lines.extend(_gen_rxn(k, v, used, subs) for k, v in sr.reactions.items())
    lines.extend(_gen_srg(v, used, subs) for v in sr.surrogates.values())
    lines.extend(_gen_rdo(k, v, used, subs) for k, v in sr.readouts.items())

    # Build import list from collected class names
    mathml_import_str = ", ".join(sorted(used))
    model_builder_str = "\n".join(
        (
            "export function initModel(): ModelBuilder {",
            "    return new ModelBuilder()",
            "\n".join(lines),
            "  }",
        )
    )
    return "\n".join(
        [
            f'import {{ {mathml_import_str} }} from "$lib/mathml";',
            'import { ModelBuilder } from "$lib/model-editor/modelBuilder";',
            "",
            model_builder_str,
        ]
    )


def generate_model_code_py(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    typed: bool = False,
) -> Codegen:
    """Transform the model into Python functions, inlining all function calls.

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments rather than
        inlining their values.
    derived_to_calculate
        Subset of derived component names to emit in the ``derived`` function.
        All transitive dependencies are included automatically. ``None`` emits
        everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.
    typed
        When ``True`` add ``float`` type annotations to local assignments.

    Returns
    -------
    Codegen
        Generated Python source split into imports, model, derived, and inits.

    """

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
        name_map={name: valid_identifier(name) for name in model.ids},
    )


def generate_model_code_ts(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> Codegen:
    """Transform the model into TypeScript functions, inlining all function calls.

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments.
    derived_to_calculate
        Subset of derived component names to emit. ``None`` emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.

    Returns
    -------
    Codegen
        Generated TypeScript source split into imports, model, derived, and inits.

    """

    def ts_type(t: str) -> str:
        return {
            "float": "number",
            "Iterable[float]": "number[]",
            "tuple[Iterable[float], Iterable[float]]": "[number[], number[]]",
        }.get(t, t)

    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:
        args_str = ", ".join(f"{k}: {ts_type(t)}" for k, t in args)
        return f"function {name}({args_str}): {ts_type(return_type)} {{"

    def variable_unpacking(variables: list[str]) -> str:
        return f"    const [{', '.join(variables)}] = variables;"

    def list_template(elements: list[str]) -> str:
        return f"[{', '.join(elements)}]"

    def assignment_template(name: str, value: str) -> str:
        return f"    const {name}: number = {value};"

    def return_template(variables: list[str]) -> str:
        return f"    return [{', '.join(variables) or '[]'}];\n}};"

    return _generate_model_code(
        model,
        free_parameters=[] if free_parameters is None else free_parameters,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_js,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=custom_fns if custom_fns is not None else {},
        imports=[],
        name_map={name: valid_identifier(name) for name in model.ids},
    )


def generate_model_code_jl(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> Codegen:
    """Transform the model into Julia functions, inlining all function calls.

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments.
    derived_to_calculate
        Subset of derived component names to emit. ``None`` emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.

    Returns
    -------
    Codegen
        Generated Julia source split into imports, model, derived, and inits.

    """

    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:  # noqa: ARG001
        args_str = ", ".join(k for k, _ in args)
        return f"function {name}({args_str})"

    def variable_unpacking(variables: list[str]) -> str:
        return f"    {', '.join(variables)} = variables"

    def list_template(elements: list[str]) -> str:
        return f"[{', '.join(elements)}]"

    def assignment_template(name: str, value: str) -> str:
        return f"    {name} = {value}"

    def return_template(variables: list[str]) -> str:
        return f"    return [{', '.join(variables) or '()'}]\nend"

    return _generate_model_code(
        model,
        free_parameters=[] if free_parameters is None else free_parameters,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_julia,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=custom_fns if custom_fns is not None else {},
        imports=[],
        name_map={name: valid_identifier(name) for name in model.ids},
    )


def generate_model_code_matlab(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> Codegen:
    """Transform the model into MATLAB/Octave functions, inlining all function calls.

    The model function uses ``dydt`` as its output variable for compatibility
    with MATLAB ODE solvers (``ode45``, ``ode15s``, etc.).

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments.
    derived_to_calculate
        Subset of derived component names to emit. ``None`` emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.

    Returns
    -------
    Codegen
        Generated MATLAB source split into imports, model, derived, and inits.

    """
    _fn_context: list[str] = ["model"]

    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:  # noqa: ARG001
        _fn_context[0] = name
        mat_args = ", ".join("t" if k == "time" else k for k, _ in args)
        if name == "model":
            ret_decl = "dydt"
        elif name == "inits":
            ret_decl = "[vars, pars]"
        else:
            ret_decl = "out"
        return f"function {ret_decl} = {name}({mat_args})"

    def variable_unpacking(variables: list[str]) -> str:
        return f"    [{', '.join(variables)}] = num2cell(variables){{:}};"

    def list_template(elements: list[str]) -> str:
        return f"[{', '.join(elements)}]"

    def assignment_template(name: str, value: str) -> str:
        return f"    {name} = {value};"

    def return_template(variables: list[str]) -> str:
        ctx = _fn_context[0]
        if ctx == "model":
            return f"    dydt = [{', '.join(variables)}]';\nend"
        if ctx == "inits":
            match variables:
                case [vars_str, pars_str, *_]:
                    return f"    vars = {vars_str}';\n    pars = {pars_str}';\nend"
                case [vars_str]:
                    return f"    vars = {vars_str}';\n    pars = [];\nend"
                case _:
                    return "    vars = []';\n    pars = []';\nend"
        return f"    out = [{', '.join(variables)}];\nend"

    return _generate_model_code(
        model,
        free_parameters=[] if free_parameters is None else free_parameters,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_matlab,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=custom_fns if custom_fns is not None else {},
        imports=[],
        name_map={name: valid_identifier(name) for name in model.ids},
    )


def generate_model_code_rs(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> Codegen:
    """Transform the model into Rust functions, inlining all function calls.

    Array sizes are determined at call time and embedded in the function
    signatures as const generics (e.g. ``[f64; N]``).

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments.
    derived_to_calculate
        Subset of derived component names to emit. ``None`` emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.

    Returns
    -------
    Codegen
        Generated Rust source split into imports, model, derived, and inits.

    """
    _free_pars = [] if free_parameters is None else free_parameters
    _custom_fns = {} if custom_fns is None else custom_fns
    n_vars = len(model.get_initial_conditions())
    n_derived = len(_get_extended_returns(model, derived_to_calculate))
    _variable_order = list(model.get_initial_conditions())
    _nsm = _normalized_symbolic_model(
        model, _free_pars, derived_to_calculate, _custom_fns
    )
    n_init_pars = len([i for i, _ in _nsm.inits if i not in _variable_order])

    _fn_context: list[str] = ["model"]

    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:
        _fn_context[0] = name

        def rs_type(t: str) -> str:
            if "Iterable" in t:
                return f"&[f64; {n_vars}]"
            return "f64"

        args_str = ", ".join(f"{k}: {rs_type(t)}" for k, t in args)
        if name == "model":
            ret_type = f"[f64; {n_vars}]"
        elif name == "derived":
            ret_type = f"[f64; {n_derived}]"
        elif name == "inits":
            ret_type = f"([f64; {n_vars}], [f64; {n_init_pars}])"
        else:
            ret_type = return_type
        return f"fn {name}({args_str}) -> {ret_type} {{"

    def variable_unpacking(variables: list[str]) -> str:
        return f"    let [{', '.join(variables)}] = *variables;"

    def list_template(elements: list[str]) -> str:
        return f"[{', '.join(elements)}]"

    def assignment_template(name: str, value: str) -> str:
        return f"    let {name}: f64 = {value};"

    def return_template(variables: list[str]) -> str:
        if _fn_context[0] == "inits":
            match variables:
                case [vars_str, pars_str, *_]:
                    return f"    return ({vars_str}, {pars_str})\n}}"
                case _:
                    return f"    return [{', '.join(variables) or '()'}]\n}}"
        return f"    return [{', '.join(variables) or '()'}]\n}}"

    return _generate_model_code(
        model,
        free_parameters=_free_pars,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_rust,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=_custom_fns,
        imports=[],
        name_map={name: valid_identifier(name) for name in model.ids},
    )


def generate_model_code_cpp(
    model: Model,
    *,
    free_parameters: list[str] | None = None,
    derived_to_calculate: list[str] | None = None,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
) -> Codegen:
    """Transform the model into C++17 functions, inlining all function calls.

    Uses ``std::array<double, N>`` for fixed-size arrays and
    ``std::pair`` for the inits return. Requires ``<array>``, ``<cmath>``,
    and ``<utility>`` headers (included in ``Codegen.imports``).

    Parameters
    ----------
    model
        Model to generate code for.
    free_parameters
        Parameter names to expose as extra function arguments.
    derived_to_calculate
        Subset of derived component names to emit. ``None`` emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.

    Returns
    -------
    Codegen
        Generated C++ source split into imports, model, derived, and inits.

    """
    _free_pars = [] if free_parameters is None else free_parameters
    _custom_fns = {} if custom_fns is None else custom_fns
    n_vars = len(model.get_initial_conditions())
    n_derived = len(_get_extended_returns(model, derived_to_calculate))
    _variable_order = list(model.get_initial_conditions())
    _nsm = _normalized_symbolic_model(
        model, _free_pars, derived_to_calculate, _custom_fns
    )
    n_init_pars = len([i for i, _ in _nsm.inits if i not in _variable_order])

    _fn_context: list[str] = ["model"]

    def fn_template(name: str, args: list[tuple[str, str]], return_type: str) -> str:
        _fn_context[0] = name

        def cpp_type(t: str) -> str:
            if "Iterable" in t:
                return f"const std::array<double, {n_vars}>&"
            return "double"

        args_str = ", ".join(f"{cpp_type(t)} {k}" for k, t in args)
        if name == "model":
            ret_type = f"std::array<double, {n_vars}>"
        elif name == "derived":
            ret_type = f"std::array<double, {n_derived}>"
        elif name == "inits":
            ret_type = f"std::pair<std::array<double, {n_vars}>, std::array<double, {n_init_pars}>>"
        else:
            ret_type = return_type
        return f"{ret_type} {name}({args_str}) {{"

    def variable_unpacking(variables: list[str]) -> str:
        return f"    const auto [{', '.join(variables)}] = variables;"

    def list_template(elements: list[str]) -> str:
        return f"{{{', '.join(elements)}}}"

    def assignment_template(name: str, value: str) -> str:
        return f"    double {name} = {value};"

    def return_template(variables: list[str]) -> str:
        if _fn_context[0] == "inits":
            match variables:
                case [vars_str, pars_str, *_]:
                    return f"    return {{{vars_str}, {pars_str}}};\n}}"
                case _:
                    inner = ", ".join(variables)
                    return f"    return {{{inner}}};\n}}"
        inner = ", ".join(variables)
        return f"    return {{{inner}}};\n}}"

    return _generate_model_code(
        model,
        free_parameters=_free_pars,
        fn_template=fn_template,
        assignment_template=assignment_template,
        expr_template=sympy_to_inline_cxx,
        variable_unpacking=variable_unpacking,
        derived_to_calculate=derived_to_calculate,
        list_formatter=list_template,
        return_formatter=return_template,
        custom_fns=_custom_fns,
        imports=["#include <array>", "#include <cmath>", "#include <utility>", ""],
        name_map={name: valid_identifier(name) for name in model.ids},
    )
