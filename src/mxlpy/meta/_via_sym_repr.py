"""Generate multi-language model code from a symbolic model representation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import sympy
from wadler_lindig import pformat

from mxlpy import _topo
from mxlpy.meta.source_tools import fn_to_sympy_expr, fn_to_sympy_exprs
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
    sympy_to_inline_cxx,
    sympy_to_inline_js,
    sympy_to_inline_julia,
    sympy_to_inline_matlab,
    sympy_to_inline_mxlweb,
    sympy_to_inline_py,
    sympy_to_inline_rust,
)
from mxlpy.surrogates import qss
from mxlpy.surrogates.abstract import AbstractSurrogate
from mxlpy.types import Derived, InitialAssignment
from mxlpy.units import Quantity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mxlpy.model import Model

_LOGGER = logging.getLogger()


__all__ = [
    "Codegen",
    "DiffLatexCodegen",
    "ExprTemplate",
    "FnDeclTemplate",
    "LatexCodegen",
    "ListTemplate",
    "NormalizedSymbolicModel",
    "ReducedSymbolicRepr",
    "ReturnTemplate",
    "SymbolicFn",
    "SymbolicParameter",
    "SymbolicReaction",
    "SymbolicRepr",
    "SymbolicSurrogate",
    "SymbolicVariable",
    "TupleTemplate",
    "VariableAssignmentTemplate",
    "VariableUnpackingTemplate",
    "generate_latex_diff",
    "generate_latex_document",
    "generate_model_code_cpp",
    "generate_model_code_jl",
    "generate_model_code_latex",
    "generate_model_code_matlab",
    "generate_model_code_mxlweb",
    "generate_model_code_py",
    "generate_model_code_rs",
    "generate_model_code_ts",
    "model_to_symbolic_repr",
    "valid_identifier",
    "valid_tex_identifier",
]

_LOGGER = logging.getLogger()


class NormalizedSymbolicModel(NamedTuple):
    """Symbolic model normalized into flat assignment lists for code generation."""

    body: list[tuple[str, sympy.Expr]]
    extended: list[tuple[str, sympy.Expr]]
    diff_eqs: dict[str, sympy.Expr]
    inits: list[tuple[str, sympy.Expr]]
    free_pars: list[str]


@dataclass
class Codegen:
    """Generated code split into four sections ready for emission."""

    imports: str
    model: str
    derived: str
    inits: str

    def full(self) -> str:
        return "\n\n".join(
            i
            for i in (self.imports, self.model, self.derived, self.inits)
            if len(i) > 0
        )


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


@dataclass(unsafe_hash=True)
class SymbolicFn:
    """Container for symbolic fn."""

    fn_name: str
    expr: sympy.Expr
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass(unsafe_hash=True)
class SymbolicVariable:
    """Container for symbolic variable."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass(unsafe_hash=True)
class SymbolicParameter:
    """Container for symbolic par."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass(unsafe_hash=True)
class SymbolicReaction:
    """Container for symbolic rxn."""

    fn: SymbolicFn
    stoichiometry: dict[str, sympy.Float | SymbolicFn]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass(unsafe_hash=True)
class SymbolicSurrogate:
    """Container for symbolic rxn."""

    fns: list[SymbolicFn]
    outputs: list[str]
    stoichiometry: dict[str, dict[str, sympy.Float | SymbolicFn]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class ReducedSymbolicRepr:
    """Container for symplifiedsymbolic model."""

    variables: dict[str, SymbolicVariable]
    parameters: dict[str, SymbolicParameter]
    derived: dict[str, SymbolicFn]
    reactions: dict[str, SymbolicFn]
    diffeqs: dict[str, sympy.Expr]

    def difference(self, other: ReducedSymbolicRepr) -> ReducedSymbolicRepr:
        return ReducedSymbolicRepr(
            variables={
                k: v2
                for k, v2 in other.variables.items()
                if (v1 := self.variables.get(k)) is None or v1 != v2
            },
            parameters={
                k: v2
                for k, v2 in other.parameters.items()
                if (v1 := self.parameters.get(k)) is None or v1 != v2
            },
            derived={
                k: v2
                for k, v2 in other.derived.items()
                if (v1 := self.derived.get(k)) is None or v1 != v2
            },
            reactions={
                k: v2
                for k, v2 in other.reactions.items()
                if (v1 := self.reactions.get(k)) is None or v1 != v2
            },
            diffeqs={
                k: v2
                for k, v2 in other.diffeqs.items()
                if (v1 := self.diffeqs.get(k)) is None or v1 != v2
            },
        )


@dataclass
class SymbolicRepr:
    """Container for symbolic model."""

    variables: dict[str, SymbolicVariable] = field(default_factory=dict)
    parameters: dict[str, SymbolicParameter] = field(default_factory=dict)
    derived: dict[str, SymbolicFn] = field(default_factory=dict)
    readouts: dict[str, SymbolicFn] = field(default_factory=dict)
    reactions: dict[str, SymbolicReaction] = field(default_factory=dict)
    surrogates: dict[str, SymbolicSurrogate] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def difference(self, other: SymbolicRepr) -> SymbolicRepr:
        return SymbolicRepr(
            variables={
                k: v2
                for k, v2 in other.variables.items()
                if (v1 := self.variables.get(k)) is None or v1 != v2
            },
            parameters={
                k: v2
                for k, v2 in other.parameters.items()
                if (v1 := self.parameters.get(k)) is None or v1 != v2
            },
            derived={
                k: v2
                for k, v2 in other.derived.items()
                if (v1 := self.derived.get(k)) is None or v1 != v2
            },
            readouts={
                k: v2
                for k, v2 in other.readouts.items()
                if (v1 := self.readouts.get(k)) is None or v1 != v2
            },
            reactions={
                k: v2
                for k, v2 in other.reactions.items()
                if (v1 := self.reactions.get(k)) is None or v1 != v2
            },
            surrogates={
                k: v2
                for k, v2 in other.surrogates.items()
                if (v1 := self.surrogates.get(k)) is None or v1 != v2
            },
        )

    def reduce(self) -> ReducedSymbolicRepr:
        all_derived = self.derived | self.readouts
        all_reactions = self.reactions

        for srg in self.surrogates.values():
            for fn, out in zip(srg.fns, srg.outputs, strict=True):
                if out in srg.stoichiometry:
                    all_reactions[out] = SymbolicReaction(
                        fn=fn, stoichiometry=srg.stoichiometry[out]
                    )
                else:
                    all_derived[out] = fn

        diffeqs: dict[str, sympy.Expr] = {i: sympy.Integer(0) for i in self.variables}
        for rate, rxn in all_reactions.items():
            for cpd, factor in rxn.stoichiometry.items():
                if isinstance(factor, SymbolicFn):
                    diffeqs[cpd] += sympy.Symbol(rate) * factor.expr  # pyright: ignore[reportOperatorIssue]
                else:
                    diffeqs[cpd] += sympy.Symbol(rate) * factor  # pyright: ignore[reportOperatorIssue]

        return ReducedSymbolicRepr(
            variables=self.variables,
            parameters=self.parameters,
            derived=all_derived,
            reactions={k: v.fn for k, v in all_reactions.items()},
            diffeqs=diffeqs,
        )

    def generate_mxlpy(self) -> str:
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
        fns: dict[str, SymbolicFn] = {}
        surr_fns: dict[str, tuple[list[str], list[SymbolicFn]]] = {}

        def _codegen_variable(k: str, el: SymbolicVariable) -> str:
            val = el.value
            if isinstance(val, SymbolicFn):
                fns[val.fn_name] = val
                return (
                    "        .add_variable(\n"
                    f"            {k!r},\n"
                    f"            initial_value=InitialAssignment(fn={val.fn_name}, args={val.args!r}),\n"
                    "        )"
                )
            return f"        .add_variable({k!r}, initial_value={float(val)!r})"

        def _codegen_parameter(k: str, el: SymbolicParameter) -> str:
            val = el.value
            if isinstance(val, SymbolicFn):
                fns[val.fn_name] = val
                return (
                    "        .add_parameter(\n"
                    f"            {k!r},\n"
                    f"            initial_value=InitialAssignment(fn={val.fn_name}, args={val.args!r}),\n"
                    "        )"
                )
            return f"        .add_parameter({k!r}, value={float(val)!r})"

        def _codegen_derived(k: str, el: SymbolicFn) -> str:
            fns[el.fn_name] = el
            return (
                "        .add_derived(\n"
                f"            {k!r},\n"
                f"            fn={el.fn_name},\n"
                f"            args={el.args!r},\n"
                "        )"
            )

        def _codegen_readout(k: str, el: SymbolicFn) -> str:
            fns[el.fn_name] = el
            return (
                "        .add_readout(\n"
                f"            {k!r},\n"
                f"            fn={el.fn_name},\n"
                f"            args={el.args!r},\n"
                "        )"
            )

        def _codegen_reaction(k: str, el: SymbolicReaction) -> str:
            fns[el.fn.fn_name] = el.fn
            stoichiometry: list[str] = []
            for var, stoich in el.stoichiometry.items():
                if isinstance(stoich, SymbolicFn):
                    fns[stoich.fn_name] = stoich
                    stoichiometry.append(
                        f'"{var}": Derived(fn={stoich.fn_name}, args={stoich.args!r})'
                    )
                else:
                    stoichiometry.append(f'"{var}": {float(stoich)!r}')
            return (
                "        .add_reaction(\n"
                f"            {k!r},\n"
                f"            fn={el.fn.fn_name},\n"
                f"            args={el.fn.args!r},\n"
                f"            stoichiometry={{{', '.join(stoichiometry)}}},\n"
                "        )"
            )

        def _codegen_surrogate(k: str, el: SymbolicSurrogate) -> str:
            fn_name = f"{k}_model"
            surr_fns[fn_name] = (el.outputs, el.fns)
            args = el.fns[0].args if el.fns else []
            stoichiometries: list[str] = []
            for rxn_name, rxn_stoich in el.stoichiometry.items():
                rxn_parts: list[str] = []
                for var, stoich in rxn_stoich.items():
                    if isinstance(stoich, SymbolicFn):
                        fns[stoich.fn_name] = stoich
                        rxn_parts.append(
                            f'"{var}": Derived(fn={stoich.fn_name}, args={stoich.args!r})'
                        )
                    else:
                        rxn_parts.append(f'"{var}": {float(stoich)!r}')
                stoichiometries.append(f'"{rxn_name}": {{{", ".join(rxn_parts)}}}')
            return (
                "        .add_surrogate(\n"
                f"            {k!r},\n"
                "            qss.Surrogate(\n"
                f"                model={fn_name},\n"
                f"                args={args!r},\n"
                f"                outputs={el.outputs!r},\n"
                f"                stoichiometries={{{', '.join(stoichiometries)}}},\n"
                "            ),\n"
                "        )"
            )

        variables = [_codegen_variable(k, v) for k, v in self.variables.items()]
        parameters = [_codegen_parameter(k, v) for k, v in self.parameters.items()]
        derived = [_codegen_derived(k, v) for k, v in self.derived.items()]
        readouts = [_codegen_readout(k, v) for k, v in self.readouts.items()]
        reactions = [_codegen_reaction(k, v) for k, v in self.reactions.items()]
        surrogates = [_codegen_surrogate(k, v) for k, v in self.surrogates.items()]

        def _gen_fn(sf: SymbolicFn) -> str:
            args_str = ", ".join(f"{a}: float" for a in sf.args)
            return (
                f"def {sf.fn_name}({args_str}) -> float:\n"
                f"    return {sympy_to_inline_py(sf.expr)}"
            )

        def _gen_surr_fn(
            fn_name: str, outputs_and_fns: tuple[list[str], list[SymbolicFn]]
        ) -> str:
            outputs, sfs = outputs_and_fns
            args = sfs[0].args if sfs else []
            args_str = ", ".join(f"{a}: float" for a in args)
            body = "\n".join(
                f"    {out} = {sympy_to_inline_py(sf.expr)}"
                for out, sf in zip(outputs, sfs, strict=True)
            )
            ret = ", ".join(outputs)
            return (
                f"def {fn_name}({args_str}) -> tuple[float, ...]:\n"
                f"{body}\n"
                f"    return {ret}"
            )

        fn_defs = [
            *map(_gen_fn, fns.values()),
            *(_gen_surr_fn(n, v) for n, v in surr_fns.items()),
        ]
        fn_block = "\n\n".join(fn_defs)

        builder_lines = (
            variables + parameters + derived + reactions + surrogates + readouts
        )

        imports = [
            "import math",
            "from mxlpy import Derived, InitialAssignment, Model",
            *(["from mxlpy.surrogates import qss"] if surrogates else []),
        ]

        return "\n".join(
            [
                *imports,
                "",
                fn_block,
                "",
                "def create_model() -> Model:",
                "    return (",
                "        Model()",
                "\n".join(builder_lines),
                "    )",
            ]
        )


###############################################################################
# Building symbolic repr
###############################################################################


def _fn_to_symbolic_repr(
    k: str,
    fn: Callable,
    model_args: list[str],
    *,
    only_warn: bool,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> SymbolicFn:
    fn_name = fn.__name__
    args = cast(list, list_of_symbols(model_args))
    if (expr := custom_fns.get(k)) is not None:
        if isinstance(expr, list):
            msg = f"Expected a single expr for '{k}' but got multiple"
            if only_warn:
                expr = sympy.Float("nan")
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

        return SymbolicFn(fn_name=fn_name, expr=expr, args=model_args)

    if (expr := fn_to_sympy_expr(fn, origin=k, model_args=args)) is None:
        msg = f"Unable to parse fn for '{k}'"
        if only_warn:
            expr = sympy.Float("nan")
            _LOGGER.warning(msg)
        else:
            raise ValueError(msg)
    return SymbolicFn(fn_name=fn_name, expr=expr, args=model_args)


def _fns_to_symbolic_reprs(
    k: str,
    fn: Callable,
    model_args: list[str],
    outputs: list[str],
    *,
    only_warn: bool,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> list[SymbolicFn]:
    fn_name = fn.__name__
    args = cast(list, list_of_symbols(model_args))
    if (exprs := custom_fns.get(k)) is not None:
        if not isinstance(exprs, list):
            msg = f"Expected multiple exprs for '{k}' but got a single expr"
            if only_warn:
                exprs = [sympy.Float("nan") for i in outputs]
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

        return [
            SymbolicFn(fn_name=f"{fn_name}_{i}", expr=expr, args=model_args)
            for i, expr in zip(outputs, exprs, strict=True)
        ]

    if (exprs := fn_to_sympy_exprs(fn, origin=k, model_args=args)) is None:
        msg = f"Unable to parse fns for '{k}'"
        if only_warn:
            exprs = [sympy.Float("nan") for _ in outputs]
            _LOGGER.warning(msg)
        else:
            raise ValueError(msg)

    return [
        SymbolicFn(fn_name=f"{fn_name}_{i}", expr=expr, args=model_args)
        for i, expr in zip(outputs, exprs, strict=True)
    ]


def model_to_symbolic_repr(
    model: Model,
    *,
    only_warn: bool = False,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]],
) -> SymbolicRepr:
    sym = SymbolicRepr()

    for k, variable in model.get_raw_variables().items():
        sym.variables[k] = SymbolicVariable(
            value=_fn_to_symbolic_repr(
                k,
                val.fn,
                val.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )
            if isinstance(val := variable.initial_value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, variable.unit),
        )

    for k, parameter in model.get_raw_parameters().items():
        sym.parameters[k] = SymbolicParameter(
            value=_fn_to_symbolic_repr(
                k,
                val.fn,
                val.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )
            if isinstance(val := parameter.value, InitialAssignment)
            else sympy.Float(val),
            unit=cast(Quantity, parameter.unit),
        )

    for k, der in model.get_raw_derived().items():
        sym.derived[k] = _fn_to_symbolic_repr(
            k,
            der.fn,
            der.args,
            only_warn=only_warn,
            custom_fns=custom_fns,
        )

    for k, rxn in model.get_raw_reactions().items():
        sym.reactions[k] = SymbolicReaction(
            fn=_fn_to_symbolic_repr(
                k,
                rxn.fn,
                rxn.args,
                only_warn=only_warn,
                custom_fns=custom_fns,
            ),
            stoichiometry={
                k: _fn_to_symbolic_repr(
                    k,
                    v.fn,
                    v.args,
                    only_warn=only_warn,
                    custom_fns=custom_fns,
                )
                if isinstance(v, Derived)
                else sympy.Float(v)
                for k, v in rxn.stoichiometry.items()
            },
        )

    for k, srg in model.get_raw_surrogates().items():
        if isinstance(srg, qss.Surrogate):
            fns = _fns_to_symbolic_reprs(
                k,
                srg.model,
                srg.args,
                srg.outputs,
                only_warn=only_warn,
                custom_fns=custom_fns,
            )

            sym.surrogates[k] = SymbolicSurrogate(
                fns=fns,
                outputs=srg.outputs,
                stoichiometry={
                    out: {
                        k: _fn_to_symbolic_repr(
                            k,
                            v.fn,
                            v.args,
                            only_warn=only_warn,
                            custom_fns=custom_fns,
                        )
                        if isinstance(v, Derived)
                        else sympy.Float(v)
                        for k, v in stoich.items()
                    }
                    for out, stoich in srg.stoichiometries.items()
                },
            )
        else:
            msg = f"Unsupported surrogate type {type(srg)}."
            if only_warn:
                _LOGGER.warning(msg)
            else:
                raise ValueError(msg)

    for k, der in model.get_raw_readouts().items():
        sym.readouts[k] = _fn_to_symbolic_repr(
            k,
            der.fn,
            der.args,
            only_warn=only_warn,
            custom_fns=custom_fns,
        )

    return sym


###############################################################################
# General utils
###############################################################################


def valid_tex_identifier(k: str) -> str:
    return k.replace("_", r"\_")


def valid_identifier(name: str) -> str:
    """Convert an arbitrary string to a valid identifier.

    Uses C99 identifier rules ([a-zA-Z_][a-zA-Z0-9_]*), which are the
    strictest common denominator across all supported target languages
    (C, C++, Rust, TypeScript, Julia, MATLAB/Octave, Python).

    Parameters
    ----------
    name
        Original component name from the model

    Returns
    -------
    str
        Sanitized identifier safe for use in all target languages

    """
    sanitized = (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
        .replace('"', "")
        .replace("'", "")
        .replace("^", "")
        .replace("|", "")
        .replace("=", "_eq_")
        .replace(">", "_lg_")
        .replace("<", "_sm_")
        .replace("+", "_plus_")
        .replace("-", "_minus_")
        .replace("*", "_star_")
        .replace("/", "_div_")
    )
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.rstrip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_"


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


###############################################################################
# Actual codegen
###############################################################################


def generate_model_code_mxlweb(
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
        if len(variables) == 0:
            return ""
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


###############################################################################
# Latex codegen
# Different to the other ones in that we return each group of elements seperately
# since there is no need to generate statements in order of execution
###############################################################################


def _sympy_to_latex(expr: sympy.Expr, symbol_names: dict[sympy.Symbol, str]) -> str:
    return sympy.latex(
        expr,
        fold_frac_powers=True,
        fold_func_brackets=True,
        fold_short_frac=True,
        symbol_names=symbol_names,
        mul_symbol="dot",
    )


def _rename_latex(s: str) -> str:
    if s[0].isdigit():
        s = s[1:]
        if s[0] == "-":
            s = s[1:]
    return (
        s.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("*", "")
    )


def _shorten_long_symbols(
    expr: sympy.Expr,
    cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> tuple[sympy.Expr, list[tuple[str, str]]]:
    """Substitute long unaliased symbol names with short stand-ins.

    For every other ``Symbol`` whose name is at least *cutoff* characters long a
    short placeholder ``x0``, ``x1``, … is created.

    Returns ``(new_expr, annotations)`` where *annotations* is a list of
    ``(short_latex, long_latex)`` pairs (empty when nothing was replaced).
    """
    subs: dict[sympy.Symbol, sympy.Symbol] = {}
    annotations: list[tuple[str, str]] = []
    i = 0
    for sym in sorted(
        (s for s in expr.free_symbols if isinstance(s, sympy.Symbol)),
        key=lambda s: s.name,
    ):
        if sym not in symbol_names and len(sym.name) >= cutoff:
            short = sympy.Symbol(f"x{i}")
            subs[sym] = short
            annotations.append((sympy.latex(short), _math_name(sym.name)))
            i += 1
    return cast(sympy.Expr, expr.subs(subs)), annotations


def _subsection_(s: str) -> str:
    # depth = 2
    return r"\FloatBarrier" + rf"\subsection*{{{s}}}"


def _mathrm(s: str) -> str:
    return rf"\mathrm{{{s}}}"


def _math_name(s: str) -> str:
    return _mathrm(valid_tex_identifier(_rename_latex(s)))


def _tex_lhs(name: str) -> str:
    return rf"\frac{{d\ {name}}}{{dt}}"


def _tex_list_as_sections(rows: Iterable[tuple[str, str]]) -> str:
    return "\n\n".join(
        [
            "\n".join(
                (
                    _subsection_(name),
                    content,
                )
            )
            for name, content in rows
            if len(content) > 0
        ]
    )


def _annotation_block(annotations: list[tuple[str, str]]) -> str:
    """Standalone align* listing symbol abbreviations used in a shortened expression."""
    rows = [r"  &\quad \mathrm{with}"] + [
        rf"  &\qquad {s} :: {lg}" for s, lg in annotations
    ]
    body = " \\\\\n".join(rows)
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


def _render_diff_rhs(
    old_expr: sympy.Expr | None,
    new_expr: sympy.Expr | None,
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> tuple[str, list[tuple[str, str]]]:
    annotations: list[tuple[str, str]] = []
    if old_expr is None:
        if new_expr is None:
            return "", []
        short, ann = _shorten_long_symbols(new_expr, long_name_cutoff, symbol_names)
        annotations.extend(ann)
        return (
            rf"{{\color{{green}} {_sympy_to_latex(short, symbol_names)}}}",
            annotations,
        )
    if new_expr is None:
        short, ann = _shorten_long_symbols(old_expr, long_name_cutoff, symbol_names)
        annotations.extend(ann)
        return (
            rf"{{\color{{red}} \stkout{{{_sympy_to_latex(short, symbol_names)}}}}}",
            annotations,
        )
    if old_expr == new_expr:
        short, ann = _shorten_long_symbols(new_expr, long_name_cutoff, symbol_names)
        annotations.extend(ann)
        return _sympy_to_latex(short, symbol_names), annotations
    short_old, ann_old = _shorten_long_symbols(old_expr, long_name_cutoff, symbol_names)
    short_new, ann_new = _shorten_long_symbols(new_expr, long_name_cutoff, symbol_names)
    annotations.extend(ann_old)
    annotations.extend(ann_new)
    old_rhs = (
        rf"{{\color{{red}} \stkout{{{_sympy_to_latex(short_old, symbol_names)}}}}}"
    )
    new_rhs = rf"{{\color{{green}} {_sympy_to_latex(short_new, symbol_names)}}}"
    return f"{old_rhs} {new_rhs}", annotations


def _tex_align_block(
    rows: list[tuple[str, sympy.Expr]],
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    parts = []
    for lhs, expr in rows:
        short_expr, annotations = _shorten_long_symbols(
            expr, long_name_cutoff, symbol_names
        )
        rhs = _sympy_to_latex(short_expr, symbol_names)
        if annotations:
            # Leading \\ terminates the equation row; subsequent lines are
            # continuation rows in the align* block.
            rhs += "\\\\\n  " + r"&\quad \mathrm{with}"
            for s, lg in annotations:
                rhs += "\\\\\n  " + rf"&\qquad {s} :: {lg}"
        parts.append(f"  {lhs} &= {rhs}")
    body = " \\\\\n".join(parts)
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


def _tex_align_block_diff(
    rows: list[tuple[str, sympy.Expr | None, sympy.Expr | None]],
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    parts = []
    for lhs, old_expr, new_expr in rows:
        rhs, annotations = _render_diff_rhs(
            old_expr, new_expr, long_name_cutoff, symbol_names
        )
        if annotations:
            rhs += "\\\\\n  " + r"&\quad \mathrm{with}"
            for s, lg in annotations:
                rhs += "\\\\\n  " + rf"&\qquad {s} :: {lg}"
        parts.append(f"  {lhs} &= {rhs}")
    body = " \\\\\n".join(parts)
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


def _tex_dmath_blocks(
    rows: list[tuple[str, sympy.Expr]],
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    parts = []
    for lhs, expr in rows:
        short_expr, annotations = _shorten_long_symbols(
            expr, long_name_cutoff, symbol_names
        )
        rhs = _sympy_to_latex(short_expr, symbol_names)
        block = f"\\begin{{dmath*}}\n  {lhs} = {rhs}\n\\end{{dmath*}}"
        if annotations:
            block += "\n" + _annotation_block(annotations)
        parts.append(block)
    return "\n".join(parts)


def _tex_dmath_blocks_diff(
    rows: list[tuple[str, sympy.Expr | None, sympy.Expr | None]],
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    parts = []
    for lhs, old_expr, new_expr in rows:
        rhs, annotations = _render_diff_rhs(
            old_expr, new_expr, long_name_cutoff, symbol_names
        )
        block = f"\\begin{{dmath*}}\n  {lhs} = {rhs}\n\\end{{dmath*}}"
        if annotations:
            block += "\n" + _annotation_block(annotations)
        parts.append(block)
    return "\n".join(parts)


def _tex_math_table2(
    headers: tuple[str, str],
    rows: list[tuple[str, sympy.Expr]],
    label: str,
    short_desc: str,
    long_desc: str,
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    columns = "|".join(["c"] * 2)
    tab = "    "
    all_annotations: dict[str, str] = {}

    def _render(lhs: str, expr: sympy.Expr) -> str:
        short_expr, annotations = _shorten_long_symbols(
            expr, long_name_cutoff, symbol_names
        )
        all_annotations.update(annotations)
        return rf"{tab} ${lhs}$ & ${_sympy_to_latex(short_expr, symbol_names)}$ \\"

    table = "\n".join(
        [
            r"\begin{longtable}" + f"{{{columns}}}",
            tab + " & ".join(headers) + r" \\",
            tab + r"\hline",
            tab + r"\endhead",
        ]
        + [_render(lhs, expr) for lhs, expr in rows]
        + [
            tab + rf"\caption[{short_desc}]{{{long_desc}}}",
            tab + rf"\label{{table:{label}}}",
            r"\end{longtable}",
        ]
    )
    if all_annotations:
        table += "\n" + _annotation_block(list(all_annotations.items()))
    return table


def _tex_math_table2_diff(
    headers: tuple[str, str],
    rows: list[tuple[str, sympy.Expr | None, sympy.Expr | None]],
    label: str,
    short_desc: str,
    long_desc: str,
    long_name_cutoff: int,
    symbol_names: dict[sympy.Symbol, str],
) -> str:
    columns = "|".join(["c"] * 2)
    tab = "    "
    all_annotations: dict[str, str] = {}

    def _render(
        lhs: str, old_expr: sympy.Expr | None, new_expr: sympy.Expr | None
    ) -> str:
        rhs, annotations = _render_diff_rhs(
            old_expr, new_expr, long_name_cutoff, symbol_names
        )
        all_annotations.update(annotations)
        return rf"{tab} ${lhs}$ & ${rhs}$ \\"

    table = "\n".join(
        [
            r"\begin{longtable}" + f"{{{columns}}}",
            tab + " & ".join(headers) + r" \\",
            tab + r"\hline",
            tab + r"\endhead",
        ]
        + [_render(lhs, old, new) for lhs, old, new in rows]
        + [
            tab + rf"\caption[{short_desc}]{{{long_desc}}}",
            tab + rf"\label{{table:{label}}}",
            r"\end{longtable}",
        ]
    )
    if all_annotations:
        table += "\n" + _annotation_block(list(all_annotations.items()))
    return table


@dataclass
class LatexCodegen:
    """Generated code split into four sections ready for emission."""

    parameters: list[tuple[str, sympy.Expr]]
    variables: list[tuple[str, sympy.Expr]]
    derived: list[tuple[str, sympy.Expr]]
    reactions: list[tuple[str, sympy.Expr]]
    diff_eqs: list[tuple[str, sympy.Expr]]
    long_name_cutoff: int
    symbol_names: dict[sympy.Symbol, str]

    def full(self) -> str:
        return "\n\n".join(i for i in self.as_default() if len(i) > 0)

    def as_default(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2(
                headers=("Parameter name", "Parameter value"),
                rows=[(k, v) for k, v in sorted(self.parameters)],
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.parameters) > 0
            else "",
            _tex_math_table2(
                headers=("Model name", "Initial concentration"),
                rows=[(k, v) for k, v in self.variables],
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.variables) > 0
            else "",
            _tex_dmath_blocks(
                self.derived,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.derived) > 0
            else "",
            _tex_dmath_blocks(
                self.reactions,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.reactions) > 0
            else "",
            _tex_align_block(
                self.diff_eqs,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.diff_eqs) > 0
            else "",
        )

    def as_aligned(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_align_block(
                self.parameters,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block(
                self.variables,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block(
                self.derived,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block(
                self.reactions,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block(
                self.diff_eqs,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
        )

    def as_tables(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2(
                headers=("Parameter name", "Parameter value"),
                rows=[(k, v) for k, v in sorted(self.parameters)],
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.parameters) > 0
            else "",
            _tex_math_table2(
                headers=("Model name", "Initial concentration"),
                rows=[(k, v) for k, v in self.variables],
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.variables) > 0
            else "",
            _tex_math_table2(
                headers=("Name", "Derived"),
                rows=[(k, v) for k, v in sorted(self.derived)],
                label="model-der",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.derived) > 0
            else "",
            _tex_math_table2(
                headers=("Name", "Reaction"),
                rows=[(k, v) for k, v in sorted(self.reactions)],
                label="model-rxn",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.reactions) > 0
            else "",
            _tex_math_table2(
                headers=("Lhs", "Rhs"),
                rows=[(k, v) for k, v in sorted(self.diff_eqs)],
                label="model-eqs",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if len(self.diff_eqs) > 0
            else "",
        )


def generate_model_code_latex(
    model: Model,
    *,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    gls: dict[str, str] | None = None,
    long_name_cutoff: int = 20,
) -> LatexCodegen:
    r"""Transform the model into LaTeX equations.

    Unlike the other ``generate_model_code_*`` functions, this does not
    produce executable code. The ``model`` field contains the full ODE
    system, ``derived`` contains the computed quantities, and ``inits``
    contains the initial conditions - each as a LaTeX ``align*`` block.

    Parameters
    ----------
    model
        Model to generate equations for.
    custom_fns
        Custom sympy expressions to substitute for named model functions.
    gls
        Optional mapping of model component names to custom LaTeX strings,
        e.g. ``{"km": r"K_m", "vmax": r"V_{\mathrm{max}}"}``.
    long_name_cutoff
        Symbol names with this many characters or more are replaced by short
        stand-ins (``x_0``, ``x_1``, …) in the equation body and annotated
        with a ``where`` clause.  Symbols already renamed via *gls* are
        never shortened.

    Returns
    -------
    LatexCodegen
        Structured LaTeX for parameters, variables, derived quantities,
        reactions, and differential equations.

    """
    symbol_names: dict[sympy.Symbol, str] = (
        {} if gls is None else {sympy.Symbol(orig): tex for orig, tex in gls.items()}
    )

    rsr = model_to_symbolic_repr(
        model,
        custom_fns={} if custom_fns is None else custom_fns,
        only_warn=True,
    ).reduce()

    return LatexCodegen(
        variables=[
            (
                _math_name(k),
                expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
            )
            for k, v in rsr.variables.items()
        ],
        parameters=[
            (
                _math_name(k),
                expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
            )
            for k, v in rsr.parameters.items()
        ],
        derived=[(_math_name(k), expr.expr) for k, expr in rsr.derived.items()],
        reactions=[(_math_name(k), rxn.expr) for k, rxn in rsr.reactions.items()],
        diff_eqs=[(_tex_lhs(_math_name(k)), eq) for k, eq in rsr.diffeqs.items()],
        long_name_cutoff=long_name_cutoff,
        symbol_names=symbol_names,
    )


@dataclass
class DiffLatexCodegen:
    """Diff of two models as LaTeX, split into five sections ready for emission."""

    parameters: list[tuple[str, sympy.Expr | None, sympy.Expr | None]]
    variables: list[tuple[str, sympy.Expr | None, sympy.Expr | None]]
    derived: list[tuple[str, sympy.Expr | None, sympy.Expr | None]]
    reactions: list[tuple[str, sympy.Expr | None, sympy.Expr | None]]
    diff_eqs: list[tuple[str, sympy.Expr | None, sympy.Expr | None]]
    long_name_cutoff: int
    symbol_names: dict[sympy.Symbol, str]

    def full(self) -> str:
        return "\n\n".join(i for i in self.as_default() if len(i) > 0)

    def as_default(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2_diff(
                headers=("Parameter name", "Parameter value"),
                rows=sorted(self.parameters, key=lambda x: x[0]),
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.parameters
            else "",
            _tex_math_table2_diff(
                headers=("Model name", "Initial concentration"),
                rows=self.variables,
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.variables
            else "",
            _tex_dmath_blocks_diff(
                self.derived,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.derived
            else "",
            _tex_dmath_blocks_diff(
                self.reactions,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.reactions
            else "",
            _tex_align_block_diff(
                self.diff_eqs,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.diff_eqs
            else "",
        )

    def as_aligned(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_align_block_diff(
                self.parameters,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block_diff(
                self.variables,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block_diff(
                self.derived,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block_diff(
                self.reactions,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
            _tex_align_block_diff(
                self.diff_eqs,
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            ),
        )

    def as_tables(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2_diff(
                headers=("Parameter name", "Parameter value"),
                rows=sorted(self.parameters, key=lambda x: x[0]),
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.parameters
            else "",
            _tex_math_table2_diff(
                headers=("Model name", "Initial concentration"),
                rows=self.variables,
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.variables
            else "",
            _tex_math_table2_diff(
                headers=("Name", "Derived"),
                rows=sorted(self.derived, key=lambda x: x[0]),
                label="model-der",
                short_desc="Model derived",
                long_desc="Model derived",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.derived
            else "",
            _tex_math_table2_diff(
                headers=("Name", "Reaction"),
                rows=sorted(self.reactions, key=lambda x: x[0]),
                label="model-rxn",
                short_desc="Model reactions",
                long_desc="Model reactions",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.reactions
            else "",
            _tex_math_table2_diff(
                headers=("Lhs", "Rhs"),
                rows=sorted(self.diff_eqs, key=lambda x: x[0]),
                label="model-eqs",
                short_desc="Model equations",
                long_desc="Model equations",
                long_name_cutoff=self.long_name_cutoff,
                symbol_names=self.symbol_names,
            )
            if self.diff_eqs
            else "",
        )


def generate_latex_diff(
    old: Model,
    new: Model,
    *,
    only_changes: bool = False,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    gls: dict[str, str] | None = None,
    long_name_cutoff: int = 20,
) -> DiffLatexCodegen:
    r"""Generate a LaTeX diff between two models.

    Removals are shown in red with strikethrough (``\sout``), additions in
    green.  Requires ``ulem`` and ``xcolor`` in the document preamble (both
    are included by :func:`generate_latex_document`).

    Parameters
    ----------
    old
        The baseline model.
    new
        The updated model.
    only_changes
        When ``True`` emit only rows that differ between the two models
        (additions, removals, and modifications).  When ``False`` (default)
        emit every row and highlight only the changed ones.
    custom_fns
        Custom sympy expressions to substitute for named model functions.
    gls
        Optional mapping of component names to custom LaTeX strings.
    long_name_cutoff
        Symbol names with this many characters or more are replaced by short
        stand-ins and annotated with a ``where`` clause.

    Returns
    -------
    DiffLatexCodegen
        Structured diff LaTeX for parameters, variables, derived quantities,
        reactions, and differential equations.

    """
    symbol_names: dict[sympy.Symbol, str] = (
        {} if gls is None else {sympy.Symbol(orig): tex for orig, tex in gls.items()}
    )
    _custom = {} if custom_fns is None else custom_fns

    old_rsr = model_to_symbolic_repr(old, custom_fns=_custom, only_warn=True).reduce()
    new_rsr = model_to_symbolic_repr(new, custom_fns=_custom, only_warn=True).reduce()

    def _build_rows(
        old_items: dict[str, sympy.Expr],
        new_items: dict[str, sympy.Expr],
        get_lhs: Callable[[str], str],
    ) -> list[tuple[str, sympy.Expr | None, sympy.Expr | None]]:
        rows: list[tuple[str, sympy.Expr | None, sympy.Expr | None]] = []
        for k, new_expr in new_items.items():
            old_expr = old_items.get(k)
            if old_expr is None:
                rows.append((get_lhs(k), None, new_expr))
            elif old_expr == new_expr:
                if not only_changes:
                    rows.append((get_lhs(k), old_expr, new_expr))
            else:
                rows.append((get_lhs(k), old_expr, new_expr))
        for k, old_expr in old_items.items():
            if k not in new_items:
                rows.append((get_lhs(k), old_expr, None))
        return rows

    def _var_exprs(
        items: dict[str, SymbolicVariable],
    ) -> dict[str, sympy.Expr]:
        return {
            k: v.value.expr if isinstance(v.value, SymbolicFn) else v.value
            for k, v in items.items()
        }

    def _par_exprs(
        items: dict[str, SymbolicParameter],
    ) -> dict[str, sympy.Expr]:
        return {
            k: v.value.expr if isinstance(v.value, SymbolicFn) else v.value
            for k, v in items.items()
        }

    def _fn_exprs(items: dict[str, SymbolicFn]) -> dict[str, sympy.Expr]:
        return {k: v.expr for k, v in items.items()}

    return DiffLatexCodegen(
        parameters=_build_rows(
            _par_exprs(old_rsr.parameters),
            _par_exprs(new_rsr.parameters),
            _math_name,
        ),
        variables=_build_rows(
            _var_exprs(old_rsr.variables),
            _var_exprs(new_rsr.variables),
            _math_name,
        ),
        derived=_build_rows(
            _fn_exprs(old_rsr.derived),
            _fn_exprs(new_rsr.derived),
            _math_name,
        ),
        reactions=_build_rows(
            _fn_exprs(old_rsr.reactions),
            _fn_exprs(new_rsr.reactions),
            _math_name,
        ),
        diff_eqs=_build_rows(
            old_rsr.diffeqs,
            new_rsr.diffeqs,
            lambda k: _tex_lhs(_math_name(k)),
        ),
        long_name_cutoff=long_name_cutoff,
        symbol_names=symbol_names,
    )


def generate_latex_document(
    latex: LatexCodegen | DiffLatexCodegen,
    author: str = "mxlpy",
    title: str = "Model construction",
) -> str:
    headers = (
        "Parameters",
        "Variables",
        "Derived",
        "Reactions",
        "Differential Equations",
    )
    content = _tex_list_as_sections(
        zip(
            headers,
            latex.as_default(),
            strict=True,
        )
    )
    return "\n".join(
        (
            r"\documentclass[fleqn]{article}",
            r"\usepackage[english]{babel}",
            r"\usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,marginparwidth=1.75cm]{geometry}",
            r"\usepackage{amsmath, amssymb, array, booktabs,",
            r"            breqn, caption, longtable, mathtools, placeins,",
            r"            ragged2e, tabularx, titlesec, titling, ulem, xcolor}",
            r"\newcommand{\stkout}[1]{\ifmmode\text{\sout{\ensuremath{#1}}}\else\sout{#1}\fi}",
            r"\newcommand{\sectionbreak}{\clearpage}",
            r"\setlength{\parindent}{0pt}",
            r"\allowdisplaybreaks",
            r"",
            rf"\title{{{title}}}",
            r"\date{} % clear date",
            rf"\author{{{author}}}",
            r"\begin{document}",
            r"\maketitle",
            rf"{content}",
            r"\end{document}",
        )
    )
