"""Generate multi-language model code from a symbolic model representation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
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
    from collections.abc import Callable

    from mxlpy.model import Model

_LOGGER = logging.getLogger()


__all__ = [
    "Codegen",
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


@dataclass
class SymbolicFn:
    """Container for symbolic fn."""

    fn_name: str
    expr: sympy.Expr
    args: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)


@dataclass
class SymbolicVariable:
    """Container for symbolic variable."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SymbolicVariable):
            return False
        if self.value != value.value:  # noqa: SIM103
            return False
        return True


@dataclass
class SymbolicParameter:
    """Container for symbolic par."""

    value: sympy.Float | SymbolicFn  # initial assignment
    unit: Quantity | None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SymbolicParameter):
            return False
        if self.value != value.value:  # noqa: SIM103
            return False
        return True


@dataclass
class SymbolicReaction:
    """Container for symbolic rxn."""

    fn: SymbolicFn
    stoichiometry: dict[str, sympy.Float | SymbolicFn]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SymbolicReaction):
            return False
        if self.fn != value.fn:
            return False
        if self.stoichiometry != value.stoichiometry:  # noqa: SIM103
            return False
        return True


@dataclass
class SymbolicSurrogate:
    """Container for symbolic rxn."""

    fns: list[SymbolicFn]
    outputs: list[str]
    stoichiometry: dict[str, dict[str, sympy.Float | SymbolicFn]]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SymbolicSurrogate):
            return False
        if self.outputs != value.outputs:
            return False
        if self.stoichiometry != value.stoichiometry:
            return False
        if self.fns != value.fns:  # noqa: SIM103
            return False
        return True


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


def _mathrm(s: str) -> str:
    return rf"\mathrm{{{s}}}"


def _math_name(s: str) -> str:
    return _mathrm(valid_tex_identifier(_rename_latex(s)))


def _lhs(name: str) -> str:
    return rf"\frac{{d\ {name}}}{{dt}}"


def _tex_align_block(rows: list[tuple[str, str]]) -> str:
    body = " \\\\\n".join(f"  {lhs} &= {rhs}" for lhs, rhs in rows)
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


def _tex_math_table2(
    headers: tuple[str, str],
    rows: list[tuple[str, str]],
    label: str,
    short_desc: str,
    long_desc: str,
) -> str:
    columns = "|".join(["c"] * 2)
    tab = "    "

    return "\n".join(
        [
            r"\begin{longtable}" + f"{{{columns}}}",
            tab + " & ".join(headers) + r" \\",
            tab + r"\hline",
            tab + r"\endhead",
        ]
        + [rf"{tab} ${i}$ & ${j}$ \\" for i, j in rows]
        + [
            tab + rf"\caption[{short_desc}]{{{long_desc}}}",
            tab + rf"\label{{table:{label}}}",
            r"\end{longtable}",
        ]
    )


@dataclass
class LatexCodegen:
    """Generated code split into four sections ready for emission."""

    parameters: list[tuple[str, str]]
    variables: list[tuple[str, str]]
    derived: list[tuple[str, str]]
    reactions: list[tuple[str, str]]
    diff_eqs: list[tuple[str, str]]

    def as_default(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2(
                headers=("Parameter name", "Parameter value"),
                rows=[(k, v) for k, v in sorted(self.parameters)],
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
            )
            if len(self.parameters) > 0
            else "",
            _tex_math_table2(
                headers=("Model name", "Initial concentration"),
                rows=[(k, v) for k, v in self.variables],
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
            )
            if len(self.variables) > 0
            else "",
            _tex_align_block(self.derived) if len(self.derived) > 0 else "",
            _tex_align_block(self.reactions) if len(self.reactions) > 0 else "",
            _tex_align_block(self.diff_eqs) if len(self.diff_eqs) > 0 else "",
        )

    def as_aligned(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_align_block(self.parameters),
            _tex_align_block(self.variables),
            _tex_align_block(self.derived),
            _tex_align_block(self.reactions),
            _tex_align_block(self.diff_eqs),
        )

    def as_tables(self) -> tuple[str, str, str, str, str]:
        return (
            _tex_math_table2(
                headers=("Parameter name", "Parameter value"),
                rows=[(k, v) for k, v in sorted(self.parameters)],
                label="model-pars",
                short_desc="Model parameters",
                long_desc="Model parameters",
            )
            if len(self.parameters) > 0
            else "",
            _tex_math_table2(
                headers=("Model name", "Initial concentration"),
                rows=[(k, v) for k, v in self.variables],
                label="model-vars",
                short_desc="Model variables",
                long_desc="Model variables",
            )
            if len(self.variables) > 0
            else "",
            _tex_math_table2(
                headers=("Name", "Derived"),
                rows=[(k, v) for k, v in sorted(self.derived)],
                label="model-der",
                short_desc="Model parameters",
                long_desc="Model parameters",
            )
            if len(self.derived) > 0
            else "",
            _tex_math_table2(
                headers=("Name", "Reaction"),
                rows=[(k, v) for k, v in sorted(self.reactions)],
                label="model-rxn",
                short_desc="Model parameters",
                long_desc="Model parameters",
            )
            if len(self.reactions) > 0
            else "",
            _tex_math_table2(
                headers=("Lhs", "Rhs"),
                rows=[(k, v) for k, v in sorted(self.diff_eqs)],
                label="model-eqs",
                short_desc="Model parameters",
                long_desc="Model parameters",
            )
            if len(self.diff_eqs) > 0
            else "",
        )


def generate_model_code_latex(
    model: Model,
    *,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    gls: dict[str, str] | None = None,
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
    free_parameters
        Parameter names to treat as free (excluded from the equation body).
    derived_to_calculate
        Subset of derived component names to emit in ``derived``. ``None``
        emits everything.
    custom_fns
        Custom sympy expressions to substitute for named model functions.
    gls
        Optional mapping of model component names to custom LaTeX strings,
        e.g. ``{"km": r"K_m", "vmax": r"V_{\mathrm{max}}"}``.

    Returns
    -------
    Codegen
        ``imports`` - ``\usepackage{amsmath}``
        ``model`` - ODE system (parameters, rates, differential equations)
        ``derived`` - computed quantities
        ``inits`` - initial conditions

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
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                ),
            )
            for k, v in rsr.variables.items()
        ],
        parameters=[
            (
                _math_name(k),
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                ),
            )
            for k, v in rsr.parameters.items()
        ],
        derived=[
            (
                _math_name(k),
                _sympy_to_latex(
                    expr.expr,
                    symbol_names,
                ),
            )
            for k, expr in rsr.derived.items()
        ],
        reactions=[
            (
                _math_name(k),
                _sympy_to_latex(
                    rxn.expr,
                    symbol_names,
                ),
            )
            for k, rxn in rsr.reactions.items()
        ],
        diff_eqs=[
            (
                _lhs(_math_name(k)),
                _sympy_to_latex(
                    eq,
                    symbol_names,
                ),
            )
            for k, eq in rsr.diffeqs.items()
        ],
    )


def generate_latex_diff(
    old: Model,
    new: Model,
    *,
    show_old: bool = True,
    custom_fns: dict[str, sympy.Expr | list[sympy.Expr]] | None = None,
    gls: dict[str, str] | None = None,
) -> LatexCodegen:
    old_repr = model_to_symbolic_repr(
        old,
        custom_fns={} if custom_fns is None else custom_fns,
        only_warn=True,
    ).reduce()
    new_repr = model_to_symbolic_repr(
        new,
        custom_fns={} if custom_fns is None else custom_fns,
        only_warn=True,
    ).reduce()

    rsr = old_repr.difference(new_repr)

    symbol_names: dict[sympy.Symbol, str] = (
        {} if gls is None else {sympy.Symbol(orig): tex for orig, tex in gls.items()}
    )

    def _show_diff(key: str, old: str | None, new: str | None) -> tuple[str, str]:
        if show_old:
            if old is None:
                return key, rf"{{\color{{green}} {new}}}"
            if new is None:
                return key, rf"\sout{{{old}}}"

            return key, rf"{{\color{{green}} {old}}} {{\color{{green}} {new}}}"

        return key, rf"\sout{{{old}}}" if new is None else new

    variables = sorted(set(old_repr.variables.keys()) | set(new_repr.variables.keys()))
    parameters = sorted(
        set(old_repr.parameters.keys()) | set(new_repr.parameters.keys())
    )
    derived = sorted(set(old_repr.derived.keys()) | set(new_repr.derived.keys()))
    reactions = sorted(set(old_repr.reactions.keys()) | set(new_repr.reactions.keys()))
    diffeqs = sorted(set(old_repr.diffeqs.keys()) | set(new_repr.diffeqs.keys()))

    return LatexCodegen(
        variables=[
            _show_diff(
                valid_tex_identifier(k),
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                )
                if (v := old_repr.variables.get(k)) is not None
                else v,
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                )
                if (v := rsr.variables.get(k)) is not None
                else v,
            )
            for k in variables
        ],
        parameters=[
            _show_diff(
                valid_tex_identifier(k),
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                )
                if (v := old_repr.parameters.get(k)) is not None
                else v,
                _sympy_to_latex(
                    expr.expr if isinstance(expr := v.value, SymbolicFn) else expr,
                    symbol_names,
                )
                if (v := rsr.parameters.get(k)) is not None
                else v,
            )
            for k in parameters
        ],
        derived=[
            _show_diff(
                valid_tex_identifier(k),
                _sympy_to_latex(v.expr, symbol_names)
                if (v := old_repr.derived.get(k)) is not None
                else v,
                _sympy_to_latex(v.expr, symbol_names)
                if (v := rsr.derived.get(k)) is not None
                else v,
            )
            for k in derived
        ],
        reactions=[
            _show_diff(
                valid_tex_identifier(k),
                _sympy_to_latex(v.expr, symbol_names)
                if (v := old_repr.reactions.get(k)) is not None
                else v,
                _sympy_to_latex(v.expr, symbol_names)
                if (v := rsr.reactions.get(k)) is not None
                else v,
            )
            for k in reactions
        ],
        diff_eqs=[
            _show_diff(
                _lhs(valid_tex_identifier(k)),
                _sympy_to_latex(v, symbol_names)
                if (v := old_repr.diffeqs.get(k)) is not None
                else v,
                _sympy_to_latex(v, symbol_names)
                if (v := rsr.diffeqs.get(k)) is not None
                else v,
            )
            for k in diffeqs
        ],
    )


def _subsection_(s: str) -> str:
    # depth = 2
    return r"\FloatBarrier" + rf"\subsection*{{{s}}}"


def _latex_list_as_sections(rows: Iterable[tuple[str, str]]) -> str:
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


def generate_latex_document(
    model: Model,
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
    content = _latex_list_as_sections(
        zip(headers, generate_model_code_latex(model).as_default(), strict=True)
    )
    return rf"""\documentclass[fleqn]{{article}}
\usepackage[english]{{babel}}
\usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,marginparwidth=1.75cm]{{geometry}}
\usepackage{{amsmath, amssymb, array, booktabs,
            breqn, caption, longtable, mathtools, placeins,
            ragged2e, tabularx, titlesec, titling, xcolor}}
\newcommand{{\sectionbreak}}{{\clearpage}}
\setlength{{\parindent}}{{0pt}}
\allowdisplaybreaks

\title{{{title}}}
\date{{}} % clear date
\author{{{author}}}
\begin{{document}}
\maketitle
{content}
\end{{document}}
"""
