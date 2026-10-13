# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Unit inference for MxlPy models via bidirectional constraint propagation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy
from sympy.physics.units.prefixes import Prefix
from wadler_lindig import pformat

from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import list_of_symbols
from mxlpy.types import Derived, InitialAssignment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sympy.physics.units.quantities import Quantity

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy.surrogates.abstract import SurrogateProtocol
    from mxlpy.types import Parameter, Reaction, Readout, Variable

__all__ = [
    "Conflict",
    "LOGGER",
    "MdText",
    "UnitInference",
    "infer_units",
    "unit_of",
    "unit_of_expr",
]

LOGGER = logging.getLogger(__name__)


def unit_of(expr: sympy.Expr) -> sympy.Expr:
    """Get unit of sympy expr.

    Parameters
    ----------
    expr
        A sympy expression to extract the unit from.

    Returns
    -------
    sympy.Expr
        The unit part of the expression.

    """
    return expr.as_coeff_Mul()[1]


def unit_of_expr(
    expr: sympy.Expr,
    units: dict[sympy.Symbol, sympy.Expr],
) -> sympy.Expr | None:
    """Get unit of a symbolic expression, given the units of its symbols.

    Unlike substituting the units into the expression, terms of a sum are
    not combined arithmetically, so ``a - b`` with ``a`` and ``b`` in
    ``mol`` yields ``mol`` instead of ``0``.
    If the terms of a sum have different dimensions, the unevaluated sum of
    their units is returned, which will not match any valid unit.

    Parameters
    ----------
    expr
        A sympy expression of model symbols.
    units
        Units of the symbols in ``expr``.

    Returns
    -------
    sympy.Expr | None
        The unit of ``expr``, or None if a symbol has no known unit.

    """
    match expr:
        case sympy.Symbol():
            return units.get(expr)
        case _ if expr.is_number:
            return sympy.S.One
        case sympy.Mul():
            factors = [unit_of_expr(arg, units) for arg in expr.args]  # type: ignore[arg-type]
            known = [i for i in factors if i is not None]
            if len(known) != len(factors):
                return None
            return unit_of(sympy.Mul(*known))
        case sympy.Pow():
            base, exp = expr.args
            if not exp.is_number:
                return sympy.S.One
            if (base_unit := unit_of_expr(base, units)) is None:
                return None
            return base_unit ** _exact(exp)
        case sympy.Add() | sympy.Min() | sympy.Max():
            return _unit_of_terms(expr.args, units)  # type: ignore[arg-type]
        case sympy.Abs():
            return unit_of_expr(expr.args[0], units)
        case sympy.Piecewise():
            return _unit_of_terms([e for e, _ in expr.args], units)  # type: ignore[misc]
        case _:
            return sympy.S.One


def _exact(exp: sympy.Expr) -> sympy.Expr:
    """Convert float exponents such as ``2.0`` (from ``x**2`` in source) to rationals."""
    if isinstance(exp, sympy.Float):
        return sympy.nsimplify(exp, rational=True)
    return exp


def _is_consistent(unit: sympy.Expr) -> bool:
    """Return False if *unit* is a sum of incompatible units from :func:`unit_of_expr`."""
    return not isinstance(unit, sympy.Add)


def _unit_of_terms(
    terms: Sequence[sympy.Expr],
    units: dict[sympy.Symbol, sympy.Expr],
) -> sympy.Expr | None:
    # a literal zero (e.g. an `else: return 0` branch) is compatible with any unit
    term_units = [unit_of_expr(term, units) for term in terms if not term.is_zero]
    if not term_units:
        return sympy.S.One
    known = [i for i in term_units if i is not None]
    if len(known) != len(term_units):
        return None
    first, *rest = known
    if all(_same_dim(first, i) for i in rest):
        return first
    return sympy.Add(*known, evaluate=False)


def _solve_unit(
    expr: sympy.Expr,
    target: sympy.Expr,
    candidate: sympy.Symbol,
    units: dict[sympy.Symbol, sympy.Expr],
) -> sympy.Expr | None:
    """Get unit of *candidate* such that *expr* has unit *target*.

    Walks the expression tree like :func:`unit_of_expr`, so every term of a
    sum must have the target unit.
    Returns None if *candidate* occurs in more than one factor of a product
    or a symbol other than *candidate* has no known unit.
    Like :func:`unit_of_expr`, function arguments and non-numeric exponents
    are not constrained, as they commonly hide unit-carrying constants
    (e.g. ``log10(h * 2.5e-4)``), so no unit is inferred through them.
    """
    if candidate not in expr.free_symbols:
        return None
    match expr:
        case sympy.Symbol():
            return target
        case sympy.Mul():
            inner = [arg for arg in expr.args if candidate in arg.free_symbols]
            if len(inner) != 1:
                return None
            rest = [arg for arg in expr.args if candidate not in arg.free_symbols]
            if (rest_unit := unit_of_expr(sympy.Mul(*rest), units)) is None:
                return None
            return _solve_unit(inner[0], unit_of(target / rest_unit), candidate, units)  # type: ignore[arg-type, operator]
        case sympy.Pow():
            base, exp = expr.args
            if not exp.is_number:
                return None
            return _solve_unit(
                base, unit_of(target ** (1 / _exact(exp))), candidate, units
            )  # type: ignore[arg-type, operator]
        case sympy.Add() | sympy.Min() | sympy.Max():
            return _solve_terms(expr.args, target, candidate, units)  # type: ignore[arg-type]
        case sympy.Abs():
            return _solve_unit(expr.args[0], target, candidate, units)  # type: ignore[arg-type]
        case sympy.Piecewise():
            return _solve_terms([e for e, _ in expr.args], target, candidate, units)  # type: ignore[misc]
        case _:
            return None


def _solve_terms(
    terms: Sequence[sympy.Expr],
    target: sympy.Expr,
    candidate: sympy.Symbol,
    units: dict[sympy.Symbol, sympy.Expr],
) -> sympy.Expr | None:
    for term in terms:
        if (unit := _solve_unit(term, target, candidate, units)) is not None:
            return unit
    return None


def _latex_view(expr: sympy.Expr | None) -> str:
    if expr is None:
        return "PARSE-ERROR"
    return f"${sympy.latex(expr)}$"


def _fmt_success(s: str) -> str:
    return f"<span style='color: green'>{s}</span>"


def _fmt_failed(s: str) -> str:
    return f"<span style='color: red'>{s}</span>"


def _same_dim(a: sympy.Expr, b: sympy.Expr) -> bool:
    """Return True if *a* and *b* represent the same physical dimension.

    Units that differ only by a scale factor from prefix stripping
    (e.g. ``mmol/s`` vs ``mol/s`` differ by ``milli``) are treated
    as consistent - the discrepancy is an artefact of how
    ``sympy.solve`` normalises ``Prefix`` objects.

    We strip all ``Prefix`` atoms by substituting their ``scale_factor``
    (a rational number) and then check whether the ratio of the two
    resulting expressions is a pure number.
    """
    if a == b:
        return True
    try:

        def _strip(expr: sympy.Expr) -> sympy.Expr:
            prefix_subs = {p: p.scale_factor for p in expr.atoms(Prefix)}
            return expr.subs(prefix_subs)  # type: ignore[arg-type]

        ratio = sympy.simplify(_strip(a) / _strip(b))  # type: ignore[operator]
        return bool(ratio.is_number)
    except Exception:  # noqa: BLE001
        return a == b


@dataclass
class MdText:
    """Generic markdown text."""

    content: list[str]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def _repr_markdown_(self) -> str:
        return "\n".join(self.content)


@dataclass
class Conflict:
    """Incompatible unit constraints for the same model component.

    Attributes
    ----------
    constraints
        Mapping from source label to the unit inferred from that source.

    """

    constraints: dict[str, sympy.Expr]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def sources(self) -> list[str]:
        """Sorted list of source labels contributing to the conflict."""
        return sorted(self.constraints)


@dataclass
class UnitInference:
    """Result of :meth:`Model.infer_units`.

    Each dict maps component names to one of:

    * ``sympy.Expr`` - a successfully inferred unit
    * :class:`Conflict` - multiple incompatible unit constraints
    * ``None`` - insufficient information to infer a unit

    Explicitly-set units are included as-is.

    Attributes
    ----------
    parameters
        Inferred units for model parameters.
    variables
        Inferred units for model variables.
    reactions
        Inferred units for reaction rate functions.
    derived
        Inferred units for derived quantities.
    readouts
        Inferred units for readout functions.
    initial_assignments
        Inferred units for :class:`InitialAssignment` expressions,
        keyed by the owning parameter or variable name.

    """

    parameters: dict[str, sympy.Expr | Conflict | None]
    variables: dict[str, sympy.Expr | Conflict | None]
    reactions: dict[str, sympy.Expr | Conflict | None]
    derived: dict[str, sympy.Expr | Conflict | None]
    readouts: dict[str, sympy.Expr | Conflict | None]
    initial_assignments: dict[str, sympy.Expr | Conflict | None]

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def all_inferred(self) -> bool:
        """Return True if every component has a successfully inferred unit."""
        all_results = [
            *self.parameters.values(),
            *self.variables.values(),
            *self.reactions.values(),
            *self.derived.values(),
            *self.readouts.values(),
            *self.initial_assignments.values(),
        ]
        return all(isinstance(r, sympy.Expr) for r in all_results)

    def has_conflicts(self) -> bool:
        """Return True if any component has conflicting unit constraints."""
        all_results = [
            *self.parameters.values(),
            *self.variables.values(),
            *self.reactions.values(),
            *self.derived.values(),
            *self.readouts.values(),
            *self.initial_assignments.values(),
        ]
        return any(isinstance(r, Conflict) for r in all_results)

    def report(self) -> MdText:
        """Export inference results as a markdown report."""
        lines: list[str] = ["## Unit Inference"]
        sections: list[tuple[str, dict[str, sympy.Expr | Conflict | None]]] = [
            ("Parameters", self.parameters),
            ("Variables", self.variables),
            ("Reactions", self.reactions),
            ("Derived", self.derived),
            ("Readouts", self.readouts),
            ("Initial Assignments", self.initial_assignments),
        ]
        for section_name, entries in sections:
            if not entries:
                continue
            lines.append(f"\n### {section_name}")
            for name, result in entries.items():
                match result:
                    case None:
                        lines.append(f"\n- {name}: " + _fmt_failed("unknown"))
                    case Conflict(constraints=cs):
                        lines.append(f"\n- {name}: " + _fmt_failed("CONFLICT"))
                        for src, u in cs.items():
                            lines.append(f"  - {src}: {_latex_view(u)}")
                    case expr:
                        lines.append(f"\n- {name}: " + _fmt_success(_latex_view(expr)))
        return MdText(lines)

    def apply_to(self, model: KineticModelBuilder) -> None:
        """Write inferred units back into *model*.

        Only fills components whose current unit is ``None``.
        Components with a :class:`Conflict` result are skipped and a warning
        is logged for each.

        Parameters
        ----------
        model
            The model to update in place.

        """
        for name, result in self.parameters.items():
            if isinstance(result, Conflict):
                LOGGER.warning(
                    "infer_units: skipping parameter %r due to conflicting units: %s",
                    name,
                    result.constraints,
                )
            elif (
                isinstance(result, sympy.Expr) and model._parameters[name].unit is None  # noqa: SLF001
            ):
                model.update_parameter(name, unit=result)

        for name, result in self.variables.items():
            if isinstance(result, Conflict):
                LOGGER.warning(
                    "infer_units: skipping variable %r due to conflicting units: %s",
                    name,
                    result.constraints,
                )
            elif (
                isinstance(result, sympy.Expr) and model._variables[name].unit is None  # noqa: SLF001
            ):
                model._variables[name].unit = result  # noqa: SLF001

        for name, result in self.reactions.items():
            if isinstance(result, Conflict):
                LOGGER.warning(
                    "infer_units: skipping reaction %r due to conflicting units: %s",
                    name,
                    result.constraints,
                )
            elif (
                isinstance(result, sympy.Expr) and model._reactions[name].unit is None  # noqa: SLF001
            ):
                model.update_reaction(name, unit=result)

        for name, result in self.derived.items():
            if isinstance(result, Conflict):
                LOGGER.warning(
                    "infer_units: skipping derived %r due to conflicting units: %s",
                    name,
                    result.constraints,
                )
            elif (
                isinstance(result, sympy.Expr) and model._derived[name].unit is None  # noqa: SLF001
            ):
                model.update_derived(name, unit=result)

        for name, result in self.readouts.items():
            if isinstance(result, Conflict):
                LOGGER.warning(
                    "infer_units: skipping readout %r due to conflicting units: %s",
                    name,
                    result.constraints,
                )
            elif (
                isinstance(result, sympy.Expr) and model._readouts[name].unit is None  # noqa: SLF001
            ):
                model._readouts[name].unit = result  # noqa: SLF001


def infer_units(
    parameters: dict[str, Parameter],
    variables: dict[str, Variable],
    reactions: dict[str, Reaction],
    derived: dict[str, Derived],
    readouts: dict[str, Readout],
    time_unit: Quantity,
    *,
    surrogates: Mapping[str, SurrogateProtocol] | None = None,
) -> UnitInference:
    """Infer physical units via bidirectional constraint propagation.

    Propagates known units (explicitly set on parameters, variables, etc.)
    forward through function graphs and backward through the ODE structure
    until no further units can be derived (fixpoint).

    Components with an explicit unit act as anchors and are never overwritten.
    When two independent inferences disagree for the same component, a
    :class:`Conflict` is reported instead of a unit.

    Parameters
    ----------
    parameters
        Model parameters.
    variables
        Model variables.
    reactions
        Model reactions.
    derived
        Model derived quantities.
    readouts
        Model readouts.
    time_unit
        The unit of time used in the model
        (e.g. ``sympy.physics.units.second``).
    surrogates
        Model surrogates.  Their explicit output units act as anchors and
        their stoichiometries link outputs to variables.

    Returns
    -------
    UnitInference
        Per-component inference results.

    """
    surrogates = {} if surrogates is None else surrogates
    # all_inferences[name] = {source_label: unit}
    # The first entry is used for further propagation (get_unit).
    # All entries are used at the end for conflict detection.
    all_inferences: dict[str, dict[str, sympy.Expr]] = {}

    def get_unit(name: str) -> sympy.Expr | None:
        sources = all_inferences.get(name)
        return next(iter(sources.values())) if sources else None

    def record(name: str, unit: sympy.Expr, source: str) -> bool:
        """Record a unit inference.  Returns True on first inference only."""
        if name not in all_inferences:
            all_inferences[name] = {}
        sources = all_inferences[name]
        if source not in sources:
            sources[source] = unit
        return len(sources) == 1

    # ── Seed ─────────────────────────────────────────────────────────────
    record("time", time_unit, "seed")  # type: ignore[arg-type]
    for name, p in parameters.items():
        if p.unit is not None:
            record(name, p.unit, "explicit")
    for name, v in variables.items():
        if v.unit is not None:
            record(name, v.unit, "explicit")
    for name, d in derived.items():
        if d.unit is not None:
            record(name, d.unit, "explicit")
    for name, r in reactions.items():
        if r.unit is not None:
            record(name, r.unit, "explicit")
    for name, ro in readouts.items():
        if ro.unit is not None:
            record(name, ro.unit, "explicit")
    for surrogate in surrogates.values():
        for name, unit in surrogate.units.items():
            record(name, unit, "explicit")

    # ── Precompute helpers ───────────────────────────────────────────────
    fn_sym_cache: dict[str, sympy.Expr | None] = {}

    def get_sym(cache_key: str, fn: ..., args: list[str]) -> sympy.Expr | None:
        if cache_key not in fn_sym_cache:
            fn_sym_cache[cache_key] = fn_to_sympy_expr(
                fn, origin=cache_key, model_args=list_of_symbols(args)
            )
        return fn_sym_cache[cache_key]

    # reactions_for_variable[var] = [(flux name, stoichiometry), ...]
    reactions_for_variable: dict[str, list[tuple[str, float | Derived]]] = {
        k: [] for k in variables
    }
    flux_stoichiometries = [
        *((n, r.stoichiometry) for n, r in reactions.items()),
        *(
            (flux, stoichs)
            for surrogate in surrogates.values()
            for flux, stoichs in surrogate.stoichiometries.items()
        ),
    ]
    for rxn_name, stoichiometry in flux_stoichiometries:
        for var_name, stoich in stoichiometry.items():
            if var_name in reactions_for_variable:
                reactions_for_variable[var_name].append((rxn_name, stoich))

    def stoich_unit(
        rxn_name: str, var_name: str, stoich: float | Derived
    ) -> sympy.Expr | None:
        if not isinstance(stoich, Derived):
            return sympy.S.One
        if stoich.unit is not None:
            return stoich.unit
        arg_units = {sympy.Symbol(a): get_unit(a) for a in stoich.args}
        if any(u is None for u in arg_units.values()):
            return None
        sym = get_sym(f"stoich:{rxn_name}:{var_name}", stoich.fn, stoich.args)
        if sym is None:
            return None
        return unit_of_expr(sym, arg_units)  # type: ignore[arg-type]

    # ── Fixpoint ─────────────────────────────────────────────────────────
    changed = True
    while changed:
        changed = False

        # Forward: derived, reactions, readouts
        fn_items: list[tuple[str, object, list[str]]] = [
            *((n, d.fn, d.args) for n, d in derived.items()),
            *((n, r.fn, r.args) for n, r in reactions.items()),
            *((n, ro.fn, ro.args) for n, ro in readouts.items()),
        ]
        for comp_name, fn, args in fn_items:
            if get_unit(comp_name) is not None:
                continue
            arg_units = {sympy.Symbol(a): get_unit(a) for a in args}
            if any(u is None for u in arg_units.values()):
                continue
            sym = get_sym(comp_name, fn, args)
            if sym is None:
                continue
            try:
                inferred = unit_of_expr(sym, arg_units)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
                continue
            if inferred is None or not _is_consistent(inferred):
                continue
            if record(comp_name, inferred, f"forward:{comp_name}"):
                changed = True

        # Forward: initial assignments (treated as constraints on owning
        # parameter / variable unit)
        for name, p in parameters.items():
            if not isinstance(p.value, InitialAssignment):
                continue
            if get_unit(name) is not None:
                continue
            ia = p.value
            arg_units = {sympy.Symbol(a): get_unit(a) for a in ia.args}
            if any(u is None for u in arg_units.values()):
                continue
            sym = get_sym(f"ia:{name}", ia.fn, ia.args)
            if sym is None:
                continue
            try:
                inferred = unit_of_expr(sym, arg_units)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
                continue
            if inferred is None or not _is_consistent(inferred):
                continue
            if record(name, inferred, f"forward_ia:{name}"):
                changed = True

        for name, v in variables.items():
            if not isinstance(v.initial_value, InitialAssignment):
                continue
            if get_unit(name) is not None:
                continue
            ia = v.initial_value
            arg_units = {sympy.Symbol(a): get_unit(a) for a in ia.args}
            if any(u is None for u in arg_units.values()):
                continue
            sym = get_sym(f"ia:{name}", ia.fn, ia.args)
            if sym is None:
                continue
            try:
                inferred = unit_of_expr(sym, arg_units)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
                continue
            if inferred is None or not _is_consistent(inferred):
                continue
            if record(name, inferred, f"forward_ia:{name}"):
                changed = True

        # ODE backward: variable unit ← reaction unit * time_unit
        for var_name in variables:
            if get_unit(var_name) is not None:
                continue
            for rxn_name, stoich in reactions_for_variable.get(var_name, []):
                rxn_unit = get_unit(rxn_name)
                factor = stoich_unit(rxn_name, var_name, stoich)
                if rxn_unit is None or factor is None:
                    continue
                candidate = unit_of(rxn_unit * factor * time_unit)  # type: ignore[operator]
                if record(var_name, candidate, f"from_rxn:{rxn_name}"):
                    changed = True

        # ODE forward: reaction unit ← variable unit / time_unit
        for var_name in variables:
            var_unit = get_unit(var_name)
            if var_unit is None:
                continue
            for rxn_name, stoich in reactions_for_variable.get(var_name, []):
                if get_unit(rxn_name) is not None:
                    continue
                if (factor := stoich_unit(rxn_name, var_name, stoich)) is None:
                    continue
                rxn_unit = unit_of(var_unit / time_unit / factor)  # type: ignore[operator]
                if record(rxn_name, rxn_unit, f"from_var:{var_name}"):
                    changed = True

        # Backward: arg unit ← fn output unit (see _solve_unit).
        # For each arg candidate (unknown or already known), if all other
        # args are known we solve for the candidate.  For unknown args this
        # drives propagation; for already-known args it records an extra
        # constraint for conflict detection.
        # An explicit unit on a derived value or readout overrides its function
        # (e.g. one hiding a unit-carrying constant), so it does not constrain
        # the function's arguments.
        backward_items: list[tuple[str, object, list[str]]] = [
            *((n, r.fn, r.args) for n, r in reactions.items()),
            *((n, d.fn, d.args) for n, d in derived.items() if d.unit is None),
            *((n, ro.fn, ro.args) for n, ro in readouts.items() if ro.unit is None),
        ]
        for comp_name, fn, args in backward_items:
            out_unit = get_unit(comp_name)
            if out_unit is None:
                continue
            sym = get_sym(comp_name, fn, args)
            if sym is None:
                continue
            for candidate in args:
                other_known = [
                    a for a in args if a != candidate and get_unit(a) is not None
                ]
                if len(other_known) != len(args) - 1:
                    continue  # Need all-but-one to be known
                other_units = {sympy.Symbol(a): get_unit(a) for a in other_known}
                try:
                    inferred = _solve_unit(
                        sym,
                        out_unit,
                        sympy.Symbol(candidate),
                        other_units,  # type: ignore[arg-type]
                    )
                except Exception:  # noqa: BLE001, S112
                    continue
                if inferred is None or not _is_consistent(inferred):
                    continue
                was_unknown = get_unit(candidate) is None
                record(candidate, inferred, f"backward:{comp_name}")
                changed = changed or was_unknown

        # Extra conflict-detection pass: record inferences for already-known
        # components (does not set changed, only enriches all_inferences)
        for var_name in variables:
            var_unit_known = get_unit(var_name)
            if var_unit_known is None:
                continue
            for rxn_name, stoich in reactions_for_variable.get(var_name, []):
                rxn_unit = get_unit(rxn_name)
                factor = stoich_unit(rxn_name, var_name, stoich)
                if rxn_unit is None or factor is None:
                    continue
                candidate = unit_of(rxn_unit * factor * time_unit)  # type: ignore[operator]
                record(var_name, candidate, f"extra_from_rxn:{rxn_name}")

    # ── Build result ─────────────────────────────────────────────────────
    def make_entry(name: str) -> sympy.Expr | Conflict | None:
        sources = all_inferences.get(name)
        if not sources:
            return None
        units = list(sources.values())
        if len(units) == 1:
            return units[0]
        first = units[0]
        if all(_same_dim(u, first) for u in units[1:]):
            return first
        return Conflict(constraints=dict(sources))

    ia_result: dict[str, sympy.Expr | Conflict | None] = {}
    for name, p in parameters.items():
        if isinstance(p.value, InitialAssignment):
            ia_result[name] = make_entry(f"ia:{name}") or make_entry(name)
    for name, v in variables.items():
        if isinstance(v.initial_value, InitialAssignment):
            ia_result[name] = make_entry(f"ia:{name}") or make_entry(name)

    return UnitInference(
        parameters={n: make_entry(n) for n in parameters},
        variables={n: make_entry(n) for n in variables},
        reactions={n: make_entry(n) for n in reactions},
        derived={n: make_entry(n) for n in derived},
        readouts={n: make_entry(n) for n in readouts},
        initial_assignments=ia_result,
    )
