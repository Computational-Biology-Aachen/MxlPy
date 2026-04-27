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
from mxlpy.types import InitialAssignment

if TYPE_CHECKING:
    from sympy.physics.units.quantities import Quantity

    from mxlpy.model import Model
    from mxlpy.types import Derived, Parameter, Reaction, Readout, Variable

__all__ = ["Conflict", "LOGGER", "MdText", "UnitInference", "infer_units", "unit_of"]

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

    def apply_to(self, model: Model) -> None:
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

    Returns
    -------
    UnitInference
        Per-component inference results.

    """
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

    # ── Precompute helpers ───────────────────────────────────────────────
    fn_sym_cache: dict[str, sympy.Expr | None] = {}

    def get_sym(cache_key: str, fn: ..., args: list[str]) -> sympy.Expr | None:
        if cache_key not in fn_sym_cache:
            fn_sym_cache[cache_key] = fn_to_sympy_expr(
                fn, origin=cache_key, model_args=list_of_symbols(args)
            )
        return fn_sym_cache[cache_key]

    reactions_for_variable: dict[str, list[str]] = {k: [] for k in variables}
    for rxn_name, rxn in reactions.items():
        for var_name in rxn.stoichiometry:
            if var_name in reactions_for_variable:
                reactions_for_variable[var_name].append(rxn_name)

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
                inferred = unit_of(sym.subs(arg_units))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
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
                inferred = unit_of(sym.subs(arg_units))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
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
                inferred = unit_of(sym.subs(arg_units))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001, S112
                continue
            if record(name, inferred, f"forward_ia:{name}"):
                changed = True

        # ODE backward: variable unit ← reaction unit * time_unit
        for var_name in variables:
            if get_unit(var_name) is not None:
                continue
            for rxn_name in reactions_for_variable.get(var_name, []):
                rxn_unit = get_unit(rxn_name)
                if rxn_unit is None:
                    continue
                candidate = unit_of(rxn_unit * time_unit)  # type: ignore[operator]
                if record(var_name, candidate, f"from_rxn:{rxn_name}"):
                    changed = True

        # ODE forward: reaction unit ← variable unit / time_unit
        for var_name in variables:
            var_unit = get_unit(var_name)
            if var_unit is None:
                continue
            for rxn_name in reactions_for_variable.get(var_name, []):
                if get_unit(rxn_name) is not None:
                    continue
                rxn_unit = unit_of(var_unit / time_unit)  # type: ignore[operator]
                if record(rxn_name, rxn_unit, f"from_var:{var_name}"):
                    changed = True

        # Backward: arg unit ← fn output unit (solve symbolically).
        # For each arg candidate (unknown or already known), if all other
        # args are known we solve for the candidate.  For unknown args this
        # drives propagation; for already-known args it records an extra
        # constraint for conflict detection.
        backward_items: list[tuple[str, object, list[str]]] = [
            *((n, r.fn, r.args) for n, r in reactions.items()),
            *((n, d.fn, d.args) for n, d in derived.items()),
            *((n, ro.fn, ro.args) for n, ro in readouts.items()),
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
                other_subs = {sympy.Symbol(a): get_unit(a) for a in other_known}
                partial = sym.subs(other_subs)
                try:
                    solutions = sympy.solve(
                        partial - out_unit,  # type: ignore[operator]
                        sympy.Symbol(candidate),
                    )
                except Exception:  # noqa: BLE001, S112
                    continue
                if len(solutions) != 1:
                    continue
                sol = solutions[0]
                if sol.free_symbols or sol == 0:
                    continue
                inferred = unit_of(sol)  # type: ignore[arg-type]
                is_first = record(candidate, inferred, f"backward:{comp_name}")
                if is_first and get_unit(candidate) is not None:
                    # arg was already known - don't re-trigger propagation
                    pass
                elif is_first:
                    changed = True

        # Extra conflict-detection pass: record inferences for already-known
        # components (does not set changed, only enriches all_inferences)
        for var_name in variables:
            var_unit_known = get_unit(var_name)
            if var_unit_known is None:
                continue
            for rxn_name in reactions_for_variable.get(var_name, []):
                rxn_unit = get_unit(rxn_name)
                if rxn_unit is None:
                    continue
                candidate = unit_of(rxn_unit * time_unit)  # type: ignore[operator]
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
