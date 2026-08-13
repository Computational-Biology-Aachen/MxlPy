"""Quality assurance reporting for kinetic models.

Provides a single entry point, :func:`report`, that inspects a
:class:`~mxlpy.KineticModelBuilder` and returns an HTML-renderable
:class:`QAReport` summarising model size, numerical stiffness, unused
parameters, structural conservation relations, and unit hygiene, each
with a green/orange/red grade where a meaningful threshold exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from mxlpy import mca
from mxlpy._kinetic_builder import Failure
from mxlpy.simulator import Simulator

if TYPE_CHECKING:
    from pathlib import Path

    from sympy.physics.units.quantities import Quantity

    from mxlpy._kinetic_builder import KineticModelBuilder

__all__ = ["Grade", "MetricResult", "QAReport", "report"]

type Grade = Literal["green", "orange", "red", "info"]

_GRADE_COLOR = {
    "green": "#2e7d32",
    "orange": "#ed6c02",
    "red": "#c62828",
    "info": "#616161",
}
_GRADE_LABEL = {
    "green": "Good",
    "orange": "OK",
    "red": "Poor",
    "info": "Info",
}

_STIFFNESS_NOTE = {
    "green": (
        "Low stiffness ratio: gradient-based/adjoint sensitivity methods are "
        "well-suited for fitting this model."
    ),
    "orange": (
        "Moderate stiffness ratio: prefer adjoint methods paired with "
        "implicit/stiff solvers; naive forward-mode sensitivities may be "
        "unreliable."
    ),
    "red": (
        "High stiffness ratio: gradient-based sensitivities are likely "
        "ill-conditioned here; prefer gradient-free optimisation (e.g. "
        "CMA-ES, Nelder-Mead, genetic algorithms)."
    ),
}


_SIZE_GREEN_MAX = 20
_SIZE_ORANGE_MAX = 100
_UNIT_COVERAGE_GREEN_MIN = 1.0
_UNIT_COVERAGE_ORANGE_MIN = 0.5


@dataclass(slots=True)
class MetricResult:
    """A single graded row of a quality report."""

    name: str
    value: str
    grade: Grade
    note: str = ""


def _grade_count(n: int) -> Grade:
    if n < _SIZE_GREEN_MAX:
        return "green"
    if n <= _SIZE_ORANGE_MAX:
        return "orange"
    return "red"


def _grade_ratio(ratio: float, *, low: float, high: float) -> Grade:
    if ratio < low:
        return "green"
    if ratio <= high:
        return "orange"
    return "red"


def _size_metric(name: str, n: int) -> MetricResult:
    return MetricResult(name=name, value=str(n), grade=_grade_count(n))


def _param_var_ratio_metric(n_pars: int, n_vars: int) -> MetricResult:
    if n_vars == 0:
        return MetricResult(
            name="Parameters / variables",
            value="n/a",
            grade="info",
            note="Model has no state variables.",
        )
    ratio = n_pars / n_vars
    grade = _grade_ratio(ratio, low=3, high=6)
    note = (
        ""
        if grade == "green"
        else (
            "High parameter-to-variable ratio; consider whether all "
            "parameters are practically identifiable from typical data."
        )
    )
    return MetricResult(
        name="Parameters / variables", value=f"{ratio:.2g}", grade=grade, note=note
    )


def _n_conservation_relations(model: KineticModelBuilder, n_vars: int) -> int:
    stoich = model.get_stoichiometries()
    if stoich.empty:
        return n_vars
    rank = np.linalg.matrix_rank(stoich.to_numpy())
    return max(n_vars - int(rank), 0)


def _conservation_metric(n_conserved: int) -> MetricResult:
    note = (
        "No structural conservation relations detected."
        if n_conserved == 0
        else (
            f"{n_conserved} structural conservation relation(s) detected "
            "(e.g. conserved moiety pools). These contribute near-zero "
            "eigenvalues to the Jacobian and are excluded from the "
            "stiffness ratios below."
        )
    )
    return MetricResult(
        name="Conservation relations", value=str(n_conserved), grade="info", note=note
    )


def _stiffness_ratio(
    eigenvalues: np.ndarray, n_filter: int
) -> tuple[float, float, float] | None:
    real_abs = np.sort(np.abs(eigenvalues.real))
    remaining = real_abs[n_filter:]
    remaining = remaining[remaining > 0]
    if len(remaining) == 0:
        return None
    min_eig = float(remaining.min())
    max_eig = float(remaining.max())
    return min_eig, max_eig, max_eig / min_eig


def _stiffness_metric(
    name: str, eigenvalues: np.ndarray, n_conserved: int
) -> MetricResult:
    result = _stiffness_ratio(eigenvalues, n_conserved)
    if result is None:
        return MetricResult(
            name=name,
            value="undefined",
            grade="info",
            note=(
                "No non-conserved eigenmodes remain after filtering "
                "structural conservation relations."
            ),
        )
    min_eig, max_eig, ratio = result
    grade = _grade_ratio(ratio, low=1e3, high=1e6)
    value = f"max|Re λ|={max_eig:.3g}, min|Re λ|={min_eig:.3g}, ratio={ratio:.3g}"
    return MetricResult(
        name=name, value=value, grade=grade, note=_STIFFNESS_NOTE[grade]
    )


def _initial_condition_stiffness_metric(
    model: KineticModelBuilder, n_conserved: int
) -> MetricResult:
    stability = mca.steady_state_stability(model, model.get_initial_conditions())
    return _stiffness_metric(
        "Stiffness ratio (initial conditions)", stability.eigenvalues, n_conserved
    )


def _steady_state_stiffness_metric(
    model: KineticModelBuilder, n_conserved: int
) -> MetricResult:
    sim_result = Simulator(model).simulate_to_steady_state().get_result()
    if isinstance(sim_result.value, Exception):
        return MetricResult(
            name="Stiffness ratio (steady state)",
            value="did not converge",
            grade="red",
            note=f"Steady-state search failed: {sim_result.value}",
        )
    steady_state = sim_result.value.get_new_y0()
    stability = mca.steady_state_stability(model, steady_state)
    return _stiffness_metric(
        "Stiffness ratio (steady state)", stability.eigenvalues, n_conserved
    )


def _unused_parameters_metric(model: KineticModelBuilder) -> MetricResult:
    unused = model.get_unused_parameters()
    if not unused:
        return MetricResult(name="Unused parameters", value="0", grade="green")
    return MetricResult(
        name="Unused parameters",
        value=", ".join(sorted(unused)),
        grade="red",
        note="Parameters not referenced anywhere in the model.",
    )


def _unit_coverage_metric(model: KineticModelBuilder) -> MetricResult:
    total = 0
    covered = 0
    for variable in model.get_raw_variables(as_copy=False).values():
        total += 1
        covered += variable.unit is not None
    for parameter in model.get_raw_parameters(as_copy=False).values():
        total += 1
        covered += parameter.unit is not None
    for reaction in model.get_raw_reactions(as_copy=False).values():
        total += 1
        covered += reaction.unit is not None

    if total == 0:
        return MetricResult(
            name="Unit coverage",
            value="n/a",
            grade="info",
            note="Model has no variables, parameters, or reactions.",
        )
    pct = covered / total
    grade: Grade = (
        "green"
        if pct >= _UNIT_COVERAGE_GREEN_MIN
        else "orange"
        if pct >= _UNIT_COVERAGE_ORANGE_MIN
        else "red"
    )
    return MetricResult(
        name="Unit coverage", value=f"{covered}/{total} ({pct:.0%})", grade=grade
    )


def _unit_consistency_metric(
    model: KineticModelBuilder, time_unit: Quantity | None
) -> MetricResult:
    if time_unit is None:
        return MetricResult(
            name="Unit consistency",
            value="not checked",
            grade="info",
            note=(
                "Pass time_unit=... to mxlpy.qa.report() to enable "
                "dimensional consistency checking of the differential "
                "equations."
            ),
        )

    check = model.check_units(time_unit)
    mismatches = [
        f"{var}/{rxn}"
        for var, per_rxn in check.per_variable.items()
        for rxn, result in per_rxn.items()
        if isinstance(result, Failure)
    ]
    n_checked = sum(
        1
        for per_rxn in check.per_variable.values()
        for result in per_rxn.values()
        if result is not None
    )
    if n_checked == 0:
        return MetricResult(
            name="Unit consistency",
            value="not checked",
            grade="info",
            note="No equations had complete unit annotations to verify.",
        )
    if not mismatches:
        return MetricResult(
            name="Unit consistency", value="no mismatches", grade="green"
        )
    return MetricResult(
        name="Unit consistency",
        value=f"{len(mismatches)} mismatch(es): {', '.join(mismatches)}",
        grade="red",
    )


_STYLE = """<style>
.mxlpy-qa-report { font-family: sans-serif; }
.mxlpy-qa-report table { border-collapse: collapse; font-size: 0.9em; width: 100%; }
.mxlpy-qa-report th, .mxlpy-qa-report td {
    padding: 6px 12px;
    text-align: left;
    border-bottom: 1px solid rgba(128, 128, 128, 0.3);
    vertical-align: top;
}
.mxlpy-qa-report th { border-bottom: 2px solid rgba(128, 128, 128, 0.5); }
</style>"""


def _badge(grade: Grade) -> str:
    color = _GRADE_COLOR[grade]
    label = _GRADE_LABEL[grade]
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
        f'font-weight:600;color:{color};background:{color}22;">{label}</span>'
    )


def _table_rows(metrics: list[MetricResult]) -> str:
    return "\n".join(
        "<tr>"
        f"<td>{m.name}</td>"
        f"<td>{m.value}</td>"
        f"<td>{_badge(m.grade)}</td>"
        f"<td>{m.note}</td>"
        "</tr>"
        for m in metrics
    )


def _render_fragment(metrics: list[MetricResult]) -> str:
    return (
        f"{_STYLE}"
        '<div class="mxlpy-qa-report">'
        "<h3>Model quality report</h3>"
        "<table><thead><tr>"
        "<th>Metric</th><th>Value</th><th>Grade</th><th>Notes</th>"
        "</tr></thead>"
        f"<tbody>{_table_rows(metrics)}</tbody>"
        "</table></div>"
    )


def _render_document(metrics: list[MetricResult]) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>mxlpy quality report</title></head><body>"
        f"{_render_fragment(metrics)}"
        "</body></html>"
    )


@dataclass(slots=True)
class QAReport:
    """Quality report for a kinetic model."""

    metrics: list[MetricResult]

    def __repr__(self) -> str:
        """Return default representation."""
        return "\n".join(
            f"{m.name}: {m.value} [{_GRADE_LABEL[m.grade]}]" for m in self.metrics
        )

    def _repr_html_(self) -> str:
        """Fancy IPython/Jupyter shell output."""
        return _render_fragment(self.metrics)

    def write(self, path: Path) -> None:
        """Write report to a standalone HTML file.

        Parameters
        ----------
        path
            File path to write the report to.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w+") as fp:
            fp.write(_render_document(self.metrics))


def report(model: KineticModelBuilder, time_unit: Quantity | None = None) -> QAReport:
    """Generate a quality report for a kinetic model.

    Parameters
    ----------
    model
        The model to analyse.
    time_unit
        Unit of time used by the model (e.g. ``sympy.physics.units.second``).
        If given, enables the unit-consistency check of the differential
        equations. If omitted, that row is reported as "not checked".

    Examples
    --------
        >>> from mxlpy import qa
        >>> qa.report(model)

    Returns
    -------
    QAReport
        HTML-renderable quality report.

    """
    n_vars = len(model.get_variable_names())
    n_pars = len(model.get_parameter_names())
    n_rxns = len(model.get_reaction_names())
    n_conserved = _n_conservation_relations(model, n_vars)

    metrics = [
        _size_metric("State variables", n_vars),
        _size_metric("Parameters", n_pars),
        _size_metric("Reactions", n_rxns),
        _param_var_ratio_metric(n_pars, n_vars),
        _conservation_metric(n_conserved),
        _initial_condition_stiffness_metric(model, n_conserved),
        _steady_state_stiffness_metric(model, n_conserved),
        _unused_parameters_metric(model),
        _unit_coverage_metric(model),
        _unit_consistency_metric(model, time_unit),
    ]
    return QAReport(metrics=metrics)
