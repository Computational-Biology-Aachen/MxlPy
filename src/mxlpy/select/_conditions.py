# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Detect condition-specific parameters with L1 regularisation.

One model is fitted to several conditions (mutants, treatments, ...). Each
candidate parameter gets a log fold-change per condition relative to a
reference, `k_c = k · exp(fc_c)`, which an L1 penalty `λ Σ|fc|` pulls to
zero. Scanning `λ` from strong to weak gives the order in which parameters
become condition-specific (the regularisation path). Every distinct
structure on the path is refitted without the penalty and the most
parsimonious one that fits is selected (Steiert et al. 2016,
Bioinformatics 32(17)).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations, pairwise
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy import fit
from mxlpy.fit._gradient import GradientResidual, transformed
from mxlpy.fit._joint import _JointGradient
from mxlpy.fit._single import _residual_for
from mxlpy.fit.abstract import _joint_settings, _Settings
from mxlpy.fit.losses import _is_likelihood
from mxlpy.fit.penalties import (
    GroupL1Penalty,
    L1Penalty,
    Penalty,
    collect_penalties,
    penalised,
)
from mxlpy.minimizers.abstract import OptimisationState
from mxlpy.parallel import parallelise
from mxlpy.runners import precompile
from mxlpy.select._criteria import (
    Criterion,
    FitCriteria,
    LikelihoodRatio,
    lrt,
    model_weights,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy.distributions import Distribution, Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol, Residual
    from mxlpy.types import Array

__all__ = [
    "ConditionFit",
    "ConditionSpecific",
    "Regularisation",
    "Structure",
    "condition_specific",
]

type Regularisation = Literal["l1", "adaptive", "elastic_net", "group", "fused"]
"""Penalty on the log fold-changes, see `condition_specific`."""

type Structure = dict[str, tuple[tuple[str, ...], ...]]
"""Per candidate parameter, the groups of conditions sharing one value.

The first group contains the reference condition (log fold-change 0).
"""

# Canonical, hashable form of a structure: groups per candidate, in order
type _Key = tuple[tuple[tuple[str, ...], ...], ...]

# Log fold-changes within this many smoothings of zero (or of each other)
# count as zero (equal): for the exact L1 penalty, every term whose optimum
# is zero has |∂loss/∂fc| <= λ, and the smoothed optimum of such a term with
# |∂loss/∂fc| <= 0.89 λ lies within 2ε of zero
_ZERO = 2.0

# The default grid starts this factor above λmax, so the smoothed penalty
# zeroes all fold-changes at its first strength (|∂loss/∂fc| <= 0.89 λ)
_LAMBDA_MARGIN = 1.2

_REGULARISATIONS = ("l1", "adaptive", "elastic_net", "group", "fused")
_CRITERIA = ("aic", "aicc", "bic", "lrt")

# Weights of the adaptive lasso are capped at this multiple of the
# smoothing's, for fold-changes that are (nearly) zero in the full fit
_ADAPTIVE_FLOOR = 1.0


###############################################################################
# Results
###############################################################################


@dataclass(kw_only=True)
class ConditionFit(FitCriteria):
    """Unpenalised fit of one structure of condition-specific parameters.

    Attributes
    ----------
    id
        `shared` if no parameter differs, else the differing groups joined by
        `+`, e.g. `k1[mut]+k2[mut,stress]` (`mut` and `stress` share one value
        of `k2`, different from the reference's)
    structure
        Per candidate parameter, the groups of conditions sharing a value,
        the reference's group first
    best_pars
        Estimated values: the shared parameters (the reference's values of
        the candidates) and the log fold-change of each non-reference group,
        named `k[cond,...]`
    parameters
        Value of every estimated parameter per condition (conditions x
        parameters), `k · exp(fc)` for the candidates
    n_data
        Number of data points of all conditions
    loss
        Negative log-likelihood of all data (up to a constant), infinite if
        the fit failed
    penalty
        Priors and further penalties (not the L1 penalty) at the best fit
    lambdas
        Penalty strengths whose path solution has this structure (empty for
        the shared and the full model if the path never reached them)
    objectives
        Objective (loss + penalty) of every start, infinite for failed ones.
        Only the shared model has several (`n_starts`), all other fits start
        once, warm-started from it or the path

    """

    id: str
    structure: Structure
    best_pars: dict[str, float]
    parameters: pd.DataFrame
    n_data: int
    loss: float
    penalty: float
    lambdas: list[float] = field(default_factory=list)
    objectives: tuple[float, ...] = ()

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def n_failed(self) -> int:
        """Number of starts whose fit failed."""
        return sum(not math.isfinite(o) for o in self.objectives)

    @property
    def differs(self) -> dict[str, list[str]]:
        """Per non-reference condition, the candidates differing from the reference."""
        out: dict[str, list[str]] = {}
        for k, groups in self.structure.items():
            for group in groups[1:]:
                for c in group:
                    out.setdefault(c, []).append(k)
        return out


@dataclass(kw_only=True)
class ConditionSpecific:
    """Result of `condition_specific`.

    Attributes
    ----------
    path
        Log fold-change of every candidate (column level 0) and
        non-reference condition (level 1) per penalty strength, from strong
        to weak penalty. NaN where the fit failed
    path_fits
        Per penalty strength: loss, penalty and the `structure` id of the
        path solution
    fits
        Unpenalised fit of every distinct structure by id: the shared model,
        every structure on the path and the full model
    criterion
        How the structures were compared
    best_id
        Id of the selected structure
    full_id
        Id of the full model (every candidate free in every condition)
    reference
        Name of the reference condition
    conditions
        Names of all conditions
    significance
        Significance level of the likelihood-ratio tests

    """

    path: pd.DataFrame
    path_fits: pd.DataFrame
    fits: dict[str, ConditionFit]
    criterion: Criterion
    best_id: str
    full_id: str
    reference: str
    conditions: list[str]
    significance: float
    _settings: dict[str, _Settings[Any]] = field(repr=False)

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __getitem__(self, id: str) -> ConditionFit:  # noqa: A002
        """Fit of a structure."""
        return self.fits[id]

    @property
    def best(self) -> ConditionFit:
        """Unpenalised fit of the selected structure."""
        return self.fits[self.best_id]

    @property
    def selected(self) -> dict[str, list[str]]:
        """Per condition, the candidates the selected structure lets differ."""
        differs = self.best.differs
        return {c: differs.get(c, []) for c in self.conditions if c != self.reference}

    def lrt(self, null: str, alt: str | None = None) -> LikelihoodRatio:
        """Likelihood-ratio test of structure `null` against `alt` (the full model).

        A failed fit of `null` (infinite loss) is rejected (p = 0), a failed
        fit of `alt` raises.
        """
        alt_fit = self.fits[self.full_id if alt is None else alt]
        if not math.isfinite(alt_fit.loss):
            msg = (
                f"The fit of {alt_fit.id!r} failed, so there is nothing to test against"
            )
            raise ValueError(msg)
        null_fit = self.fits[null]
        return lrt(
            null_fit.log_likelihood,
            alt_fit.log_likelihood,
            alt_fit.n_parameters - null_fit.n_parameters,
        )

    @property
    def table(self) -> pd.DataFrame:
        """All structures with their criteria.

        Columns: estimated parameters `k`, data points `n`, log-likelihood,
        AIC, AICc, BIC, `delta` and `weight` of the selection criterion
        (Akaike / BIC weights among the structures, not for likelihood-ratio
        tests), the `p_value` of the likelihood-ratio test against the full
        model (NaN if its fit failed) and the strongest penalty `lambda`
        whose path solution has the structure. Sorted by the criterion; for
        likelihood-ratio tests by `k` and log-likelihood.
        """
        full = self.fits[self.full_id]
        rows = {}
        for k, c in self.fits.items():
            df = full.n_parameters - c.n_parameters
            test = df > 0 and math.isfinite(full.loss)
            rows[k] = {
                "k": c.n_parameters,
                "n": c.n_data,
                "log_likelihood": c.log_likelihood,
                "aic": c.aic,
                "aicc": c.aicc,
                "bic": c.bic,
                "p_value": self.lrt(k).p_value if test else math.nan,
                "lambda": max(c.lambdas, default=math.nan),
            }
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "structure"
        if self.criterion == "lrt":
            return out.sort_values(["k", "log_likelihood"], ascending=[True, False])
        values = cast("pd.Series", out[self.criterion])
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        out.insert(6, "delta", values - (finite.min() if len(finite) else math.nan))
        out.insert(7, "weight", model_weights(values))
        return out.sort_values(self.criterion, kind="stable")

    def models(self, id: str | None = None) -> dict[str, Any]:  # noqa: A002
        """Copies of the models of all conditions with a structure's values set.

        The selected structure by default.
        """
        parameters = self.fits[self.best_id if id is None else id].parameters
        out = {}
        for c, s in self._settings.items():
            model: Any = deepcopy(s.model)
            # Steady-state models have no y0
            if s.y0 is not None:
                model.update_variables(s.y0)
            for k, v in parameters.loc[c].items():
                if k in s.v_names:
                    model.update_variable(k, float(v))
                elif k in s.p_names:
                    model.update_parameter(k, float(v))
            out[c] = model
        return out


###############################################################################
# Parametrisation
###############################################################################


def _fc_name(parameter: str, group: Sequence[str]) -> str:
    return f"{parameter}[{','.join(group)}]"


def _structure_id(structure: Structure) -> str:
    parts = [_fc_name(k, g) for k, groups in structure.items() for g in groups[1:]]
    return "+".join(parts) if parts else "shared"


@dataclass(kw_only=True)
class _Layout:
    """Maps the estimated values of one structure to each condition's parameters.

    The estimated values are the shared ones (`theta`) followed by one log
    fold-change per non-reference group of every candidate.
    """

    structure: Structure
    theta: list[str]
    local: list[list[str]]  # per condition, the names its model estimates
    conditions: list[str]

    def __post_init__(self) -> None:
        """Index the fold-change of every candidate in every condition."""
        self.fc_names = [
            _fc_name(k, g) for k, groups in self.structure.items() for g in groups[1:]
        ]
        self.names = [*self.theta, *self.fc_names]
        index = {k: i for i, k in enumerate(self.names)}
        # (candidate, condition) -> position of its fold-change, if not the reference's
        self.fc_index: dict[tuple[str, str], int] = {
            (k, c): index[_fc_name(k, g)]
            for k, groups in self.structure.items()
            for g in groups[1:]
            for c in g
        }
        self.theta_index = [[index[k] for k in names] for names in self.local]

    def updates(self, i: int, values: dict[str, float]) -> dict[str, float]:
        """Parameter values of condition `i`."""
        c = self.conditions[i]
        out = {}
        for k in self.local[i]:
            v = values[k]
            if (j := self.fc_index.get((k, c))) is not None:
                v *= math.exp(values[self.names[j]])
            out[k] = v
        return out

    def matrix(self, i: int, values: dict[str, float]) -> Array:
        """Derivative of condition `i`'s parameters w.r.t. the estimated values."""
        c = self.conditions[i]
        out = np.zeros((len(self.local[i]), len(self.names)))
        for row, (k, col) in enumerate(
            zip(self.local[i], self.theta_index[i], strict=True)
        ):
            if (j := self.fc_index.get((k, c))) is None:
                out[row, col] = 1.0
            else:
                factor = math.exp(values[self.names[j]])
                out[row, col] = factor
                out[row, j] = values[k] * factor
        return out

    def parameters(self, values: dict[str, float]) -> pd.DataFrame:
        """Value of every shared name per condition."""
        rows = {}
        for c in self.conditions:
            row = {}
            for k in self.theta:
                v = values[k]
                if (j := self.fc_index.get((k, c))) is not None:
                    v *= math.exp(values[self.names[j]])
                row[k] = v
            rows[c] = row
        return pd.DataFrame.from_dict(rows, orient="index")


class _ConditionsLoss:
    """Summed loss of all conditions as a function of a layout's values."""

    def __init__(self, parts: list[GradientResidual], layout: _Layout) -> None:
        self.parts = parts
        self.layout = layout

    def __call__(self, updates: dict[str, float]) -> float:
        total = 0.0
        for i, part in enumerate(self.parts):
            total += part(self.layout.updates(i, updates))
            if not math.isfinite(total):
                return math.inf
        return total


class _ConditionsGradient(_JointGradient):
    """`_JointGradient` whose parts see the conditions' parameters of a layout."""

    def __init__(self, parts: list[GradientResidual], layout: _Layout) -> None:
        self.loss = cast("Residual", _ConditionsLoss(parts, layout))
        self.parts = parts
        self.layout = layout

    def _sub(self, i: int, updates: dict[str, float]) -> dict[str, float]:
        return self.layout.updates(i, updates)

    def _pull(self, i: int, updates: dict[str, float], deriv: Array) -> Array:
        return deriv @ self.layout.matrix(i, updates)

    def _pull_hessian(self, i: int, updates: dict[str, float], hess: Array) -> Array:
        m = self.layout.matrix(i, updates)
        return m.T @ hess @ m


###############################################################################
# Fitting
###############################################################################


@dataclass(kw_only=True)
class _Problem:
    """Shared settings of all fits of a condition-specific selection."""

    settings: list[_Settings[Any]]
    conditions: list[str]
    theta: list[str]
    minimizer: MinimizerProtocol
    bounds: Bounds
    penalties: list[Penalty]
    transform: Transform | None
    max_fc: float

    def layout(self, structure: Structure) -> _Layout:
        return _Layout(
            structure=structure,
            theta=self.theta,
            local=[[*s.p_names, *s.v_names] for s in self.settings],
            conditions=self.conditions,
        )

    def residual(self, layout: _Layout) -> _ConditionsGradient:
        parts = [
            GradientResidual(_residual_for(s.data, s.protocol), s)
            for s in self.settings
        ]
        return _ConditionsGradient(parts, layout)

    def fit(
        self,
        structure: Structure,
        start: dict[str, float],
        l1: Penalty | None = None,
    ) -> tuple[float, float, dict[str, float]] | None:
        """Loss, penalty (without `l1`) and values of a fit, None if it failed."""
        layout = self.layout(structure)
        p0 = {k: start[k] for k in layout.names}
        bounds = dict(self.bounds) | dict.fromkeys(
            layout.fc_names, (-self.max_fc, self.max_fc)
        )
        residual: Any = self.residual(layout)
        penalties = [*self.penalties, *([] if l1 is None else [l1])]
        inner = residual
        if penalties:
            residual = penalised(residual, penalties)
        # Only the shared values, as the fold-changes are already logarithmic
        transform = self.transform
        if isinstance(transform, str):
            transform = cast("Transform", dict.fromkeys(self.theta, transform))
        residual, p0_t, bounds_t, back = transformed(residual, p0, bounds, transform)
        match self.minimizer(residual, p0_t, bounds_t).value:
            case OptimisationState(parameters):
                best = back(parameters)
                loss = float(inner(best))
                penalty = sum(
                    float(p(np.array([best[k] for k in p.names], dtype=float)))
                    for p in self.penalties
                )
                if not math.isfinite(loss + penalty):
                    return None
                return loss, float(penalty), best
            case _:
                return None

    def gradient(self, structure: Structure, values: dict[str, float]) -> Array:
        """Gradient of the loss (without penalties) w.r.t. a layout's values."""
        layout = self.layout(structure)
        return self.residual(layout).gradient({k: values[k] for k in layout.names})


def _fit_task(
    task: tuple[_Problem, Structure, dict[str, float]],
) -> tuple[float, float, dict[str, float]] | None:
    problem, structure, start = task
    return problem.fit(structure, start)


###############################################################################
# Structures
###############################################################################


def _canonical(
    groups: Sequence[Sequence[str]], reference: str, order: list[str]
) -> tuple[tuple[str, ...], ...]:
    """Groups sorted by condition order, the reference's group first."""
    sorted_groups = [tuple(sorted(g, key=order.index)) for g in groups]
    sorted_groups.sort(key=lambda g: (reference not in g, order.index(g[0])))
    return tuple(sorted_groups)


def _structure(key: _Key, candidates: list[str]) -> Structure:
    return dict(zip(candidates, key, strict=True))


def _clusters(values: dict[str, float], tol: float) -> list[list[str]]:
    """Groups of conditions whose values are within `tol` (single linkage)."""
    names = sorted(values, key=values.__getitem__)
    groups = [[names[0]]]
    for a, b in pairwise(names):
        if values[b] - values[a] <= tol:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


@dataclass(kw_only=True)
class _Detector:
    """Structure of a path solution and the L1 penalty of a regularisation."""

    regularisation: Regularisation
    candidates: list[str]
    conditions: list[str]
    reference: str
    smoothing: float
    l1_ratio: float

    @property
    def others(self) -> list[str]:
        return [c for c in self.conditions if c != self.reference]

    def full(self) -> _Key:
        return tuple(
            ((self.reference,), *((c,) for c in self.others)) for _ in self.candidates
        )

    def shared(self) -> _Key:
        return tuple(
            (tuple(self.conditions),) for _ in self.candidates
        )  # reference first, as conditions keep their order

    def fc(self, values: dict[str, float], k: str) -> dict[str, float]:
        """Log fold-changes of a candidate per condition in the full layout."""
        return {self.reference: 0.0} | {
            c: values[_fc_name(k, (c,))] for c in self.others
        }

    def detect(self, values: dict[str, float]) -> _Key:
        """Structure of a solution of the full layout."""
        tol = _ZERO * self.smoothing
        out = []
        for k in self.candidates:
            fc = self.fc(values, k)
            match self.regularisation:
                case "fused":
                    groups = _clusters(fc, tol)
                case "group":
                    norm = math.sqrt(sum(v * v for v in fc.values()))
                    groups = (
                        [self.conditions]
                        if norm <= tol
                        else [[self.reference], *([c] for c in self.others)]
                    )
                case _:
                    zero = [c for c in self.conditions if abs(fc[c]) <= tol]
                    groups = [zero, *([c] for c in self.conditions if c not in zero)]
            out.append(_canonical(groups, self.reference, self.conditions))
        return tuple(out)

    def fc_names(self) -> list[str]:
        return [_fc_name(k, (c,)) for k in self.candidates for c in self.others]

    def penalty(self, weight: float, scales: Array | None) -> Penalty:
        """The L1 penalty on the full layout's fold-changes with strength `weight`."""
        names = self.fc_names()
        match self.regularisation:
            case "group":
                return GroupL1Penalty(
                    {
                        k: [_fc_name(k, (c,)) for c in self.others]
                        for k in self.candidates
                    },
                    weight,
                    smoothing=self.smoothing,
                )
            case "fused":
                return L1Penalty(
                    names,
                    weight,
                    matrix=self._fused_matrix(),
                    smoothing=self.smoothing,
                )
            case "elastic_net":
                return L1Penalty(
                    names, weight, l1_ratio=self.l1_ratio, smoothing=self.smoothing
                )
            case _:
                return L1Penalty(names, weight, scales=scales, smoothing=self.smoothing)

    def _fused_matrix(self) -> Array:
        """Rows `fc_a - fc_b` for all pairs of conditions per candidate (fc_ref = 0)."""
        names = self.fc_names()
        index = {k: i for i, k in enumerate(names)}
        rows = []
        for k in self.candidates:
            for a, b in combinations(self.conditions, 2):
                row = np.zeros(len(names))
                if a != self.reference:
                    row[index[_fc_name(k, (a,))]] += 1.0
                if b != self.reference:
                    row[index[_fc_name(k, (b,))]] -= 1.0
                rows.append(row)
        return np.array(rows)

    def lambda_max(self, gradient: Array, scales: Array | None) -> float:
        """Penalty strength from which all fold-changes are zero.

        From the gradient of the loss w.r.t. the fold-changes at the shared
        fit. Exact for the lasso, adaptive lasso, elastic net and group lasso,
        an upper bound for the fused lasso.
        """
        g = np.abs(gradient)
        match self.regularisation:
            case "group":
                n = len(self.others)
                norms = np.linalg.norm(g.reshape(len(self.candidates), n), axis=1)
                return float(norms.max() / math.sqrt(n))
            case "elastic_net":
                return float(g.max() / max(self.l1_ratio, 1e-12))
            case "adaptive":
                return float(np.max(g / cast("Array", scales)))
            case _:
                return float(g.max())


def _start_for(
    key: _Key,
    candidates: list[str],
    theta: dict[str, float],
    fc: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Start of a structure: shared values and the mean fold-change of each group."""
    out = dict(theta)
    for k, groups in zip(candidates, key, strict=True):
        for g in groups[1:]:
            out[_fc_name(k, g)] = float(np.mean([fc[k][c] for c in g]))
    return out


###############################################################################
# Entry point
###############################################################################


def _check(
    conditions: Mapping[str, fit.FitSettings[Any]],
    *,
    p0: Mapping[str, float | Distribution],
    candidates: Sequence[str],
    reference: str,
    regularisation: str,
    lambdas: Sequence[float] | int,
    criterion: str,
    significance: float,
    l1_ratio: float,
    gamma: float,
    max_fold_change: float,
    n_starts: int,
    bounds: Bounds | None,
    transform: Transform | None,
) -> None:
    if len(conditions) < 2:  # noqa: PLR2004
        msg = "Condition-specific parameters need at least two conditions"
        raise ValueError(msg)
    if reference not in conditions:
        msg = f"Reference {reference!r} isn't one of the conditions {list(conditions)}"
        raise ValueError(msg)
    if not candidates:
        msg = "No candidate parameters"
        raise ValueError(msg)
    if len(set(candidates)) != len(candidates):
        msg = "Candidates have to be unique"
        raise ValueError(msg)
    if unknown := sorted(set(candidates) - set(p0)):
        msg = f"Candidates have to be estimated (in p0): {unknown}"
        raise ValueError(msg)
    if regularisation not in _REGULARISATIONS:
        msg = (
            f"Unknown regularisation {regularisation!r}, use one of {_REGULARISATIONS}"
        )
        raise ValueError(msg)
    if criterion not in _CRITERIA:
        msg = f"Unknown criterion {criterion!r}, use one of {_CRITERIA}"
        raise ValueError(msg)
    if isinstance(lambdas, int):
        if lambdas < 1:
            msg = f"Need at least one penalty strength, got {lambdas}"
            raise ValueError(msg)
    elif not lambdas or any(not (math.isfinite(x) and x > 0) for x in lambdas):
        msg = "Penalty strengths have to be positive and finite"
        raise ValueError(msg)
    if not 0 < significance < 1:
        msg = f"significance has to be within (0, 1), got {significance}"
        raise ValueError(msg)
    if not 0 < l1_ratio <= 1:
        msg = f"l1_ratio has to be within (0, 1], got {l1_ratio}"
        raise ValueError(msg)
    if not (math.isfinite(gamma) and gamma > 0):
        msg = f"gamma has to be positive, got {gamma}"
        raise ValueError(msg)
    if not max_fold_change > 1:
        msg = f"max_fold_change has to be above 1, got {max_fold_change}"
        raise ValueError(msg)
    if n_starts < 0:
        msg = f"n_starts can't be negative, got {n_starts}"
        raise ValueError(msg)
    named: dict[str, Mapping[str, object]] = {"bounds": bounds or {}}
    if isinstance(transform, dict):
        named["transform"] = transform
    for arg, mapping in named.items():
        if unknown := sorted(set(mapping) - set(p0)):
            msg = f"{arg} of parameters that aren't estimated (not in p0): {unknown}"
            raise ValueError(msg)


def _nominal(
    p0: Mapping[str, float | Distribution], settings: Mapping[str, _Settings[Any]]
) -> dict[str, float]:
    """Start values: fixed ones as given, the models' values for distributions."""
    out = {}
    for k, v in p0.items():
        if isinstance(v, int | float):
            out[k] = float(v)
            continue
        for s in settings.values():
            if k in s.p_names:
                out[k] = float(s.model.get_parameter_values()[k])
                break
            if k in s.v_names:
                model = cast(Any, s.model)
                out[k] = float({**model.get_initial_conditions(), **(s.y0 or {})}[k])
                break
    return out


def _random_starts(
    p0: Mapping[str, float | Distribution],
    nominal: dict[str, float],
    bounds: Bounds,
    n: int,
    seed: int,
) -> list[dict[str, float]]:
    """Random starts drawn from the distributions of `p0`, clipped to the bounds."""
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    columns = {
        k: np.full(n, nominal[k])
        if isinstance(v, int | float)
        else np.asarray(v.sample(n, rng), dtype=float)
        for k, v in p0.items()
    }
    starts = []
    for i in range(n):
        start = {}
        for k, col in columns.items():
            lo, hi = bounds.get(k, (None, None))
            start[k] = float(np.clip(col[i], lo, hi))
        starts.append(start)
    return starts


def _select(
    fits: dict[str, ConditionFit],
    full_id: str,
    criterion: Criterion,
    significance: float,
) -> str:
    finite = {k: v for k, v in fits.items() if math.isfinite(v.loss)}
    if not finite:
        msg = "All fits failed"
        raise RuntimeError(msg)
    if criterion != "lrt":
        return min(finite, key=lambda k: finite[k].criterion(criterion))
    full = fits[full_id]
    if not math.isfinite(full.loss):
        msg = (
            "The fit of the full model failed, so there is nothing to test against; "
            "use an information criterion"
        )
        raise RuntimeError(msg)
    accepted = [
        k
        for k, v in finite.items()
        if k == full_id
        or not lrt(
            v.log_likelihood, full.log_likelihood, full.n_parameters - v.n_parameters
        ).rejects(significance)
    ]
    return min(accepted, key=lambda k: (finite[k].n_parameters, finite[k].loss))


def condition_specific(
    conditions: Mapping[str, fit.FitSettings[Any]],
    p0: Mapping[str, float | Distribution],
    candidates: Sequence[str],
    *,
    loss_fn: LossFn,
    reference: str | None = None,
    regularisation: Regularisation = "l1",
    lambdas: Sequence[float] | int = 20,
    criterion: Criterion = "lrt",
    significance: float = 0.05,
    l1_ratio: float = 0.5,
    gamma: float = 1.0,
    smoothing: float = 1e-3,
    max_fold_change: float = 1e3,
    n_starts: int = 0,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    runner: Any | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    transform: Transform | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int = 0,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> ConditionSpecific:
    """Find which parameters have to differ between conditions.

    Each candidate parameter `k` gets a log fold-change per non-reference
    condition, `k_c = k · exp(fc_c)`, and all conditions are fitted jointly
    with an L1 penalty `λ Σ|fc|` that pulls the fold-changes to zero:

    1. Fit the shared model (all fold-changes 0, from `p0` and `n_starts`
       random starts) and the full model (all free, unpenalised).
    2. Solve the regularisation path from strong to weak penalty, each fit
       warm-started from the previous one. By default 20 strengths from just
       above `λ_max` (from which all fold-changes are zero) down to
       `λ_max / 1000`.
    3. Refit every distinct structure on the path without the penalty, with
       zero fold-changes fixed (in parallel).
    4. Select the most parsimonious structure the likelihood-ratio test
       against the full model doesn't reject (`criterion="lrt"`), or the one
       with the best information criterion.

    The penalty is smoothed (`fit.L1Penalty`), so every minimizer works;
    fold-changes within `2 · smoothing` of zero count as zero.

    Parameters
    ----------
    conditions
        Model and data per condition, e.g. `fit.FitSettings(model, data)`.
        A Series fits a steady state, a DataFrame a time course, or a
        protocol time course if the settings have a `protocol`. The models
        may differ (e.g. a knock-out), but all have to have the candidates.
        A per-condition `loss_fn` has to be a likelihood as well
    p0
        Estimated parameters (and initial values), shared by all conditions,
        with their start value or the distribution random starts are drawn
        from (starting from the model's value). Each model estimates those
        it has
    candidates
        Parameters (or initial values) of `p0` allowed to differ between
        conditions, with positive values
    loss_fn
        A likelihood loss, e.g. `fit.GaussianNll(sigma)`
    reference
        Condition the fold-changes are relative to, the first by default
    regularisation
        `l1` (lasso), `adaptive` (lasso weighted by `1 / |fc|^gamma` of the
        full fit), `elastic_net` (with `l1_ratio`), `group` (all
        fold-changes of a parameter together, so it differs in all
        conditions or none) or `fused` (differences between all pairs of
        conditions, so conditions can share a value different from the
        reference's)
    lambdas
        Penalty strengths, or the number of strengths of the default grid
    criterion
        `lrt` (default), `aic`, `aicc` or `bic`
    significance
        Significance level of the likelihood-ratio tests
    l1_ratio
        Share of the L1 term of the elastic net, in (0, 1]
    gamma
        Exponent of the adaptive lasso's weights
    smoothing
        Smoothing `ε` of the L1 penalty, in log fold-change units
    max_fold_change
        Bound of the fold-changes, `|fc| <= ln(max_fold_change)`
    n_starts
        Random starts of the shared fit (in parallel), in addition to the
        start values. All other fits are warm-started from its optimum
    minimizer
        Minimizer of all fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of all conditions, unless their settings have one
    runner
        How the models are simulated, unless their settings have one
    bounds
        Bounds of the shared parameters (the reference's values)
    prior
        Priors of shared parameters, added to every fit's objective but not
        to the criteria
    penalty
        Further penalties of shared parameters, like `prior`
    transform
        Log transform of the shared parameters, see `mxlpy.fit`. Fold-changes
        are already logarithmic
    postprocess_fn
        Applied to every simulation before the loss
    seed
        Seed of the random starts
    parallel
        Whether to run the random starts and refits in parallel
    max_workers
        Maximum number of worker processes
    timeout
        Maximum time (in seconds) of one random start or refit

    Returns
    -------
    ConditionSpecific
        Regularisation path, refits of all structures and the selected one

    Examples
    --------
        >>> res = select.condition_specific(
        ...     {"wt": fit.FitSettings(model, df_wt),
        ...      "mut": fit.FitSettings(model, df_mut)},
        ...     p0={"k1": 1.0, "k2": 0.5},
        ...     candidates=["k1", "k2"],
        ...     loss_fn=fit.GaussianNll(sigma=0.05),
        ... )
        >>> res.selected
        {'mut': ['k2']}
        >>> plot.condition_specific_path(res)

    """
    names = list(conditions)
    reference = names[0] if reference is None else reference
    _check(
        conditions,
        p0=p0,
        candidates=candidates,
        reference=reference,
        regularisation=regularisation,
        lambdas=lambdas,
        criterion=criterion,
        significance=significance,
        l1_ratio=l1_ratio,
        gamma=gamma,
        max_fold_change=max_fold_change,
        n_starts=n_starts,
        bounds=bounds,
        transform=transform,
    )

    # The reference first, so it leads every structure's first group
    order = [reference, *(c for c in names if c != reference)]
    settings = {
        c: _joint_settings(
            conditions[c],
            p0=dict.fromkeys(p0, 0.0),
            y0=y0,
            runner=runner,
            loss_fn=loss_fn,
            postprocess_fn=postprocess_fn,
            as_deepcopy=True,
            standard_scale=False,
            needs_dynamics=not isinstance(conditions[c].data, pd.Series)
            or conditions[c].protocol is not None,
        )
        for c in order
    }
    for c, s in settings.items():
        if not _is_likelihood(s.loss_fn):
            msg = f"Selecting condition-specific parameters needs a likelihood loss like fit.GaussianNll(sigma), not the one of condition {c!r}"
            raise ValueError(msg)
        if missing := sorted(set(candidates) - set(s.p_names) - set(s.v_names)):
            msg = f"The model of condition {c!r} doesn't have the candidates {missing}"
            raise ValueError(msg)
    if unknown := sorted(
        set(p0) - {k for s in settings.values() for k in [*s.p_names, *s.v_names]}
    ):
        msg = f"p0 has names that are neither parameters nor variables of the models: {unknown}"
        raise ValueError(msg)
    theta = _nominal(p0, settings)
    bounds = {} if bounds is None else dict(bounds)
    starts = [theta, *_random_starts(p0, theta, bounds, n_starts, seed)]
    if bad := sorted({k for st in starts for k in candidates if not st[k] > 0}):
        msg = f"Fold-changes need positive values of the candidates, not of {bad} (start values or random starts)"
        raise ValueError(msg)
    for s in settings.values():
        precompile(s.runner, s.model)

    candidate_list = list(candidates)
    detector = _Detector(
        regularisation=regularisation,
        candidates=candidate_list,
        conditions=order,
        reference=reference,
        smoothing=smoothing,
        l1_ratio=l1_ratio,
    )
    problem = _Problem(
        settings=list(settings.values()),
        conditions=order,
        theta=list(theta),
        minimizer=fit.LeastSquaresMinimizer() if minimizer is None else minimizer,
        bounds=bounds,
        penalties=collect_penalties(prior, penalty, list(theta)),
        transform=transform,
        max_fc=math.log(max_fold_change),
    )
    n_data = sum(
        int(np.isfinite(s.data.to_numpy(dtype=float)).sum()) for s in settings.values()
    )

    outcomes: dict[_Key, tuple[float, float, dict[str, float]] | None] = {}
    shared_key, full_key = detector.shared(), detector.full()
    structure = partial(_structure, candidates=candidate_list)

    # 1. Shared model (multistart) and full model
    shared_runs = dict(
        parallelise(
            _fit_task,
            inputs=[
                (i, (problem, structure(shared_key), st)) for i, st in enumerate(starts)
            ],
            parallel=parallel and (len(starts) > 1 or timeout is not None),
            max_workers=max_workers,
            timeout=timeout,
            disable_tqdm=len(starts) == 1,
            tqdm_desc="shared fits",
        )
    )
    shared_objectives = tuple(
        math.inf if (r := shared_runs.get(i)) is None else r[0] + r[1]
        for i in range(len(starts))
    )
    done = [r for r in shared_runs.values() if r is not None]
    if not done:
        msg = "The fit of the shared model (no condition-specific parameters) failed"
        raise RuntimeError(msg)
    outcomes[shared_key] = shared = min(done, key=lambda r: r[0] + r[1])
    fc0 = dict.fromkeys(detector.fc_names(), 0.0)
    start = shared[2] | fc0
    full = problem.fit(structure(full_key), start)

    scales = None
    if regularisation == "adaptive":
        if full is None:
            msg = "The adaptive lasso needs a fit of the full model, which failed"
            raise RuntimeError(msg)
        floor = _ADAPTIVE_FLOOR * smoothing
        scales = np.array(
            [max(abs(full[2][k]), floor) ** -gamma for k in detector.fc_names()]
        )

    # 2. Regularisation path
    if isinstance(lambdas, int):
        gradient = problem.gradient(structure(full_key), start)[len(theta) :]
        lambda_max = detector.lambda_max(gradient, scales)
        if not (math.isfinite(lambda_max) and lambda_max > 0):
            msg = f"Can't scale the penalty: maximal penalty strength {lambda_max}"
            raise RuntimeError(msg)
        grid = np.geomspace(_LAMBDA_MARGIN * lambda_max, lambda_max / 1e3, lambdas)
    else:
        grid = np.sort(np.asarray(lambdas, dtype=float))[::-1]

    path_rows: list[dict[str, float]] = []
    path_info: list[dict[str, Any]] = []
    seen: dict[_Key, list[float]] = {}
    first_values: dict[_Key, dict[str, float]] = {}
    for lam in grid:
        l1 = detector.penalty(float(lam), scales)
        res = problem.fit(structure(full_key), start, l1)
        if res is None:
            path_rows.append(dict.fromkeys(detector.fc_names(), math.nan))
            path_info.append({"loss": math.inf, "penalty": math.nan, "structure": None})
            continue
        loss, _, values = res
        start = values
        key = detector.detect(values)
        seen.setdefault(key, []).append(float(lam))
        first_values.setdefault(key, values)
        path_rows.append({k: values[k] for k in detector.fc_names()})
        path_info.append(
            {
                "loss": loss,
                "penalty": l1(np.array([values[k] for k in l1.names])),
                "structure": _structure_id(structure(key)),
            }
        )

    # 3. Unpenalised refits of all structures on the path. A failed fit of the
    # full model is retried from the weakest penalty's solution
    if full is not None:
        outcomes[full_key] = full
    elif full_key not in first_values:
        first_values[full_key] = start
    tasks = []
    for key, values in first_values.items():
        if key in outcomes:
            continue
        fc = {k: detector.fc(values, k) for k in candidate_list}
        theta_values = {k: values[k] for k in theta}
        tasks.append(
            (
                key,
                (
                    problem,
                    structure(key),
                    _start_for(key, candidate_list, theta_values, fc),
                ),
            )
        )
    outcomes |= dict(
        parallelise(
            _fit_task,
            inputs=tasks,
            parallel=parallel and (len(tasks) > 1 or timeout is not None),
            max_workers=max_workers,
            timeout=timeout,
            tqdm_desc="refits",
        )
    )

    fits: dict[str, ConditionFit] = {}
    for key in [shared_key, *seen, full_key]:
        s = structure(key)
        sid = _structure_id(s)
        if sid in fits:
            continue
        layout = problem.layout(s)
        if (out := outcomes.get(key)) is None:
            loss, pen = math.inf, 0.0
            best = dict.fromkeys(layout.names, math.nan)
        else:
            loss, pen, best = out
        fits[sid] = ConditionFit(
            id=sid,
            structure=s,
            best_pars={k: float(best[k]) for k in layout.names},
            parameters=layout.parameters(best),
            n_data=n_data,
            loss=float(loss),
            penalty=float(pen),
            lambdas=seen.get(key, []),
            objectives=shared_objectives if key == shared_key else (loss + pen,),
        )

    full_id = _structure_id(structure(full_key))
    columns = pd.MultiIndex.from_tuples(
        [(k, c) for k in candidate_list for c in detector.others],
        names=["parameter", "condition"],
    )
    path = pd.DataFrame(
        [[row[_fc_name(k, (c,))] for k, c in columns] for row in path_rows],
        index=pd.Index(grid, name="lambda"),
        columns=columns,
    )
    return ConditionSpecific(
        path=path,
        path_fits=pd.DataFrame(path_info, index=pd.Index(grid, name="lambda")),
        fits=fits,
        criterion=criterion,
        best_id=_select(fits, full_id, criterion, significance),
        full_id=full_id,
        reference=reference,
        conditions=order,
        significance=significance,
        _settings=settings,
    )
