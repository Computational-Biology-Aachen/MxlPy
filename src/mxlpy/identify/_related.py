# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Functionally related parameters from the parameter paths along profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mxlpy.identify._profile import ProfileLikelihood
    from mxlpy.types import Array

__all__ = [
    "ParameterRelation",
    "RelatedParameters",
    "related_parameters",
]

# Smallest R² improvement worth adding another parameter for
_MIN_GAIN = 1e-6
# Fewest points within the threshold to regress on
_MIN_POINTS = 3


@dataclass(frozen=True, kw_only=True)
class ParameterRelation:
    """A combination of parameters that stays constant along a profile.

    `Σ cⱼ uⱼ ≈ const` along the profile of `profile`, with `uⱼ = log10 θⱼ`
    for log parameters and `θⱼ / |θⱼ*|` for the others. As the fit doesn't
    change along the profile, this combination is what the data determine:
    with log parameters, `∏ θⱼ^cⱼ` is identifiable while the parameters
    themselves aren't (e.g. `k2 · k3` for coefficients 1, 1).

    The coefficients describe the path of this profile. If the flat
    direction has more than one dimension (e.g. only `k1 · a · b` is
    identified), parameters moving in lockstep along the path can't be told
    apart: one of them gets the coefficient and the others are listed as
    `collinear`. Use `RelatedParameters.groups`, or
    `fisher_information(...).non_identifiable()` for the combinations.

    Attributes
    ----------
    profile
        Parameter whose profile the relation was found on, with coefficient 1
    coefficients
        Coefficient per parameter of the relation
    collinear
        Further parameters whose paths the relation's parameters explain
        (R² ≥ the threshold), so they belong to the same group
    r2
        Fraction of the variance of the profiled coordinate the other
        parameters explain
    log_parameters
        Parameters of the relation whose coordinates are log10 values

    """

    profile: str
    coefficients: pd.Series
    collinear: list[str]
    r2: float
    log_parameters: list[str]

    @property
    def parameters(self) -> list[str]:
        """Parameters of the relation, including the collinear ones."""
        return [*map(str, self.coefficients.index), *self.collinear]

    def __str__(self) -> str:
        """The relation as an equation."""
        terms = [
            f"{c:+.3g}·{f'log10({k})' if k in self.log_parameters else k}"
            for k, c in self.coefficients.items()
        ]
        out = " ".join(terms).lstrip("+") + " ≈ const"
        if self.collinear:
            out += f" ({', '.join(self.collinear)} collinear)"
        return out


@dataclass(kw_only=True)
class RelatedParameters:
    """Result of `related_parameters`.

    Attributes
    ----------
    relations
        Relation per profile that has one

    """

    relations: list[ParameterRelation]

    @property
    def groups(self) -> list[list[str]]:
        """Groups of related parameters, merging relations that share parameters."""
        groups: list[set[str]] = []
        for rel in self.relations:
            members = set(rel.parameters)
            overlapping = [g for g in groups if g & members]
            for g in overlapping:
                members |= g
                groups.remove(g)
            groups.append(members)
        return sorted((sorted(g) for g in groups), key=lambda g: g[0])


def _r2(x: Array, y: Array) -> tuple[float, Array]:
    """R² and coefficients of the least-squares fit `y ≈ a + x b`."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    ss_res = float(np.sum((y - design @ coef) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot, coef[1:]


def _relation(
    coords: pd.DataFrame,
    name: str,
    r2: float,
    min_change: float,
    log: list[str],
) -> ParameterRelation | None:
    """Greedy forward selection of the parameters explaining the profiled one."""
    y = coords[name].to_numpy()
    if len(y) < _MIN_POINTS or np.ptp(y) == 0:
        return None
    candidates = [
        k
        for k in coords.columns
        if k != name and np.ptp(coords[k].to_numpy()) >= min_change
    ]
    chosen: list[str] = []
    best, coef = 0.0, np.zeros(0)
    while best < r2 and candidates:
        scores = {k: _r2(coords[[*chosen, k]].to_numpy(), y) for k in candidates}
        k = max(scores, key=lambda c: scores[c][0])
        if scores[k][0] - best < _MIN_GAIN:
            break
        chosen.append(k)
        candidates.remove(k)
        best, coef = scores[k]
    if best < r2 or not chosen:
        return None
    basis = coords[chosen].to_numpy()
    collinear = [
        k
        for k in candidates
        if np.ptp(coords[k].to_numpy()) > 0
        and _r2(basis, coords[k].to_numpy())[0] >= r2
    ]
    coefficients = pd.Series([1.0, *(-coef)], index=[name, *chosen], name="coefficient")
    return ParameterRelation(
        profile=name,
        coefficients=coefficients,
        collinear=collinear,
        r2=best,
        log_parameters=[k for k in [*coefficients.index, *collinear] if k in log],
    )


def related_parameters(
    profiles: ProfileLikelihood,
    *,
    names: Sequence[str] | None = None,
    r2: float = 0.95,
    min_change: float = 1e-3,
) -> RelatedParameters:
    """Groups of functionally related parameters from profile likelihoods.

    Along the profile of a non-identifiable parameter, other parameters
    compensate for it so the fit stays the same. Regressing the profiled
    coordinate on the paths of the others (points within the threshold,
    greedy forward selection of the parameter that explains most, until
    `R² ≥ r2`) gives the combination that stays constant, which is the one
    the data identify. Coordinates are log10 values for log parameters,
    else relative to the optimal value. Similar to MOTA (Hengl et al. 2007).

    Parameters
    ----------
    profiles
        Result of `profile_likelihood`, with all related parameters fitted
    names
        Profiles to analyse, by default the (practically or structurally)
        non-identifiable ones
    r2
        Fraction of the variance of the profiled coordinate a relation has to
        explain
    min_change
        Smallest change (in the coordinates) along a profile for a parameter
        to be a candidate

    Examples
    --------
        >>> profiles = identify.profile_likelihood(model, data, res, loss_fn=nll,
        ...                                        transform="log")
        >>> related = identify.related_parameters(profiles)
        >>> related.groups                 # [["k2", "k3"]]
        >>> print(related.relations[0])    # 1·log10(k2) +1·log10(k3) ≈ const

    """
    if not 0 < r2 <= 1:
        msg = f"r2 has to be in (0, 1], got {r2}"
        raise ValueError(msg)
    if names is None:
        names = [
            k
            for k, p in profiles.profiles.items()
            if p.identifiability
            in ("practically_non_identifiable", "structurally_non_identifiable")
        ]
    elif unknown := sorted(set(names) - set(profiles.profiles)):
        msg = f"No profiles of {unknown}"
        raise ValueError(msg)
    log = profiles.log_parameters
    relations = []
    for name in names:
        prof = profiles[name]
        inside = (prof.delta <= prof.threshold).to_numpy()
        pars = prof.parameters.loc[inside]
        coords = pd.DataFrame(
            {
                k: np.log10(pars[k].to_numpy(dtype=float))
                if k in log
                else pars[k].to_numpy(dtype=float) / (abs(profiles.optimum[k]) or 1.0)
                for k in pars.columns
            }
        )
        if (rel := _relation(coords, name, r2, min_change, log)) is not None:
            relations.append(rel)
    return RelatedParameters(relations=relations)
