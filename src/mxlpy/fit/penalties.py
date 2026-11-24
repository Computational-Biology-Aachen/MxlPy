# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Priors and penalties of point-estimate fits (MAP estimation, regularisation).

The fit functions minimise `loss + Σ penalty`. A `prior` mapping becomes a
`PriorPenalty`, the negative log density of each prior (see
`mxlpy.distributions.prior_penalty`). With a likelihood loss
(`mxlpy.fit.GaussianNll`) this is a maximum-a-posteriori estimate, with a
heuristic loss like `rmse` a regularisation whose strength is set by the
`weight` of the penalty.

`L1Penalty` and `GroupL1Penalty` are (smoothed) lasso penalties, which
pull values (or linear combinations of them) to zero, e.g. to find which
parameters have to differ between conditions (`mxlpy.select.condition_specific`).

Custom penalties implement the `Penalty` protocol, e.g. penalties coupling
several parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from mxlpy.distributions import prior_penalty

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy.distributions import Prior
    from mxlpy.minimizers.abstract import Residual
    from mxlpy.types import Array

__all__ = [
    "GroupL1Penalty",
    "L1Penalty",
    "Penalty",
    "PriorPenalty",
]

_STEP = 1e-6


@runtime_checkable
class Penalty(Protocol):
    """A penalty on fitted parameters, added to the loss of a fit.

    `names` are the parameters it depends on, which all have to be fitted,
    and `__call__` gets their values in this order.

    Optionally, `residuals(values)` returns a pseudo-residual vector `r` with
    `Σr² / 2 == penalty(values)`, so least-squares minimizers can use the
    penalty, and `jacobian(values)` its derivative (`len(r)` x `len(names)`,
    else by finite differences). Without residuals, gradients come from
    finite differences of the penalty and Newton-type methods fall back to
    quasi-Newton updates.
    """

    @property
    def names(self) -> Sequence[str]:
        """Names of the parameters the penalty depends on."""
        ...

    def __call__(self, values: Array) -> float:
        """Penalty at the values of `names`."""
        ...


@dataclass(frozen=True)
class PriorPenalty:
    """Negative log density of independent priors per parameter.

    The penalty of `prior={...}` in the fit functions, see
    `mxlpy.distributions.prior_penalty` for how each prior is turned into a
    penalty. Use it directly to weight the priors against a heuristic loss.

    Parameters
    ----------
    prior
        Prior per parameter, e.g. from `mxlpy.distributions`
    weight
        Factor of the penalty. 1 for a maximum-a-posteriori estimate with a
        likelihood loss (`mxlpy.fit.GaussianNll`)

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data, minimizer=minimizer,
        ...                 penalty=fit.PriorPenalty({"k1": Normal(1, 0.1)}, weight=0.1))

    """

    prior: Mapping[str, Prior]
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Check the weight."""
        _check_weight(self.weight)

    @property
    def names(self) -> list[str]:
        """Names of the parameters with a prior."""
        return list(self.prior)

    @property
    def has_residuals(self) -> bool:
        """Whether all priors provide a `penalty_residual`."""
        return all(
            callable(getattr(p, "penalty_residual", None)) for p in self.prior.values()
        )

    def __call__(self, values: Array) -> float:
        """Weighted sum of the penalties of all priors."""
        return self.weight * sum(
            prior_penalty(p, float(x))
            for p, x in zip(self.prior.values(), values, strict=True)
        )

    def residuals(self, values: Array) -> Array:
        """Signed pseudo-residuals of the priors, scaled by `sqrt(weight)`."""
        if not self.has_residuals:
            missing = sorted(
                k
                for k, p in self.prior.items()
                if not callable(getattr(p, "penalty_residual", None))
            )
            msg = f"Least-squares minimizers need priors with a penalty_residual, not the ones of {missing}"
            raise TypeError(msg)
        return math.sqrt(self.weight) * np.array(
            [
                cast(Any, p).penalty_residual(float(x))
                for p, x in zip(self.prior.values(), values, strict=True)
            ],
            dtype=float,
        )

    def jacobian(self, values: Array) -> Array:
        """Diagonal derivative of `residuals`, by central differences."""
        out = np.zeros((len(values), len(values)))
        for i, (p, x) in enumerate(zip(self.prior.values(), values, strict=True)):
            fn = cast(Any, p).penalty_residual
            out[i, i] = math.sqrt(self.weight) * _derivative(fn, float(x))
        return out


def _check_weight(weight: float) -> None:
    if not (math.isfinite(weight) and weight >= 0):
        msg = f"The weight of a penalty has to be non-negative, got {weight}"
        raise ValueError(msg)


def _check_smoothing(smoothing: float) -> None:
    if not (math.isfinite(smoothing) and smoothing > 0):
        msg = f"The smoothing of an L1 penalty has to be positive, got {smoothing}"
        raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class L1Penalty:
    """Smoothed lasso (L1) penalty, optionally an elastic net.

    `weight · Σⱼ sⱼ (a |tⱼ|_ε + (1 - a) tⱼ² / 2)` of the terms `t = matrix ·
    values` (the values themselves without a matrix), with the smooth
    absolute value `|t|_ε = sqrt(t² + ε²) - ε`. `|t|_ε` is quadratic within
    about `ε` of zero and `|t| - ε` beyond, so minimizers needing smooth
    objectives work, while terms whose optimum is zero for the exact L1
    penalty end up within about `ε` of zero.

    Parameters
    ----------
    names
        Names of the penalised parameters
    weight
        Strength of the penalty (λ)
    scales
        Factor `sⱼ` per term, e.g. the weights of an adaptive lasso. All 1 by
        default
    matrix
        Terms as linear combinations of the values, `(n_terms, len(names))`,
        e.g. rows `[1, -1]` to penalise differences (fused lasso). The
        identity by default
    l1_ratio
        `a`, 1 for the lasso, between 0 and 1 for an elastic net
    smoothing
        `ε` of the smooth absolute value, in units of the terms

    Examples
    --------
        >>> fit.time_course(model, p0=p0, data=data, minimizer=minimizer,
        ...                 penalty=fit.L1Penalty(["k1", "k2"], weight=0.5))

    """

    names: Sequence[str]
    weight: float = 1.0
    scales: Sequence[float] | Array | None = None
    matrix: Array | None = None
    l1_ratio: float = 1.0
    smoothing: float = 1e-3

    def __post_init__(self) -> None:
        """Check the settings."""
        _check_weight(self.weight)
        _check_smoothing(self.smoothing)
        if not 0 <= self.l1_ratio <= 1:
            msg = f"l1_ratio has to be within [0, 1], got {self.l1_ratio}"
            raise ValueError(msg)
        if self.matrix is not None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != len(self.names):  # noqa: PLR2004
                msg = f"The matrix needs {len(self.names)} columns, got shape {matrix.shape}"
                raise ValueError(msg)
            object.__setattr__(self, "matrix", matrix)
        n_terms = len(self.names) if self.matrix is None else len(self.matrix)
        if self.scales is not None:
            scales = np.asarray(self.scales, dtype=float)
            if scales.shape != (n_terms,) or not np.all(scales >= 0):
                msg = f"scales need one non-negative value per term ({n_terms})"
                raise ValueError(msg)
            object.__setattr__(self, "scales", scales)

    def terms(self, values: Array) -> Array:
        """The penalised terms `matrix · values`."""
        values = np.asarray(values, dtype=float)
        return values if self.matrix is None else self.matrix @ values

    def _factor(self, t: Array) -> tuple[Array, Array, Array]:
        """`c = weight · s`, `q = sqrt(t² + ε²)` and `φ` with `r = t φ`."""
        eps = self.smoothing
        c = self.weight * (
            np.ones_like(t) if self.scales is None else np.asarray(self.scales)
        )
        q = np.sqrt(t * t + eps * eps)
        a = self.l1_ratio
        return c, q, np.sqrt(c * (2 * a / (q + eps) + (1 - a)))

    def __call__(self, values: Array) -> float:
        """Penalty at the values."""
        t = self.terms(values)
        c, q, _ = self._factor(t)
        a = self.l1_ratio
        return float(np.sum(c * (a * (q - self.smoothing) + (1 - a) * t * t / 2)))

    def residuals(self, values: Array) -> Array:
        """Signed pseudo-residuals `tⱼ φⱼ`, smooth through zero."""
        t = self.terms(values)
        return t * self._factor(t)[2]

    def jacobian(self, values: Array) -> Array:
        """Derivative of `residuals`, `(n_terms, len(names))`."""
        t = self.terms(values)
        c, q, phi = self._factor(t)
        eps = self.smoothing
        with np.errstate(divide="ignore", invalid="ignore"):
            d_phi = np.where(
                phi > 0,
                -c * self.l1_ratio * t / (q * (q + eps) ** 2 * phi),
                0.0,
            )
        d_t = phi + t * d_phi
        if self.matrix is None:
            return np.diag(d_t)
        return d_t[:, None] * self.matrix


@dataclass(frozen=True, eq=False)
class GroupL1Penalty:
    """Smoothed group lasso penalty.

    `weight · Σ_g s_g |x_g|_ε` with the smooth Euclidean norm `|x|_ε =
    sqrt(Σ x² + ε²) - ε` of the values `x_g` of each group, so groups go to
    zero together (all values within about `ε` of zero).

    Parameters
    ----------
    groups
        Names of the parameters of each group, disjoint
    weight
        Strength of the penalty (λ)
    scales
        Factor `s_g` per group, `sqrt(group size)` by default so groups of
        different size are comparable
    smoothing
        `ε` of the smooth norm

    Examples
    --------
        >>> fit.GroupL1Penalty({"k1": ["k1_a", "k1_b"], "k2": ["k2_a", "k2_b"]}, 0.5)

    """

    groups: Mapping[str, Sequence[str]]
    weight: float = 1.0
    scales: Mapping[str, float] | None = None
    smoothing: float = 1e-3

    def __post_init__(self) -> None:
        """Check the settings."""
        _check_weight(self.weight)
        _check_smoothing(self.smoothing)
        names = self.names
        if len(set(names)) != len(names):
            msg = "The groups of a GroupL1Penalty have to be disjoint"
            raise ValueError(msg)
        if any(len(g) == 0 for g in self.groups.values()):
            msg = "Groups can't be empty"
            raise ValueError(msg)
        if self.scales is not None and (
            set(self.scales) != set(self.groups)
            or any(not v >= 0 for v in self.scales.values())
        ):
            msg = "scales need one non-negative value per group"
            raise ValueError(msg)

    @property
    def names(self) -> list[str]:
        """Names of all parameters, group after group."""
        return [k for g in self.groups.values() for k in g]

    def _slices(self) -> list[tuple[slice, float]]:
        out = []
        start = 0
        for name, group in self.groups.items():
            scale = math.sqrt(len(group)) if self.scales is None else self.scales[name]
            out.append((slice(start, start + len(group)), self.weight * scale))
            start += len(group)
        return out

    def norms(self, values: Array) -> Array:
        """Euclidean norm of the values of each group."""
        values = np.asarray(values, dtype=float)
        return np.array([np.linalg.norm(values[s]) for s, _ in self._slices()])

    def __call__(self, values: Array) -> float:
        """Penalty at the values."""
        eps = self.smoothing
        return float(
            sum(
                c * (math.sqrt(n * n + eps * eps) - eps)
                for (_, c), n in zip(self._slices(), self.norms(values), strict=True)
            )
        )

    def residuals(self, values: Array) -> Array:
        """Pseudo-residuals `x_g φ_g`, smooth through zero."""
        values = np.asarray(values, dtype=float)
        eps = self.smoothing
        out = np.empty_like(values)
        for s, c in self._slices():
            q = math.sqrt(float(values[s] @ values[s]) + eps * eps)
            out[s] = values[s] * math.sqrt(2 * c / (q + eps))
        return out

    def jacobian(self, values: Array) -> Array:
        """Derivative of `residuals`, block diagonal over the groups."""
        values = np.asarray(values, dtype=float)
        eps = self.smoothing
        out = np.zeros((len(values), len(values)))
        for s, c in self._slices():
            x = values[s]
            q = math.sqrt(float(x @ x) + eps * eps)
            phi = math.sqrt(2 * c / (q + eps))
            out[s, s] = phi * np.eye(len(x)) - phi / (2 * q * (q + eps)) * np.outer(
                x, x
            )
        return out


def _step(x: float) -> float:
    """Relative finite-difference step, so tiny parameters stay accurate."""
    return _STEP * abs(x) if x != 0 else _STEP


def _derivative(fn: Any, x: float) -> float:
    """Central difference of a scalar function, one-sided at its domain's edge.

    Zero where neither is finite, e.g. outside the support of a uniform prior,
    whose penalty is flat there for the minimizer.
    """
    h = _step(x)
    up, down = fn(x + h), fn(x - h)
    if math.isfinite(d := (up - down) / (2 * h)):
        return d
    base = fn(x)
    for d in ((up - base) / h, (base - down) / h):
        if math.isfinite(d):
            return d
    return 0.0


def _residual_capable(penalty: Penalty) -> bool:
    return callable(getattr(penalty, "residuals", None)) and getattr(
        penalty, "has_residuals", True
    )


def _penalty_jacobian(penalty: Penalty, values: Array) -> Array:
    """Pseudo-residual Jacobian of a penalty, analytic or by differences."""
    if callable(jac := getattr(penalty, "jacobian", None)):
        return np.asarray(jac(values), dtype=float)
    residuals = cast(Any, penalty).residuals
    columns = []
    for i, x in enumerate(values):
        h = _step(float(x))
        up, down = values.copy(), values.copy()
        up[i] += h
        down[i] -= h
        columns.append((residuals(up) - residuals(down)) / (2 * h))
    return np.column_stack(columns)


class _FixedPenalty:
    """A penalty with some of its parameters held at fixed values.

    Lets a fit with fixed parameters (e.g. a profile likelihood) keep the
    penalties that depend on them.
    """

    def __init__(self, inner: Penalty, fixed: Mapping[str, float]) -> None:
        self.inner = inner
        self._full = np.array([fixed.get(k, np.nan) for k in inner.names])
        self._free = [i for i, k in enumerate(inner.names) if k not in fixed]

    @property
    def names(self) -> list[str]:
        return [self.inner.names[i] for i in self._free]

    @property
    def has_residuals(self) -> bool:
        return _residual_capable(self.inner)

    def _values(self, values: Array) -> Array:
        full = self._full.copy()
        full[self._free] = values
        return full

    def __call__(self, values: Array) -> float:
        return self.inner(self._values(values))

    def residuals(self, values: Array) -> Array:
        return np.asarray(
            cast(Any, self.inner).residuals(self._values(values)), dtype=float
        )

    def jacobian(self, values: Array) -> Array:
        return _penalty_jacobian(self.inner, self._values(values))[:, self._free]


def fix_penalties(
    penalties: Sequence[Penalty],
    fixed: Mapping[str, float],
) -> tuple[list[Penalty], float]:
    """Penalties of a fit with `fixed` parameters left out of it.

    Returns the penalties of the remaining parameters (with the fixed values
    inserted) and the constant sum of the penalties that only depend on
    fixed parameters.
    """
    free: list[Penalty] = []
    constant = 0.0
    for p in penalties:
        if not set(p.names) & set(fixed):
            free.append(p)
        elif set(p.names) <= set(fixed):
            constant += float(p(np.array([fixed[k] for k in p.names], dtype=float)))
        else:
            free.append(_FixedPenalty(p, fixed))
    return free, constant


def collect_penalties(
    prior: Mapping[str, Prior] | None,
    penalty: Penalty | Sequence[Penalty] | None,
    names: Sequence[str],
) -> list[Penalty]:
    """Penalties of a fit, checking they only depend on fitted parameters."""
    out: list[Penalty] = []
    if prior:
        out.append(PriorPenalty(prior))
    if penalty is not None:
        out.extend([penalty] if isinstance(penalty, Penalty) else penalty)
    fitted = set(names)
    for p in out:
        if unknown := sorted(set(p.names) - fitted):
            msg = f"Priors and penalties need fitted parameters, these aren't in p0: {unknown}"
            raise ValueError(msg)
    return out


###############################################################################
# Residual wrappers
###############################################################################


class _Penalised:
    """Residual of a fit plus penalties on its parameters."""

    def __init__(self, inner: Residual, penalties: list[Penalty]) -> None:
        self.inner = inner
        self.penalties = penalties

    def penalty(self, updates: dict[str, float]) -> float:
        """Sum of all penalties at `updates`."""
        return float(
            sum(
                p(np.array([updates[k] for k in p.names], dtype=float))
                for p in self.penalties
            )
        )

    def __call__(self, updates: dict[str, float]) -> float:
        value = self.inner(updates)
        if not np.isfinite(value):
            return value
        return value + self.penalty(updates)


class _PenalisedGradient(_Penalised):
    """`_Penalised` adding the penalties' derivatives to the residual's."""

    def _values(self, penalty: Penalty, updates: dict[str, float]) -> Array:
        return np.array([updates[k] for k in penalty.names], dtype=float)

    def _columns(self, penalty: Penalty, updates: dict[str, float]) -> list[int]:
        index = {k: i for i, k in enumerate(updates)}
        return [index[k] for k in penalty.names]

    def _residual_scale(self) -> float:
        """Factor of the penalties' pseudo-residuals in least squares.

        Least squares minimizes `Σr² / 2` of the residual vector. For the
        penalties to have the weight they have in the loss, the loss has to be
        `c Σr² / 2` of the fit's residuals (`GaussianNll`: c = 1,
        `mean_squared`: c = 2 / n), and the pseudo-residuals are divided by
        `sqrt(c)`.
        """
        fn = getattr(self.inner, "squares_factor", None)
        factor = cast("float | None", fn()) if callable(fn) else None
        if factor is None:
            msg = (
                "LeastSquaresMinimizer minimizes a sum of squares, so with priors or "
                "penalties the loss has to be one too (GaussianNll or mean_squared); "
                "use LocalScipyMinimizer for other losses"
            )
            raise TypeError(msg)
        return 1 / math.sqrt(factor)

    def _jacobian(self, penalty: Penalty, values: Array) -> Array:
        return _penalty_jacobian(penalty, values)

    def residuals(self, updates: dict[str, float]) -> Array:
        """Residual vector of the fit followed by the penalties' pseudo-residuals."""
        scale = self._residual_scale()
        parts = [np.asarray(cast(Any, self.inner).residuals(updates), dtype=float)]
        for p in self.penalties:
            if not _residual_capable(p):
                msg = f"Least-squares minimizers need penalties with residuals, {type(p).__name__} has none"
                raise TypeError(msg)
            r = cast(Any, p).residuals(self._values(p, updates))
            parts.append(scale * np.asarray(r, dtype=float))
        return np.concatenate(parts)

    def jacobian(self, updates: dict[str, float]) -> Array:
        """Derivative of `residuals`, (n_residuals, n_parameters)."""
        scale = self._residual_scale()
        rows = [np.asarray(cast(Any, self.inner).jacobian(updates), dtype=float)]
        for p in self.penalties:
            jac = self._jacobian(p, self._values(p, updates))
            block = np.zeros((jac.shape[0], len(updates)))
            block[:, self._columns(p, updates)] = scale * jac
            rows.append(block)
        return np.vstack(rows)

    def gradient(self, updates: dict[str, float]) -> Array:
        """Gradient of the loss plus the penalties."""
        grad = np.array(cast(Any, self.inner).gradient(updates), dtype=float)
        for p in self.penalties:
            values = self._values(p, updates)
            columns = self._columns(p, updates)
            if _residual_capable(p):
                r = np.asarray(cast(Any, p).residuals(values), dtype=float)
                g = r @ self._jacobian(p, values)
            else:
                g = np.array(
                    [
                        _derivative(_along(p, values, i), float(x))
                        for i, x in enumerate(values)
                    ]
                )
            grad[columns] += g
        return grad

    def hessian(self, updates: dict[str, float]) -> Array | None:
        """Gauss-Newton Hessian of the loss plus the penalties.

        None if the loss has none or a penalty has no residuals.
        """
        if not all(_residual_capable(p) for p in self.penalties):
            return None
        hess = cast(Any, self.inner).hessian(updates)
        if hess is None:
            return None
        hess = np.array(hess, dtype=float)
        for p in self.penalties:
            columns = self._columns(p, updates)
            jac = self._jacobian(p, self._values(p, updates))
            hess[np.ix_(columns, columns)] += jac.T @ jac
        return hess


def _along(penalty: Penalty, values: Array, i: int) -> Any:
    """The penalty as a function of its `i`-th value only."""

    def fn(x: float) -> float:
        v = values.copy()
        v[i] = x
        return penalty(v)

    return fn


def penalised(residual: Residual, penalties: list[Penalty]) -> _Penalised:
    """Add penalties to a residual, passing derivatives through if it has them."""
    cls = _PenalisedGradient if hasattr(residual, "jacobian") else _Penalised
    return cls(residual, penalties)
