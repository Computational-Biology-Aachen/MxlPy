# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Model selection: multistart fits of candidate models and structure search."""

from __future__ import annotations

import math
import warnings
import zlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from wadler_lindig import pformat

from mxlpy import fit
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit._single import _evaluate_loss, _fit_data
from mxlpy.fit.losses import _is_likelihood
from mxlpy.fit.penalties import Penalty, fix_penalties
from mxlpy.parallel import parallelise
from mxlpy.runners import default_runner, precompile
from mxlpy.select._criteria import (
    Criterion,
    FitCriteria,
    InformationCriterion,
    LikelihoodRatio,
    lrt,
    model_weights,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mxlpy._kinetic_builder import KineticModelBuilder
    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy.distributions import Distribution, Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol
    from mxlpy.select._space import ModelSpace, Move

__all__ = [
    "CandidateFit",
    "Method",
    "Selection",
    "compare",
    "run",
]

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

type Method = Literal["forward", "backward", "adaptive", "brute_force"]
"""Structure search, see `run`."""

# Positive start of log-transformed values without a positive value or bound
_LOG_FLOOR = 1e-6

# Starts whose objective is within this of the best count as converged
# (half of Δχ² = 1 for a negative log-likelihood)
_CONVERGED = 0.5


###############################################################################
# Results
###############################################################################


@dataclass(kw_only=True)
class CandidateFit[M: _Model](FitCriteria):
    """Best multistart fit of one candidate model.

    Attributes
    ----------
    id
        Candidate id: its switches that are on joined by `+` (or `base`), or
        the name given to `compare`
    on
        Switches that are on (empty for `compare`)
    model
        The candidate model with the fitted parameters (and initial values)
        set, or unchanged if all fits failed
    best_pars
        Estimated parameters (and initial values) at the best fit, like
        `fit.Fit.best_pars`
    fixed
        Values the candidate's switches fix instead of estimating them
    n_data
        Number of data points (finite values of the data)
    loss
        Negative log-likelihood of the data at the best fit (up to a constant
        that is the same for all candidates), infinite if all fits failed
    penalty
        Priors and penalties at the best fit, which the fits minimised along
        with the loss but which don't enter the criteria
    objectives
        Objective (loss + penalty) of every start, the warm start first,
        infinite for failed starts
    parent
        Id of the candidate whose optimum warm-started this one

    """

    id: str
    on: frozenset[str]
    model: M
    best_pars: dict[str, float]
    fixed: dict[str, float]
    n_data: int
    loss: float
    penalty: float
    objectives: tuple[float, ...]
    parent: str | None = None

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    @property
    def n_failed(self) -> int:
        """Number of starts whose fit failed."""
        return sum(not math.isfinite(o) for o in self.objectives)

    def converged(self, tol: float = _CONVERGED) -> float:
        """Fraction of starts whose objective is within `tol` of the best.

        A low fraction means the optimum is hard to find and more starts may
        find a better one. The default `tol` is half of Δχ² = 1.
        """
        best = self.objective
        if not math.isfinite(best) or not self.objectives:
            return 0.0
        return sum(o - best <= tol for o in self.objectives) / len(self.objectives)


@dataclass(kw_only=True)
class Selection[M: _Model]:
    """Result of a model selection, see `run` and `compare`.

    Attributes
    ----------
    candidates
        Fit of every evaluated candidate by id, in the order they were fitted
    criterion
        Criterion the models were compared with
    best_id
        Id of the selected model: the end of the search path, or the
        candidate with the best criterion for `brute_force` and `compare`.
        The table ranks all evaluated candidates (including reused ones), so
        with `min_improvement` or `reuse` its top may differ
    path
        Ids of the models the search moved through, from the start to the
        selected one
    complete
        False if the search stopped at `max_candidates` before converging
    switches
        Switch names of the model space (empty for `compare`)
    significance
        Significance level of the likelihood-ratio tests

    """

    candidates: dict[str, CandidateFit[M]]
    criterion: Criterion
    best_id: str
    path: list[str]
    complete: bool = True
    switches: list[str] = field(default_factory=list)
    significance: float = 0.05

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def __getitem__(self, id: str) -> CandidateFit[M]:  # noqa: A002
        """Fit of a candidate."""
        return self.candidates[id]

    @property
    def best(self) -> CandidateFit[M]:
        """Fit of the selected model."""
        return self.candidates[self.best_id]

    @property
    def best_model(self) -> M:
        """The selected model, with the fitted parameters set."""
        return self.best.model

    @property
    def table(self) -> pd.DataFrame:
        """Ranked table of all candidates.

        Columns: number of estimated parameters `k`, data points `n`,
        log-likelihood, AIC, AICc, BIC, `delta` and `weight` of the selection
        criterion (Akaike / BIC weights among the evaluated candidates), the
        fraction of `converged` starts, failed starts, the `parent` the fit
        was warm-started from and one column per switch. Sorted by the
        criterion; for likelihood-ratio tests (without delta and weight) by
        `k` and log-likelihood.
        """
        fits = list(self.candidates.values())
        df = pd.DataFrame(
            {
                "k": [c.n_parameters for c in fits],
                "n": [c.n_data for c in fits],
                "log_likelihood": [c.log_likelihood for c in fits],
                "aic": [c.aic for c in fits],
                "aicc": [c.aicc for c in fits],
                "bic": [c.bic for c in fits],
            },
            index=pd.Index([c.id for c in fits], name="model"),
        )
        if self.criterion != "lrt":
            values = cast("pd.Series", df[self.criterion])
            finite = values[np.isfinite(values)]
            df["delta"] = values - (finite.min() if len(finite) else np.nan)
            df["weight"] = model_weights(values)
        df["converged"] = [c.converged() for c in fits]
        df["n_failed"] = [c.n_failed for c in fits]
        df["parent"] = [c.parent for c in fits]
        for name in self.switches:
            df[name] = [name in c.on for c in fits]
        if self.criterion == "lrt":
            return df.sort_values(["k", "log_likelihood"], ascending=[True, False])
        return df.sort_values(self.criterion, kind="stable")

    def lrt(self, null: str, alt: str) -> LikelihoodRatio:
        """Likelihood-ratio test of candidate `null` nested in candidate `alt`.

        The test assumes `null` is a special case of `alt`, e.g. with the
        extra parameters of `alt` fixed (`free_parameter`).
        """
        a, b = self.candidates[null], self.candidates[alt]
        return lrt(a.log_likelihood, b.log_likelihood, b.n_parameters - a.n_parameters)


###############################################################################
# Fitting candidates
###############################################################################


@dataclass(kw_only=True)
class _Problem:
    """Everything a worker needs to fit one candidate from one start."""

    model: _Model
    data: pd.Series | pd.DataFrame
    loss_fn: LossFn
    minimizer: MinimizerProtocol
    y0: dict[str, float] | None
    protocol: pd.DataFrame | None
    runner: Any
    bounds: Bounds
    prior: dict[str, Prior] | None
    penalty: Penalty | Sequence[Penalty] | None
    transform: Transform | None
    postprocess_fn: PostprocessFn | None


type _Outcome = tuple[float, float, float, dict[str, float]]


def _fit_start(task: tuple[_Problem, dict[str, float]]) -> _Outcome | None:
    """Objective, loss, penalty and parameters of one fit, None if it failed."""
    problem, p0 = task
    if not p0:
        # Nothing to estimate, only evaluate the loss
        loss = _evaluate_loss(
            deepcopy(problem.model),
            problem.data,
            problem.protocol,
            y0=problem.y0,
            runner=problem.runner,
            loss_fn=problem.loss_fn,
            postprocess_fn=problem.postprocess_fn,
            standard_scale=False,
        )
        return (loss, loss, 0.0, {}) if math.isfinite(loss) else None
    res = _fit_data(
        problem.model,
        problem.data,
        problem.protocol,
        p0=p0,
        y0=problem.y0,
        minimizer=problem.minimizer,
        runner=problem.runner,
        loss_fn=problem.loss_fn,
        postprocess_fn=problem.postprocess_fn,
        bounds=problem.bounds,
        prior=problem.prior,
        penalty=problem.penalty,
        transform=problem.transform,
    )
    match res.value:
        case fit.Fit() as best if math.isfinite(best.objective):
            return best.objective, best.loss, best.penalty, best.best_pars
        case _:
            return None


def _names(model: _Model) -> tuple[set[str], set[str]]:
    """Parameter and variable names of a model."""
    if isinstance(model, SteadyStateModelBuilder):
        return set(model.get_parameter_names()), set()
    return set(model.get_parameter_names()), set(model.get_variable_names())


@dataclass(kw_only=True)
class _Spec:
    """A candidate to fit."""

    id: str
    on: frozenset[str]
    model: _Model
    fixed: dict[str, float]
    parent: CandidateFit[Any] | None


@dataclass(kw_only=True)
class _Fitter:
    """Fits candidates: shared settings of a selection."""

    data: pd.Series | pd.DataFrame
    p0: Mapping[str, float | Distribution]
    loss_fn: LossFn
    minimizer: MinimizerProtocol
    n_starts: int
    y0: dict[str, float] | None
    protocol: pd.DataFrame | None
    runner: Any
    bounds: Bounds
    prior: Mapping[str, Prior] | None
    penalty: Sequence[Penalty]
    transform: Transform | None
    postprocess_fn: PostprocessFn | None
    seed: int
    parallel: bool
    max_workers: int | None
    timeout: float | None
    n_data: int = field(init=False)

    def __post_init__(self) -> None:
        """Count the data points and check the names of the settings."""
        self.n_data = int(np.isfinite(self.data.to_numpy(dtype=float)).sum())
        named = {"bounds": self.bounds, "prior": self.prior or {}}
        if isinstance(self.transform, dict):
            named["transform"] = self.transform
        for arg, mapping in named.items():
            if unknown := sorted(set(mapping) - set(self.p0)):
                msg = (
                    f"{arg} of parameters that aren't estimated (not in p0): {unknown}"
                )
                raise ValueError(msg)

    def check_names(self, models: Iterable[_Model]) -> None:
        """Raise for estimated names that none of `models` has, e.g. typos."""
        known: set[str] = set()
        for model in models:
            known.update(_names(model)[0] | _names(model)[1])
        if unknown := sorted(set(self.p0) - known):
            msg = f"p0 has names that are neither parameters nor variables of the models: {unknown}"
            raise ValueError(msg)

    def _log(self, free: list[str]) -> set[str]:
        if self.transform is None:
            return set()
        if isinstance(self.transform, str):
            return set(free) if self.transform == "log" else set()
        return {k for k in free if self.transform.get(k) == "log"}

    def _start(self, name: str, value: float | None, *, log: bool) -> float:
        """Start value clipped to the bounds, positive for log transforms."""
        lower, upper = self.bounds.get(name, (None, None))
        if log:
            floor = lower if lower is not None and lower > 0 else _LOG_FLOOR
            value = floor if value is None or value <= 0 else value
            lower = max(lower or 0.0, floor)
        elif value is None:
            value = 0.0 if lower is None else lower
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        return float(value)

    def _starts(
        self,
        spec: _Spec,
        free: list[str],
        nominal: dict[str, float],
    ) -> list[dict[str, float]]:
        """Warm start from the parent's optimum, then random starts.

        Starts of log-transformed values are kept positive.
        """
        log = self._log(free)
        parent = spec.parent
        warm = {}
        for k in free:
            candidates = [nominal[k]]
            if (
                parent is not None
                and (value := parent.best_pars.get(k, parent.fixed.get(k))) is not None
            ):
                candidates.insert(0, value)
            value = next(
                (
                    v
                    for v in candidates
                    if math.isfinite(v) and not (k in log and v <= 0)
                ),
                None,
            )
            warm[k] = self._start(k, value, log=k in log)
        if self.n_starts == 0 or not free:
            return [warm]
        rng = np.random.default_rng([self.seed, zlib.crc32(spec.id.encode())])
        columns = {}
        for k in free:
            dist = self.p0[k]
            columns[k] = (
                np.full(self.n_starts, float(dist))
                if isinstance(dist, int | float)
                else np.asarray(dist.sample(self.n_starts, rng), dtype=float)
            )
        sampled = [
            {k: self._start(k, float(columns[k][i]), log=k in log) for k in free}
            for i in range(self.n_starts)
        ]
        return [warm, *sampled]

    def _problem(self, spec: _Spec, free: list[str], v_names: set[str]) -> _Problem:
        model = spec.model
        y0 = self.y0
        if fixed_v := {k: v for k, v in spec.fixed.items() if k in v_names}:
            y0 = {**(y0 or {}), **fixed_v}
        transform = self.transform
        if isinstance(transform, dict):
            transform = cast(
                "Transform", {k: v for k, v in transform.items() if k in free}
            )
        runner = default_runner(model) if self.runner is None else self.runner
        precompile(runner, model)
        return _Problem(
            model=model,
            data=self.data,
            loss_fn=self.loss_fn,
            minimizer=self.minimizer,
            y0=y0,
            protocol=self.protocol,
            runner=runner,
            bounds={k: v for k, v in self.bounds.items() if k in free},
            prior=(
                {k: v for k, v in self.prior.items() if k in free}
                if self.prior
                else None
            ),
            # Penalties of fixed parameters are constants, which don't matter
            penalty=fix_penalties(self.penalty, spec.fixed)[0],
            transform=transform,
            postprocess_fn=self.postprocess_fn,
        )

    def fit(self, specs: list[_Spec]) -> list[CandidateFit[Any]]:
        """Fit all starts of all candidates in parallel."""
        if not specs:
            return []
        prepared = []
        tasks: list[tuple[tuple[int, int], tuple[_Problem, dict[str, float]]]] = []
        for i, spec in enumerate(specs):
            p_names, v_names = _names(spec.model)
            if unknown := sorted(set(spec.fixed) - p_names - v_names):
                msg = f"Candidate {spec.id!r} fixes unknown parameters {unknown}"
                raise ValueError(msg)
            free = [
                k
                for k in self.p0
                if (k in p_names or k in v_names) and k not in spec.fixed
            ]
            nominal = dict(spec.model.get_parameter_values())
            if v_names:
                model = cast("KineticModelBuilder | OdeModelBuilder", spec.model)
                nominal |= model.get_initial_conditions() | (self.y0 or {})
            starts = self._starts(spec, free, nominal)
            problem = self._problem(spec, free, v_names)
            prepared.append((spec, free, starts, v_names))
            tasks.extend(((i, j), (problem, p0)) for j, p0 in enumerate(starts))

        outcomes = dict(
            parallelise(
                _fit_start,
                inputs=tasks,
                parallel=self.parallel and (len(tasks) > 1 or self.timeout is not None),
                max_workers=self.max_workers,
                timeout=self.timeout,
                tqdm_desc="fits",
            )
        )

        out = []
        for i, (spec, free, starts, v_names) in enumerate(prepared):
            results = [outcomes.get((i, j)) for j in range(len(starts))]
            objectives = tuple(math.inf if r is None else r[0] for r in results)
            model: Any = deepcopy(spec.model)
            for k, v in spec.fixed.items():
                if k in v_names:
                    model.update_variable(k, v)
            done = [r for r in results if r is not None]
            if done:
                _, loss, penalty, best = min(done, key=lambda r: r[0])
                for k, v in best.items():
                    if k in v_names:
                        model.update_variable(k, v)
                    else:
                        model.update_parameter(k, v)
            else:
                loss, penalty, best = math.inf, 0.0, dict(starts[0])
            out.append(
                CandidateFit(
                    id=spec.id,
                    on=spec.on,
                    model=model,
                    best_pars={k: float(best[k]) for k in free},
                    fixed=dict(spec.fixed),
                    n_data=self.n_data,
                    loss=float(loss),
                    penalty=float(penalty),
                    objectives=objectives,
                    parent=None if spec.parent is None else spec.parent.id,
                )
            )
        return out


###############################################################################
# Search
###############################################################################


def _penalties(penalty: Penalty | Sequence[Penalty] | None) -> list[Penalty]:
    if penalty is None:
        return []
    return [penalty] if isinstance(penalty, Penalty) else list(penalty)


def _check_args(
    loss_fn: LossFn,
    n_starts: int,
    model: _Model,
    y0: dict[str, float] | None,
) -> None:
    if not _is_likelihood(loss_fn):
        msg = "Model selection needs a likelihood loss like fit.GaussianNll(sigma)"
        raise ValueError(msg)
    if n_starts < 0:
        msg = f"n_starts can't be negative, got {n_starts}"
        raise ValueError(msg)
    if isinstance(model, SteadyStateModelBuilder) and y0 is not None:
        msg = "A SteadyStateModelBuilder has no variables, so y0 doesn't apply"
        raise TypeError(msg)


@dataclass(kw_only=True)
class _Search[M: _Model]:
    space: ModelSpace[M]
    fitter: _Fitter
    criterion: Criterion
    significance: float
    min_improvement: float
    max_candidates: int | None
    fits: dict[str, CandidateFit[M]]
    n_fitted: int = 0

    def evaluate(
        self,
        candidates: Iterable[frozenset[str]],
        parent: CandidateFit[M] | None,
    ) -> list[CandidateFit[M]] | None:
        """Fits of candidates, fitting those not fitted yet.

        None if fitting them would exceed `max_candidates`.
        """
        ids = {self.space.id(c): c for c in candidates}
        new = [k for k in ids if k not in self.fits]
        if (
            self.max_candidates is not None
            and self.n_fitted + len(new) > self.max_candidates
        ):
            return None
        specs = []
        for k in new:
            model, fixed = self.space.build(ids[k])
            specs.append(
                _Spec(id=k, on=ids[k], model=model, fixed=fixed, parent=parent)
            )
        for res in self.fitter.fit(specs):
            self.fits[res.id] = res
        self.n_fitted += len(new)
        return [self.fits[k] for k in ids]

    def better(
        self, current: CandidateFit[M], move: Move, fits: list[CandidateFit[M]]
    ) -> CandidateFit[M] | None:
        """The neighbour to move to, None if none improves on `current`."""
        if self.criterion != "lrt":
            name = cast("InformationCriterion", self.criterion)
            best = min(fits, key=lambda c: c.criterion(name), default=None)
            if (
                best is not None
                and best.criterion(name)
                < current.criterion(name) - self.min_improvement
            ):
                return best
            return None
        if move == "forward":
            tests = [
                (lrt(current.log_likelihood, c.log_likelihood, df).p_value, c)
                for c in fits
                if (df := c.n_parameters - current.n_parameters) >= 1
            ]
            significant = [t for t in tests if t[0] < self.significance]
            return (
                min(significant, key=lambda t: (t[0], -t[1].log_likelihood))[1]
                if significant
                else None
            )
        tests = [
            (lrt(c.log_likelihood, current.log_likelihood, df).p_value, c)
            for c in fits
            if (df := current.n_parameters - c.n_parameters) >= 1
        ]
        removable = [t for t in tests if t[0] >= self.significance]
        return (
            max(removable, key=lambda t: (t[0], t[1].log_likelihood))[1]
            if removable
            else None
        )

    def run(self, start: frozenset[str], method: Method) -> tuple[list[str], bool]:
        """Greedy search from `start`: path and whether it converged."""
        root = self.evaluate([start], None)
        if root is None:
            return [], False
        current = root[0]
        if not math.isfinite(current.loss):
            msg = f"All fits of the start model {current.id!r} failed"
            raise ValueError(msg)
        path = [current.id]
        moves: list[Move]
        match method:
            case "forward":
                moves = ["forward"]
            case "backward":
                moves = ["backward"]
            case _:
                moves = ["forward", "backward"]
                if self.criterion != "lrt":
                    moves.append("swap")
        move: Move = moves[0]
        failed: set[Move] = set()
        while True:
            neighbours = self.space.neighbours(current.on, move)
            fits = self.evaluate(neighbours, current) if neighbours else []
            if fits is None:
                return path, False
            if (nxt := self.better(current, move, fits)) is not None:
                current = nxt
                path.append(current.id)
                failed.clear()
                if move == "swap":
                    move = "forward"
                continue
            failed.add(move)
            remaining: list[Move] = [m for m in moves if m not in failed]
            if not remaining:
                return path, True
            move = remaining[0]


###############################################################################
# Entry points
###############################################################################


def run[M: _Model](
    space: ModelSpace[M],
    data: pd.Series | pd.DataFrame,
    p0: Mapping[str, float | Distribution],
    *,
    loss_fn: LossFn,
    criterion: Criterion = "aicc",
    method: Method = "adaptive",
    start: Iterable[str] | None = None,
    n_starts: int = 10,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
    runner: Any | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    transform: Transform | None = None,
    postprocess_fn: PostprocessFn | None = None,
    significance: float = 0.05,
    min_improvement: float = 0.0,
    max_candidates: int | None = None,
    reuse: Selection[M] | None = None,
    seed: int = 0,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> Selection[M]:
    """Select a model structure from a model space.

    Every candidate is fitted from a warm start and `n_starts` random starts,
    and compared by the best fit's log-likelihood and number of estimated
    parameters. The search moves between candidates that differ by one
    switch, fitting all neighbours of the current model in parallel and
    moving to the best one if it improves the criterion:

    - `forward`: turns one switch on at a time, from `start` (default: all off)
    - `backward`: turns one switch off at a time, from `start` (default: all on)
    - `adaptive`: forward until no neighbour improves, then backward, then
      exchanges of one switch for another (not with likelihood-ratio tests),
      returning to forward after an improvement, until no move improves
      (like FAMoS). Candidates are fitted once, revisits reuse their fits
    - `brute_force`: fits all candidates

    With an information criterion a move improves if it lowers the criterion
    by more than `min_improvement`. With likelihood-ratio tests (`lrt`), a
    forward move is taken if the larger model is significantly better (the
    lowest p-value below `significance`), a backward move if the smaller one
    isn't significantly worse (the highest p-value at least `significance`); this assumes
    each switch adds parameters to a nested model, e.g. `free_parameter`.

    Each candidate is warm-started from the optimum of the model it is a
    neighbour of: shared parameters start at their estimates and parameters
    the parent fixed start at the fixed value, so with `free_parameter`
    switches the warm start reproduces the parent's fit.

    Parameters
    ----------
    space
        The candidate models
    data
        A Series fits a steady state, a DataFrame (indexed by time) a time
        course, or a protocol time course if `protocol` is given
    p0
        Parameters (and initial values) to estimate, with the distribution
        random starts are drawn from, or a fixed start value. Each candidate
        estimates those of them its model has and doesn't fix; the others
        keep their values in the model. Names in none of the base model and
        the models with one switch on raise an error (typos)
    loss_fn
        A likelihood loss, e.g. `fit.GaussianNll(sigma)`, whose negative is
        the log-likelihood of the criteria
    criterion
        `aic`, `aicc` (default), `bic` or `lrt` (likelihood-ratio tests,
        not with `brute_force`)
    method
        Search strategy, see above
    start
        Switches that are on in the first model of the search
    n_starts
        Random starts per candidate, in addition to the warm start
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits
    protocol
        Protocol of a protocol time course
    runner
        How the models are simulated, see `mxlpy.runners`. Defaults to the
        default runner of each model
    bounds
        Bounds of the estimated parameters, random starts are clipped to them
    prior
        Priors of the estimated parameters, as in the fit functions. The fits
        are then MAP estimates, while the criteria use the data's likelihood
    penalty
        Further penalties, as in the fit functions. Penalties on values a
        candidate fixes become constants, which don't change its fit
    transform
        Fit `log` (log10) or `linear` values, for all or per parameter
    postprocess_fn
        Applied to the simulation before the loss, as in the fit functions
    significance
        Significance level of the likelihood-ratio tests
    min_improvement
        Decrease of an information criterion a move has to achieve
    max_candidates
        Maximum number of candidates to fit (reused fits don't count). The
        search stops before a step that would exceed it, with
        `Selection.complete` False
    reuse
        A previous selection on the same space, data and fit settings, whose
        candidates aren't fitted again, e.g. to continue it with another
        method. Only the number of data points is checked
    seed
        Seed of the random starts, which are drawn per candidate, so they
        don't depend on the order candidates are fitted in
    parallel
        Whether to fit in parallel. All starts of all candidates of a search
        step are fitted at once
    max_workers
        Maximum number of worker processes
    timeout
        Timeout in seconds per fit when fitting in parallel; fits that time
        out count as failed. Errors raised by a fit (not failures to
        converge) abort the selection

    Returns
    -------
    Selection
        Fits of all evaluated candidates, the selected model and the path
        of the search

    Examples
    --------
        >>> space = select.ModelSpace(
        ...     base=model,
        ...     switches={
        ...         "feedback": select.free_parameter("k_fb", fixed_value=0.0),
        ...         "degradation": lambda m: m.add_reaction("deg", ...),
        ...     },
        ... )
        >>> selection = select.run(
        ...     space, data, {"k1": LogUniform(1e-2, 1e2), "k_fb": LogUniform(1e-2, 1e2)},
        ...     loss_fn=fit.GaussianNll(sigma=0.05), criterion="aicc",
        ... )
        >>> selection.table
        >>> selection.best_model

    """
    _check_args(loss_fn, n_starts, space.base, y0)
    if criterion not in ("aic", "aicc", "bic", "lrt"):
        msg = f"Unknown criterion {criterion!r}, use 'aic', 'aicc', 'bic' or 'lrt'"
        raise ValueError(msg)
    if method not in ("forward", "backward", "adaptive", "brute_force"):
        msg = f"Unknown method {method!r}"
        raise ValueError(msg)
    if criterion == "lrt" and method == "brute_force":
        msg = "Likelihood-ratio tests compare neighbours, use an information criterion for brute_force"
        raise ValueError(msg)
    if not 0 < significance < 1:
        msg = f"significance has to be between 0 and 1, got {significance}"
        raise ValueError(msg)
    if not min_improvement >= 0:
        msg = f"min_improvement can't be negative, got {min_improvement}"
        raise ValueError(msg)
    if max_candidates is not None and max_candidates < 1:
        msg = f"max_candidates has to be positive, got {max_candidates}"
        raise ValueError(msg)

    fitter = _Fitter(
        data=data,
        p0=p0,
        loss_fn=loss_fn,
        minimizer=fit.LeastSquaresMinimizer() if minimizer is None else minimizer,
        n_starts=n_starts,
        y0=y0,
        protocol=protocol,
        runner=runner,
        bounds={} if bounds is None else dict(bounds),
        prior=prior,
        penalty=_penalties(penalty),
        transform=transform,
        postprocess_fn=postprocess_fn,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        timeout=timeout,
    )
    fitter.check_names(
        [space.build([])[0], *(space.build([k])[0] for k in space.names)]
    )
    fits: dict[str, CandidateFit[M]] = {}
    if reuse is not None:
        if any(c.n_data != fitter.n_data for c in reuse.candidates.values()):
            msg = "reuse has to be a selection on the same data"
            raise ValueError(msg)
        fits = dict(reuse.candidates)
    search = _Search(
        space=space,
        fitter=fitter,
        criterion=criterion,
        significance=significance,
        min_improvement=min_improvement,
        max_candidates=max_candidates,
        fits=fits,
    )

    if method == "brute_force":
        candidates = space.candidates()
        if search.evaluate(candidates, None) is None:
            msg = f"brute_force needs {len(candidates)} candidates, more than max_candidates"
            raise ValueError(msg)
        name = cast("InformationCriterion", criterion)
        ids = [space.id(c) for c in candidates]
        best_id = min(ids, key=lambda k: fits[k].criterion(name))
        path, complete = [best_id], True
    else:
        if start is None:
            start = space.names if method == "backward" else ()
        on = frozenset(start)
        if not space.is_allowed(on):
            msg = f"The start model {space.id(on)!r} isn't allowed"
            raise ValueError(msg)
        path, complete = search.run(on, method)
        if not path:
            msg = "max_candidates is too small to fit the start model"
            raise ValueError(msg)
        best_id = path[-1]
        if not complete:
            warnings.warn(
                "The search stopped at max_candidates before converging",
                stacklevel=2,
            )
    return Selection(
        candidates=fits,
        criterion=criterion,
        best_id=best_id,
        path=path,
        complete=complete,
        switches=space.names,
        significance=significance,
    )


def compare[M: _Model](
    models: Mapping[str, M],
    data: pd.Series | pd.DataFrame,
    p0: Mapping[str, float | Distribution],
    *,
    loss_fn: LossFn,
    criterion: InformationCriterion = "aicc",
    n_starts: int = 10,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
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
) -> Selection[M]:
    """Fit and rank explicitly given model variants.

    Each model is fitted from its nominal parameter values and `n_starts`
    random starts (all in parallel) and ranked by an information criterion.
    For a likelihood-ratio test of two nested variants, use `Selection.lrt`.

    Parameters
    ----------
    models
        Model per name, which aren't modified
    data
        A Series fits a steady state, a DataFrame (indexed by time) a time
        course, or a protocol time course if `protocol` is given
    p0
        Parameters (and initial values) to estimate with the distribution
        random starts are drawn from, or a fixed start value. Each model
        estimates those of them it has; names no model has raise an error
    loss_fn
        A likelihood loss, e.g. `fit.GaussianNll(sigma)`
    criterion
        `aic`, `aicc` (default) or `bic`
    n_starts
        Random starts per model, in addition to its nominal values

    See `run` for the other parameters.

    Examples
    --------
        >>> selection = select.compare(
        ...     {"base": m0, "+feedback": m1}, data,
        ...     {"k1": LogUniform(1e-2, 1e2), "k_fb": LogUniform(1e-2, 1e2)},
        ...     loss_fn=fit.GaussianNll(sigma=0.05), criterion="bic",
        ... )
        >>> selection.table

    """
    if not models:
        msg = "No models to compare"
        raise ValueError(msg)
    for model in models.values():
        _check_args(loss_fn, n_starts, model, y0)
    if criterion not in ("aic", "aicc", "bic"):
        msg = f"compare needs 'aic', 'aicc' or 'bic', not {criterion!r}; use Selection.lrt for tests"
        raise ValueError(msg)
    fitter = _Fitter(
        data=data,
        p0=p0,
        loss_fn=loss_fn,
        minimizer=fit.LeastSquaresMinimizer() if minimizer is None else minimizer,
        n_starts=n_starts,
        y0=y0,
        protocol=protocol,
        runner=runner,
        bounds={} if bounds is None else dict(bounds),
        prior=prior,
        penalty=_penalties(penalty),
        transform=transform,
        postprocess_fn=postprocess_fn,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        timeout=timeout,
    )
    fitter.check_names(models.values())
    specs = [
        _Spec(id=k, on=frozenset(), model=deepcopy(m), fixed={}, parent=None)
        for k, m in models.items()
    ]
    fits: dict[str, CandidateFit[M]] = {c.id: c for c in fitter.fit(specs)}
    best_id = min(fits, key=lambda k: fits[k].criterion(criterion))
    return Selection(
        candidates=fits,
        criterion=criterion,
        best_id=best_id,
        path=[best_id],
    )
