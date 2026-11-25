# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Two-dimensional profile likelihoods and joint confidence regions."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from mxlpy.fit._problem import Constraint, Prediction, _Problem, start_values
from mxlpy.identify._prediction import (
    _SIGMA,
    _setup,
    _Walker,
    predicted_values,
    prediction_scale,
)
from mxlpy.identify._profile import (
    SideStatus,
    Step,
    _Coordinate,
    coordinates_of,
    crossing,
    walk,
)
from mxlpy.parallel import parallelise

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mxlpy import fit
    from mxlpy.distributions import Prior
    from mxlpy.fit._gradient import Transform
    from mxlpy.fit._problem import _Model
    from mxlpy.fit.abstract import Data, PostprocessFn
    from mxlpy.fit.callbacks import Callback
    from mxlpy.fit.penalties import Penalty
    from mxlpy.minimizers.abstract import Bounds, LossFn, MinimizerProtocol

__all__ = ["ProfileLikelihood2D", "profile_likelihood_2d"]

# A point whose objective exceeds a neighbour's by more than this fraction
# of the threshold is refitted from the neighbour's parameters
_KINK = 0.5
# Neighbours of a point checked for kinks
_NEIGHBOURS = 4
# Padding of the default grid range beyond the joint confidence region
_PAD = 0.25


@dataclass(kw_only=True)
class ProfileLikelihood2D:
    """Two-dimensional likelihood profile, see `profile_likelihood_2d`.

    Each point fixes `x` and `y` (for a prediction `y`: traces it with a
    pseudo data point) and refits all other parameters.

    Attributes
    ----------
    x
        The profiled parameter on the x axis
    y
        The profiled parameter or prediction on the y axis
    method
        `slices` or `grid`
    optimum
        Optimum the profile starts from (refitted from the given one)
    objective
        Objective at the optimum
    threshold
        Increase of the objective bounding the joint confidence region
    alpha
        Confidence level the threshold was computed from, None if the
        threshold was given
    log_x, log_y
        Whether the axes were profiled in log10 space
    points
        Profile points with columns `x`, `y` (their values), `delta`
        (increase of the objective over the optimum) and `slice` (index of
        the x slice or grid row)
    parameters
        Values of all fitted parameters per point (same index)
    slices
        Per slice: `x`, `delta` of its start, and `lower_status` /
        `upper_status`, why the profile of `y` stopped below / above its
        start (`threshold`, `bound`, `max_steps`, `failed`; `grid` for grid
        rows)
    n_repaired
        Number of points improved by refitting from a neighbour

    """

    x: str
    y: str | Prediction
    method: Literal["slices", "grid"]
    optimum: dict[str, float]
    objective: float
    threshold: float
    alpha: float | None
    log_x: bool
    log_y: bool
    points: pd.DataFrame
    parameters: pd.DataFrame
    slices: pd.DataFrame
    n_repaired: int

    @property
    def y_name(self) -> str:
        """Name of the y axis: the parameter or the prediction's target."""
        return self.y.target if isinstance(self.y, Prediction) else self.y

    def region(
        self, alpha: float | None = None, *, threshold: float | None = None
    ) -> pd.DataFrame:
        """Boundary of the joint confidence region, as a `y` interval per x slice.

        Columns `lower` and `upper` indexed by `x`, for the slices whose
        smallest objective is within the region. Bounds are interpolated
        where the profile crosses the threshold, infinite where the profile
        of a slice stopped before it (or, for a prediction stopped at the end
        of the reachable predictions, the most extreme one).

        Parameters
        ----------
        alpha
            Confidence level of the region, for profiles of likelihood
            losses. The profile's own by default
        threshold
            Increase of the objective bounding the region instead

        """
        if threshold is None:
            threshold = self.threshold
            if alpha is not None:
                if self.alpha is None:
                    msg = "alpha needs a profile of a likelihood loss, pass threshold="
                    raise ValueError(msg)
                threshold = float(stats.chi2.ppf(alpha, df=2) / 2)
        if self.method == "slices" and threshold > self.threshold * (1 + 1e-12):
            msg = (
                "Slices stop at the profile's threshold, so the region can't be "
                f"wider: threshold {threshold:.4g} > {self.threshold:.4g}"
            )
            raise ValueError(msg)
        rows = {}
        for i, unsorted in self.points.groupby("slice", sort=True):
            group = unsorted.sort_values("y")
            delta = pd.Series(
                group["delta"].to_numpy(), index=group["y"].to_numpy(dtype=float)
            )
            if not (delta < threshold).any():
                continue
            info = self.slices.loc[i]
            centre = float(delta.index[int(np.argmin(delta.to_numpy()))])
            bounds = []
            pairs: tuple[tuple[Literal[-1, 1], str], ...] = (
                (-1, info["lower_status"]),
                (1, info["upper_status"]),
            )
            for side, status in pairs:
                bound = crossing(delta, centre, threshold, side, log=self.log_y)
                if (
                    math.isinf(bound)
                    and status == "bound"
                    and isinstance(self.y, Prediction)
                ):
                    x = delta.index.to_numpy(dtype=float)
                    bound = float(x.max() if side > 0 else x.min())
                bounds.append(bound)
            rows[float(info["x"])] = bounds
        out = pd.DataFrame.from_dict(rows, orient="index", columns=["lower", "upper"])
        out.index.name = self.x
        return out.sort_index()


###############################################################################
# Refits at fixed points
###############################################################################


@dataclass(frozen=True, kw_only=True)
class _Axis:
    """How a point with fixed `x` and `y` (or pseudo data point `z`) is refitted."""

    problem: _Problem
    x: str
    y: str | Prediction
    objective: float
    sigma: float

    def refit(
        self, x: float, y: float, start: dict[str, float]
    ) -> tuple[float, float, dict[str, float]] | None:
        """`(delta, y value, parameters)` with `x` fixed and `y` (or z = `y`)."""
        model = deepcopy(self.problem.model)
        if isinstance(self.y, Prediction):
            p0 = {k: v for k, v in start.items() if k != self.x}
            constraint = Constraint(
                prediction=self.y,
                model=self.problem.condition(self.y),
                z=y,
                sigma=self.sigma,
            )
            res = self.problem.fit_constrained(model, {self.x: x}, p0, constraint)
            if res is None:
                return None
            return res[0] - self.objective, res[2], res[1]
        p0 = {k: v for k, v in start.items() if k not in (self.x, self.y)}
        if (fitted := self.problem.fit(model, {self.x: x, self.y: y}, p0)) is None:
            return None
        return fitted[0] - self.objective, y, fitted[1]


def _y_slice(
    task: tuple[int, int],
    *,
    axis: _Axis,
    starts: list[tuple[float, float, dict[str, float]]],
    coordinate: _Coordinate | None,
    scale: float,
    threshold: float,
    max_steps: int,
) -> tuple[list[tuple[float, float, dict[str, float], float]], SideStatus]:
    """Profile `y` in one direction at the x value of slice `i`.

    Returns `(y, delta, parameters, z)` per point, `z` the pseudo data point
    of a prediction (NaN for a parameter).
    """
    i, direction = task
    x, delta, pars = starts[i]
    problem = axis.problem
    if isinstance(axis.y, Prediction):
        walker = _Walker(
            problem=problem,
            prediction=axis.y,
            objective=axis.objective,
            sigma=axis.sigma,
            validation=False,
            fixed={axis.x: x},
        )
        g = problem.predict(axis.y, problem.condition(axis.y), pars)
        if g is None:
            return [], "failed"
        steps, status = walker.walk(
            direction,
            start=Step(float(g[0]), delta, pars, info=float(g[0])),
            scale=scale,
            threshold=threshold,
            max_steps=max_steps,
        )
        return [(p.info, p.delta, p.pars, p.u) for p in steps], status
    if coordinate is None:
        msg = "A parameter axis needs a coordinate"
        raise ValueError(msg)
    points, status = walk(
        (axis.y, direction),
        problem=problem,
        optimum=pars,
        objective=axis.objective,
        threshold=threshold,
        coordinates={axis.y: coordinate},
        max_steps=max_steps,
        fixed={axis.x: x},
        start_delta=delta,
    )
    return [(yv, d, p, math.nan) for yv, d, p in points], status


def _grid_row(
    task: tuple[float, dict[str, float], float],
    *,
    axis: _Axis,
    ys: list[float],
) -> list[tuple[float, float, dict[str, float]] | None]:
    """Refits along one grid row, outwards from the y closest to the start's."""
    x, start, y_start = task
    centre = int(np.argmin([abs(y - y_start) for y in ys]))
    out: list[tuple[float, float, dict[str, float]] | None] = [None] * len(ys)
    for order in (range(centre, len(ys)), range(centre - 1, -1, -1)):
        pars = start
        for j in order:
            if (res := axis.refit(x, ys[j], pars)) is not None:
                out[j] = res
                pars = res[2]
    return out


def _refit_task(
    task: tuple[float, float, dict[str, float]], *, axis: _Axis
) -> tuple[float, float, dict[str, float]] | None:
    x, y, start = task
    return axis.refit(x, y, start)


###############################################################################
# Kink repair
###############################################################################


def _repair(
    points: list[dict[str, Any]],
    *,
    axis: _Axis,
    threshold: float,
    log_x: bool,
    log_y: bool,
    parallel: bool,
    max_workers: int | None,
) -> int:
    """Refit points much worse than a neighbour from the neighbour's parameters.

    Failed or stuck fits leave kinks: points whose objective exceeds a
    nearby point's by more than `_KINK · threshold`. Each is refitted from
    its best neighbour (at its own x and y, or z for a prediction) and
    replaced if that improves it. Returns the number of improved points.
    """
    if len(points) <= _NEIGHBOURS:
        return 0
    coords = np.array(
        [
            [
                math.log10(p["x"]) if log_x else p["x"],
                math.log10(p["y"]) if log_y and p["y"] > 0 else p["y"],
            ]
            for p in points
        ]
    )
    span = np.ptp(coords, axis=0)
    coords = coords / np.where(span > 0, span, 1.0)
    deltas = np.array([p["delta"] for p in points])
    tasks = []
    for i, p in enumerate(points):
        dist = np.hypot(*(coords - coords[i]).T)
        dist[i] = np.inf
        nearest = np.argsort(dist)[:_NEIGHBOURS]
        best = int(nearest[np.argmin(deltas[nearest])])
        if deltas[i] - deltas[best] > _KINK * threshold:
            y = p["z"] if isinstance(axis.y, Prediction) else p["y"]
            tasks.append((i, (p["x"], y, points[best]["pars"])))
    if not tasks:
        return 0
    results = parallelise(
        partial(_refit_task, axis=axis),
        inputs=tasks,
        parallel=parallel,
        max_workers=max_workers,
        tqdm_desc="repair",
    )

    def objective(delta: float, y: float, z: float) -> float:
        """What the refit minimised: for a prediction including its pseudo data point."""
        if isinstance(axis.y, Prediction):
            return delta + ((y - z) / axis.sigma) ** 2 / 2
        return delta

    n = 0
    for i, res in results:
        p = points[i]
        if res is not None and objective(res[0], res[1], p["z"]) < objective(
            p["delta"], p["y"], p["z"]
        ):
            p.update(delta=res[0], y=res[1], pars=res[2])
            n += 1
    return n


###############################################################################
# Entry point
###############################################################################


def _ranges(
    names: Sequence[str],
    profiles: Mapping[
        tuple[str, int], tuple[list[tuple[float, float, dict[str, float]]], SideStatus]
    ],
    coordinates: Mapping[str, _Coordinate],
    threshold: float,
) -> dict[str, tuple[float, float]]:
    """Grid range per parameter: its 1D profile's interval, padded.

    The interval of a 1D profile at the threshold of the 2D region is the
    projection of the region onto that axis. Sides not crossing it end at
    the furthest profiled value.
    """
    out = {}
    for k in names:
        coord = coordinates[k]
        ends = []
        for d in (-1, 1):
            points, status = profiles[(k, d)]
            xs = [coord.value, *(p[0] for p in points)]
            delta = pd.Series(
                [0.0, *(p[1] for p in points)], index=np.asarray(xs, dtype=float)
            ).sort_index()
            bound = (
                crossing(
                    delta, coord.value, threshold, 1 if d > 0 else -1, log=coord.log
                )
                if status == "threshold"
                else math.inf
            )
            if math.isinf(bound):
                bound = max(xs) if d > 0 else min(xs)
            ends.append(coord.to_u(bound))
        width = (ends[1] - ends[0]) or abs(coord.to_u(coord.value)) or 1.0
        lower = max(ends[0] - _PAD * width, coord.to_u(coord.lower))
        upper = min(ends[1] + _PAD * width, coord.to_u(coord.upper))
        out[k] = (coord.to_x(lower), coord.to_x(upper))
    return out


def _axis_values(value_range: tuple[float, float], n: int, *, log: bool) -> list[float]:
    lower, upper = value_range
    if log:
        return [float(v) for v in np.geomspace(lower, upper, n)]
    return [float(v) for v in np.linspace(lower, upper, n)]


def profile_likelihood_2d(
    model: _Model,
    data: Data,
    parameters: fit.Fit | Mapping[str, float],
    *,
    x: str,
    y: str | Prediction,
    loss_fn: LossFn | None = None,
    method: Literal["slices", "grid"] = "slices",
    alpha: float = 0.95,
    threshold: float | None = None,
    n_steps: int = 20,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    minimizer: MinimizerProtocol | None = None,
    y0: dict[str, float] | None = None,
    protocol: pd.DataFrame | None = None,
    runner: Any | None = None,
    bounds: Bounds | None = None,
    prior: Mapping[str, Prior] | None = None,
    penalty: Penalty | Sequence[Penalty] | None = None,
    transform: Transform | None = None,
    postprocess_fn: PostprocessFn | None = None,
    standard_scale: bool = True,
    max_steps: int = 100,
    parallel: bool = True,
    max_workers: int | None = None,
    callback: Callback | None = None,
) -> ProfileLikelihood2D:
    """Two-dimensional profile likelihood and joint confidence region.

    Profiles the objective jointly over two parameters `x` and `y` (or a
    parameter and a prediction), refitting all other parameters at each
    point. The shape of the region shows how the two compensate for each
    other, e.g. a valley along `k1 / k2 = const` if only the ratio is
    identifiable, which neither 1D profiles nor 2D scans with the other
    parameters fixed show.

    - `slices` (default): profiles `x` (as `profile_likelihood`), then at
      each of its points profiles `y` with `x` fixed in both directions,
      until the threshold. No fits are spent outside the region, and `y`
      can be a `Prediction` (traced as in `prediction_profile`)
    - `grid`: refits on an `n_steps` x `n_steps` grid over `x_range` x
      `y_range` (log-spaced for log-transformed parameters), warm-started
      along each row, rows in parallel. Without ranges, 1D profiles of `x`
      and `y` at the threshold give the region's extent, padded by 25%.
      Parameter pairs only

    Afterwards, points much worse than a neighbour (kinks from failed fits)
    are refitted from the neighbour's parameters.

    With a likelihood loss the threshold is `chi2.ppf(alpha, 2) / 2` of a
    joint region of two quantities. Takes the same arguments as
    `profile_likelihood`, see there.

    Parameters
    ----------
    model
        The model, which isn't modified
    data
        Data of the fit, as for `profile_likelihood`
    parameters
        Fitted parameters (and initial values), e.g. a fit result
    x
        Fitted parameter on the x axis
    y
        Fitted parameter or `Prediction` on the y axis
    loss_fn
        Loss of the fits, the same as for the optimum
    method
        `slices` or `grid`, see above
    alpha
        Confidence level of the region, for likelihood losses
    threshold
        Increase of the objective bounding the region
    n_steps
        Grid points per axis of the `grid` method
    x_range, y_range
        Ranges of the grid, from 1D profiles by default
    minimizer
        Minimizer of the fits, `fit.LeastSquaresMinimizer()` by default
    y0
        Initial conditions of the fits
    protocol
        Protocol of a protocol time course fit
    runner
        How the model is simulated, see `mxlpy.runners`
    bounds
        Bounds of the fits per parameter, which also limit the profiles
    prior
        Priors of the parameters, as in the fit functions
    penalty
        Further penalties, as in the fit functions
    transform
        `log` (or per parameter) to fit and profile these parameters in
        log10 space
    postprocess_fn
        Applied to the simulation before the loss and the prediction
    standard_scale
        Whether to standard-scale data and prediction, as in the fit
        functions. Likelihood losses are never scaled
    max_steps
        Maximum number of fits per side of each 1D profile
    parallel
        Whether to compute in parallel
    max_workers
        Maximum number of worker processes
    callback
        Called with a `FitEvent` after every evaluation and at the end of
        every refit, see `mxlpy.fit.callbacks`. Has to be picklable with
        `parallel=True`

    Examples
    --------
        >>> pl2d = identify.profile_likelihood_2d(model, data, res, x="k1", y="k2",
        ...                                       loss_fn=fit.GaussianNll(0.05))
        >>> pl2d.region()
        >>> plot.profile_likelihood_2d(pl2d)

    """
    if method not in ("slices", "grid"):
        msg = f"method has to be 'slices' or 'grid', got {method!r}"
        raise ValueError(msg)
    if method == "grid" and isinstance(y, Prediction):
        msg = "The grid method needs two parameters, profile predictions with slices"
        raise ValueError(msg)
    if n_steps < 2:  # noqa: PLR2004, a grid needs two points per axis
        msg = f"n_steps has to be at least 2, got {n_steps}"
        raise ValueError(msg)
    start = start_values(parameters)
    names = [x] if isinstance(y, Prediction) else [x, y]
    if unknown := sorted(set(names) - set(start)):
        msg = f"Profiled parameters have to be fitted ones, not {unknown}"
        raise ValueError(msg)
    if len(set(names)) < len(names):
        msg = "x and y have to differ"
        raise ValueError(msg)
    setup = _setup(
        model,
        data,
        parameters,
        loss_fn=loss_fn,
        alpha=alpha,
        threshold=threshold,
        max_steps=max_steps,
        what="2D profiles",
        df=2,
        minimizer=minimizer,
        y0=y0,
        protocol=protocol,
        runner=runner,
        bounds=bounds,
        prior=prior,
        penalty=penalty,
        transform=transform,
        postprocess_fn=postprocess_fn,
        standard_scale=standard_scale,
        callback=callback,
    )
    problem, optimum, thr = setup.problem, setup.optimum, setup.threshold
    coordinates = coordinates_of(names, optimum, problem, setup.log)
    run = partial(parallelise, parallel=parallel, max_workers=max_workers)

    scale, sigma = 1.0, 1.0
    if isinstance(y, Prediction):
        value = predicted_values(problem, y, optimum)
        scale = float(
            prediction_scale(
                problem,
                y,
                model=problem.condition(y),
                optimum=optimum,
                values=value,
                log=setup.log,
            )[0]
        )
        sigma = _SIGMA * scale
    axis = _Axis(problem=problem, x=x, y=y, objective=setup.objective, sigma=sigma)

    walk_fn = partial(
        walk,
        problem=problem,
        optimum=optimum,
        objective=setup.objective,
        threshold=thr,
        coordinates=coordinates,
        max_steps=max_steps,
    )
    if method == "grid" and x_range is not None and y_range is not None:
        to_profile = [x]
    else:
        to_profile = names if method == "grid" else [x]
    profiles = dict(
        run(
            walk_fn,
            inputs=[((k, d), (k, d)) for k in to_profile for d in (-1, 1)],
            tqdm_desc=f"profile of {x}",
        )
    )
    x_points = sorted(
        [*profiles[(x, -1)][0], (optimum[x], 0.0, optimum), *profiles[(x, 1)][0]],
        key=lambda p: p[0],
    )

    rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    if method == "slices":
        starts = [p for p in x_points if p[1] < thr]
        sides = dict(
            run(
                partial(
                    _y_slice,
                    axis=axis,
                    starts=starts,
                    coordinate=None if isinstance(y, Prediction) else coordinates[y],
                    scale=scale,
                    threshold=thr,
                    max_steps=max_steps,
                ),
                inputs=[((i, d), (i, d)) for i in range(len(starts)) for d in (-1, 1)],
                tqdm_desc="slices",
            )
        )
        for i, (xv, delta, pars) in enumerate(starts):
            y_start = (
                float(predicted_values(problem, y, pars)[0])
                if isinstance(y, Prediction)
                else pars[y]
            )
            lower, lower_status = sides[(i, -1)]
            upper, upper_status = sides[(i, 1)]
            # At the slice's start the pseudo data point equals the prediction
            z_start = y_start if isinstance(y, Prediction) else math.nan
            for yv, d, p, z in [*lower, (y_start, delta, pars, z_start), *upper]:
                rows.append(
                    {"x": xv, "y": yv, "delta": d, "slice": i, "pars": p, "z": z}
                )
            slice_rows.append(
                {
                    "x": xv,
                    "delta": delta,
                    "lower_status": lower_status,
                    "upper_status": upper_status,
                }
            )
    else:
        y_name = str(y)
        ranges = _ranges(to_profile, profiles, coordinates, thr)
        xs = _axis_values(
            x_range if x_range is not None else ranges[x], n_steps, log=x in setup.log
        )
        ys = _axis_values(
            y_range if y_range is not None else ranges[y_name],
            n_steps,
            log=y_name in setup.log,
        )
        profile_x = np.array([p[0] for p in x_points])
        tasks = []
        for i, xv in enumerate(xs):
            nearest = x_points[int(np.argmin(np.abs(profile_x - xv)))][2]
            tasks.append((i, (xv, nearest, nearest[y_name])))
        for i, fits in run(
            partial(_grid_row, axis=axis, ys=ys), inputs=tasks, tqdm_desc="grid"
        ):
            for j, res in enumerate(fits):
                if res is not None:
                    rows.append(
                        {
                            "x": xs[i],
                            "y": ys[j],
                            "delta": res[0],
                            "slice": i,
                            "pars": res[2],
                        }
                    )
            slice_rows.append(
                {
                    "x": xs[i],
                    "delta": min(
                        (r[0] for r in fits if r is not None), default=math.nan
                    ),
                    "lower_status": "grid",
                    "upper_status": "grid",
                }
            )

    log_x = x in setup.log
    log_y = not isinstance(y, Prediction) and y in setup.log
    for row in rows:
        row.setdefault("z", math.nan)
    n_repaired = _repair(
        rows,
        axis=axis,
        threshold=thr,
        log_x=log_x,
        log_y=log_y,
        parallel=parallel,
        max_workers=max_workers,
    )
    points = pd.DataFrame(
        {
            "x": [r["x"] for r in rows],
            "y": [r["y"] for r in rows],
            "delta": [r["delta"] for r in rows],
            "slice": [r["slice"] for r in rows],
        }
    )
    if isinstance(y, Prediction):
        points["z"] = [r["z"] for r in rows]
    return ProfileLikelihood2D(
        x=x,
        y=y,
        method=method,
        optimum=optimum,
        objective=setup.objective,
        threshold=thr,
        alpha=setup.alpha,
        log_x=log_x,
        log_y=log_y,
        points=points,
        parameters=pd.DataFrame([r["pars"] for r in rows], columns=list(optimum)),
        slices=pd.DataFrame(slice_rows),
        n_repaired=n_repaired,
    )
