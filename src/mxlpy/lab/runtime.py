# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Runtime estimates of the fitting lab.

The cost of a step is its number of evaluations (simulations with
sensitivities of the residuals) times the time per evaluation. Step 0 seeds
the time per evaluation with one simulation and one sensitivity solve; every
evaluation reported later refines it with a running window per number of
estimated parameters, which follows changes of stiffness. Evaluation counts
come as a (low, typical, high) range per step, so the estimate is a range:
optimistic (few evaluations at the typical cost), typical, and pessimistic
(many evaluations at the 90th percentile of the cost).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Estimate", "RuntimeModel", "fit_evaluations"]

_WINDOW = 200


@dataclass(frozen=True)
class Estimate:
    """Runtime range in seconds: optimistic, typical and pessimistic.

    Optimistic: the fewest expected evaluations at the median cost per
    evaluation; pessimistic: the most at its 90th percentile.
    """

    low: float
    typical: float
    high: float

    def __add__(self, other: Estimate) -> Estimate:
        """Sum of two estimates (steps run one after the other)."""
        return Estimate(
            self.low + other.low, self.typical + other.typical, self.high + other.high
        )

    def scaled(self, factor: float) -> Estimate:
        """The estimate times `factor`."""
        return Estimate(self.low * factor, self.typical * factor, self.high * factor)

    def to_dict(self) -> dict[str, float]:
        """JSON-compatible dict."""
        return {"low": self.low, "typical": self.typical, "high": self.high}

    def __str__(self) -> str:
        """Human-readable range."""
        return f"{_fmt(self.low)} to {_fmt(self.high)} (typ. {_fmt(self.typical)})"


def _fmt(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds:.0f} s" if seconds >= 1 else f"{seconds * 1e3:.0f} ms"
    if seconds < 3600:  # noqa: PLR2004
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def fit_evaluations(n_parameters: int) -> tuple[float, float, float]:
    """Evaluations (residuals with Jacobian) of one local fit: low, typical, high."""
    p = max(n_parameters, 1)
    return 5.0 + p, 10.0 + 3.0 * p, 30.0 + 10.0 * p


@dataclass
class RuntimeModel:
    """Time per evaluation, refined online.

    Attributes
    ----------
    simulation
        Seconds of one plain simulation of all data sets (from step 0)
    sensitivity
        Seconds of one simulation with the sensitivities of all `n_reference`
        parameters (from step 0)
    n_reference
        Number of parameters of the `sensitivity` timing
    workers
        Parallel workers available to the analyses

    """

    simulation: float = math.nan
    sensitivity: float = math.nan
    n_reference: int = 1
    workers: int = 1
    _observed: dict[int, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=_WINDOW)),
        repr=False,
    )

    def seed(self, simulation: float, sensitivity: float, n_reference: int) -> None:
        """Set the timings of step 0."""
        self.simulation = float(simulation)
        self.sensitivity = float(sensitivity)
        self.n_reference = max(int(n_reference), 1)

    def observe(self, seconds: float, n_parameters: int) -> None:
        """Record the duration of one evaluation with `n_parameters`."""
        if math.isfinite(seconds) and seconds > 0:
            self._observed[int(n_parameters)].append(float(seconds))

    def _seeded(self, n_parameters: int) -> float:
        """Seconds per evaluation interpolated from the step 0 timings."""
        if not math.isfinite(self.simulation):
            return math.nan
        sens = self.sensitivity if math.isfinite(self.sensitivity) else self.simulation
        slope = max(sens - self.simulation, 0.0) / self.n_reference
        # one residual and one Jacobian per evaluation
        return 2 * self.simulation + slope * n_parameters

    def per_evaluation(self, n_parameters: int) -> tuple[float, float]:
        """Typical and p90 seconds per evaluation with `n_parameters`."""
        observed = self._observed.get(int(n_parameters))
        if observed:
            values = np.asarray(observed, dtype=float)
            return float(np.median(values)), float(np.quantile(values, 0.9))
        seeded = self._seeded(n_parameters)
        if math.isfinite(seeded) and self._observed:
            # Scale the seed by how far off it was for other parameter counts
            ratios = [
                float(np.median(np.asarray(v))) / self._seeded(k)
                for k, v in self._observed.items()
                if v and self._seeded(k) > 0
            ]
            if ratios:
                seeded *= float(np.median(ratios))
        return seeded, 1.5 * seeded

    def estimate(
        self,
        evaluations: tuple[float, float, float],
        n_parameters: int,
        *,
        parallel_tasks: int = 1,
    ) -> Estimate:
        """Runtime of `evaluations` (low, typical, high) with `n_parameters`.

        `parallel_tasks` independent tasks share the evaluations and run on
        up to `workers` processes.
        """
        typical, high = self.per_evaluation(n_parameters)
        speedup = max(1, min(self.workers, parallel_tasks))
        low_n, typ_n, high_n = evaluations
        return Estimate(
            low=low_n * typical / speedup,
            typical=typ_n * typical / speedup,
            high=high_n * high / speedup,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-compatible state."""
        return {
            "simulation": self.simulation,
            "sensitivity": self.sensitivity,
            "n_reference": self.n_reference,
            "workers": self.workers,
            "observed": {str(k): list(v) for k, v in self._observed.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> RuntimeModel:
        """Restore a state written by `to_dict`."""
        out = cls(
            simulation=float(state.get("simulation", math.nan)),  # type: ignore[arg-type]
            sensitivity=float(state.get("sensitivity", math.nan)),  # type: ignore[arg-type]
            n_reference=int(state.get("n_reference", 1)),  # type: ignore[arg-type]
            workers=int(state.get("workers", 1)),  # type: ignore[arg-type]
        )
        for k, values in dict(state.get("observed", {})).items():  # type: ignore[arg-type]
            for v in values:
                out.observe(float(v), int(k))
        return out
