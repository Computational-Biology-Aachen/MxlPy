# ADR 0007: `fit/` (Problem Framing) vs. `minimizers/` (Generic Optimization)

**Status:** Implemented
**Scope:** `src/mxlpy/fit/`, `src/mxlpy/minimizers/`

---

## 1. Context

`minimizers/` (`AbstractMinimizer`, `MinimizerProtocol`, `OptimisationState`, the
step-bounded `_fixed_n` minimizer) and `fit/` (`Fit`, `GroupFit`, `EnsembleFit`,
`JointFit`, `FitSettings`) both live under parameter-fitting functionality but at
different levels of abstraction.

## 2. Decision

`minimizers/` is the generic, model-agnostic numerical optimization layer: given a
residual function and a starting point, minimize it. It has no knowledge of `Model`,
ODEs, or time-series data.

`fit/` is the domain layer on top: it knows how to turn "fit this `Model` to this
experimental time-series/steady-state dataset" into the residual function a minimizer
consumes, and orchestrates the scientific variants — single fit, group fit, joint fit,
ensemble fit, mixed-effects fit.

## 3. Rationale

This mirrors the same separation of concerns as `integrators/` (generic ODE solving)
vs. `Model`/`simulator.py` (what's being solved): a generic numerical layer stays
reusable and swappable on its own, while the mechanistic-modeling-specific problem
framing stays separate so it can evolve (new fit variants, new residual/loss shapes)
without the optimizer algorithms needing to know anything changed.

## 4. Consequences

- A new optimization *algorithm* belongs in `minimizers/` and should not import
  anything from `fit/` or `Model`.
- A new *fitting scenario* (a new way of framing what data gets fit to what) belongs in
  `fit/` and should be built on the existing `MinimizerProtocol`, not by adding
  model-awareness into `minimizers/`.
