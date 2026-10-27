# ADR 0014: Builder-Polymorphic `fit`/`scan`, Direct `evaluate()` for Steady-State Models

**Status:** Accepted
**Scope:** `src/mxlpy/scan.py`, `src/mxlpy/fit/`, `src/mxlpy/simulator.py`,
`src/mxlpy/simulation.py`, `src/mxlpy/_ode_builder.py`,
`src/mxlpy/_steady_state_builder.py`

---

## 1. Context

mxlpy has three model builders: `KineticModelBuilder` (`Model`, $N \cdot v = \dot x$),
`OdeModelBuilder` ($\dot x = f(x)$) and `SteadyStateModelBuilder` (no variables, only
explicit functions of parameters). The latter two were initially only used for MxlWeb
codegen, so only the `KineticModelBuilder` could be simulated, scanned or fitted.

PR #184 added simulation and parameter scans for the `SteadyStateModelBuilder` as
methods *on the builder* (`model.simulate()`, `model.simulate_scan()`). It was merged by
accident and reverted in #188, because it moved scanning out of `mxlpy.scan` for one
builder only, so the same task would have had two different homes depending on the model
type.

## 2. Decision

1. **`fit.*` and `scan.*` are polymorphic over the builder**, via `@overload`s on the
   public functions. Each overload only lists the arguments meaningful for that builder
   (e.g. the `SteadyStateModelBuilder` overload of `scan.steady_state` has no
   `integrator`, `y0`, `tolerance` or `oscillation_detector`). Dispatch on the model type
   happens once, when choosing the default worker / residual function; minimisation and
   parallelisation stay shared.
2. **Return types are generic where only the model type differs, distinct where the
   result shape differs.** `Fit[M]` hands back the builder type that was passed in. An
   `OdeModelBuilder` simulation is an `OdeSimulation` (no fluxes), and the scan containers
   are parameterised by it (`SteadyStateScan[OdeSimulation]`), with flux accessors only
   available for `Simulation`. A `SteadyStateModelBuilder` scan returns an
   `EvaluationScan`, as it has neither variables nor fluxes.
3. **The `SteadyStateModelBuilder` breaks the `Simulator` convention** and is evaluated
   directly via `model.evaluate()`. Going through
   `Simulator(model).simulate().get_result().unwrap_or_err()` would be pure ceremony for
   what is a single forward evaluation. The method is called `evaluate` rather than
   `simulate` (nothing is integrated), `solve` (nothing is root-found) or `get_args`
   (which on the other builders evaluates at given variable values).

## 3. Rationale

**One home per task.** Users look for scans in `scan` and fits in `fit`. Methods on one
builder would split that, and every future analysis (`mc`, `mca`, …) would face the same
question again.

**Overloads over unions.** A single signature taking a union would accept
`integrator=` for a model that is never integrated, silently ignoring it. Overloads let
the type checker reject that, while the implementation stays one function.

**Overloads over separate functions.** Separate functions per builder (e.g.
`fit.evaluate`, `scan.evaluate`) would avoid overloads, but multiply the API surface
across the ~20 `fit` variants as builder support grows.

## 4. Consequences

- Only the single-model routines are builder-polymorphic so far: `fit.steady_state`
  (all three builders), `fit.time_course` / `fit.protocol_time_course` and all `scan.*`
  functions (`KineticModelBuilder` and `OdeModelBuilder`; `scan.steady_state` also for
  `SteadyStateModelBuilder`). Group, ensemble, joint and carousel fits, `mc` and `mca`
  remain `KineticModelBuilder`-only; extending them should follow the same pattern.
- The private scan workers return `Simulation | OdeSimulation`; `KineticModelBuilder`-only
  callers (`mc`, `mca`, `carousel`) `cast` their results to `Simulation`.
- `Simulator` is generic over the builder; `get_result()` returns `Result[Simulation]` or
  `Result[OdeSimulation]` depending on it.
