# ADR 0014: Builder-Polymorphic Routines via Runners, Direct `evaluate()` for Steady-State Models

**Status:** Accepted
**Scope:** `src/mxlpy/runners.py`, `src/mxlpy/scan.py`, `src/mxlpy/fit/`,
`src/mxlpy/mc.py`, `src/mxlpy/mca.py`, `src/mxlpy/sensitivity.py`,
`src/mxlpy/carousel.py`, `src/mxlpy/simulator.py`, `src/mxlpy/simulation.py`,
`src/mxlpy/_ode_builder.py`, `src/mxlpy/_steady_state_builder.py`

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

A first design made `fit` and `scan` polymorphic by `@overload`ing each public function
per builder, with each overload dropping the arguments meaningless for that builder
(`integrator`, `tolerance`, `rel_norm`, `oscillation_detector`, `y0`, `worker`). That does
not scale: those arguments were repeated, inconsistently, across ~40 routines (e.g.
steady-state fits couldn't set `tolerance`, `identify` and `fuzzy` couldn't set the
integrator at all), so every routine × builder needed its own pruned overload, and the
injectable `worker=` protocols typed every result as a union, forcing casts in `mc`,
`mca` and `carousel`.

## 2. Decision

1. **One home per task.** Routines stay in `scan`, `fit`, `mc`, `mca` and `sensitivity`
   and accept every builder for which their question makes sense. Questions about
   reactions (flux elasticities, `rate_characteristics`, carousels, `compare`) stay
   `KineticModelBuilder`-only.
2. **Runners own how a model is simulated.** `mxlpy.runners` provides one runner per
   builder (`KineticRunner`, `OdeRunner`, `EvaluationRunner`) typed by the protocols
   `SteadyStateRunner` and `DynamicRunner`. The integrator and the steady-state search
   settings are runner fields; routines only take `runner=` (defaulting to the builder's
   runner) and describe the task. Custom behaviour is a runner subclass overriding one
   method, replacing the per-routine `worker=` arguments.
3. **Runners return `Result`** ([ADR 0003](0003-result-option-vs-exceptions.md)); each
   routine owns its failure policy: NaN rows for scans and `mc`, an infinite residual for
   fits, propagation for `mca`.
4. **Result types share accessors, shapes differ only where the model does.**
   `Simulation`, `OdeSimulation` (no fluxes) and `Evaluation` (no variables) all offer
   `get_combined()` / `get_args()`, so fit residuals and scans have one code path.
   Containers (`SteadyStateScan[S]`, `ResponseCoefficients[S]`, `Fit[M]`, …) are generic;
   `variables` / `fluxes` are restricted to the result types having them. Type variables
   use `typing_extensions.TypeVar` defaults, so bare names keep their kinetic meaning.
5. **Misuse is caught statically and at runtime.** Overloads exist only where the result
   type or the valid arguments differ (e.g. no `y0` for `SteadyStateModelBuilder`); the
   same misuse raises a `TypeError` naming the right alternative, for users without a
   type checker. Overloaded routines delegate to private non-overloaded implementations,
   which internal callers use with not-yet-resolved model types.
6. **Routines never mutate the caller's model.** `scan`, `mc`, `mca`, `carousel`,
   `identify` and `fuzzy` copy the model at entry (applying `y0` to the copy); `fit`
   keeps its `as_deepcopy` flag.
7. **The `SteadyStateModelBuilder` breaks the `Simulator` convention** and is evaluated
   directly via `model.evaluate()`. Going through `Simulator` would be pure ceremony for
   a single forward evaluation. The method is called `evaluate` rather than `simulate`
   (nothing is integrated), `solve` (nothing is root-found) or `get_args` (which on the
   other builders evaluates at given variable values). `Simulator` itself is unchanged
   and remains the low-level API, wrapped by the runners.

## 3. Rationale

**Configuration belongs to what it configures.** Integrator and steady-state settings
describe *how* a model is run, not the scan or fit using it. Bundling them in a runner
makes them consistent across routines, and a `SteadyStateModelBuilder` simply has a
runner without such fields instead of every routine pruning them.

**Strategy per builder, not per routine.** A runner subclass customises every routine at
once, whereas a `worker=` only reached routines that happened to expose one.

**Overloads only where they carry information.** With configuration in runners, the
remaining overloads map model type → runner type → result type, so adding a builder or a
routine doesn't multiply pruned signatures.

**`y0` stays on routines.** It is model state rather than simulation configuration, and
users expect the convenience; copying the model at entry removes the side effects it used
to have.

## 4. Consequences

- `integrator=`, `tolerance=`, `rel_norm=`, `oscillation_detector=` and `worker=` were
  removed from all routines without a deprecation period (pre-1.0), as were the
  `…Worker` protocols, `EvaluationScan` and `evaluate_residual`. Projects with custom
  workers migrate them to runner subclasses.
- New builders or routines plug in by adding a runner and, where the result shape
  differs, overloads mapping to it.
- Results without fluxes or variables raise a `TypeError` on `.fluxes` / `.variables`;
  `combined` is always available.
- Runners must be picklable for parallel routines, which plain dataclasses are.
