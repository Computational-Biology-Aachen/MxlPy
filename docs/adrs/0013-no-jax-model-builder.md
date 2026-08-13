# ADR 0013: No Standalone `JaxModelBuilder`

**Status:** Declined
**Scope:** `src/mxlpy/jax/models.py`, `src/mxlpy/_ode_builder.py`,
`src/mxlpy/_kinetic_builder.py`, `src/mxlpy/meta/_via_sym_repr.py`

---

## 1. Context

`OdeModelBuilder` and `KineticModelBuilder` (exported as `Model`) are near-identical
fluent builders — their diff is almost entirely "diff_eq" ↔ "reaction/stoichiometry"
renaming (see [ADR 0002](0002-fluent-builder-and-cache-invalidation.md)). It's a natural
question whether `jax/models.py` should get a third sibling, `JaxModelBuilder`, offering
the same `add_variable`/`add_parameter`/`add_reaction` mutation API but constructing a
JAX-native `eqx.Module` (`Ode`, `FluxOde`, ...) directly instead of a `ModelCache`.

This was prompted by an explicit feasibility check, not a concrete user request blocked
on its absence.

## 2. Decision

Do not build a `JaxModelBuilder`. No new class was added.

## 3. Rationale

**The bridge to JAX already exists and is backend-agnostic.** `OdeModelBuilder`/
`KineticModelBuilder` feed a shared sympy IR (`meta/_via_sym_repr.py:
model_to_symbolic_repr`), mechanically projected to seven targets including JAX — see
[ADR 0008](0008-meta-codegen-single-source-of-truth.md). `Ode.from_mxlpy(model,
parameters_to_fit=..., free_parameters=...)` / `FluxOde.from_mxlpy(...)` already take a
model built with the ordinary fluent API and return a trainable `eqx.Module`. That
classmethod pair *is* the builder this ADR considered adding as a new class — JAX is not
a gap in the existing architecture.

**A hand-written fluent builder targeting JAX directly would duplicate, not replace,
substantial logic.** `OdeModelBuilder`/`_kinetic_builder.py` are ~2000–3000 lines each
of cache invalidation, topological sort (`_topo`), dynamic-stoichiometry resolution, and
unit inference. A `JaxModelBuilder` would need the same bookkeeping to produce a
dependency-ordered, traceable right-hand side — or it would just re-implement a subset
of `model_to_symbolic_repr` + `generate_model_code_jax` by another name. Either way, two
independently maintained implementations of "turn a fluent spec into an ordered RHS"
is exactly the drift [ADR 0008](0008-meta-codegen-single-source-of-truth.md) exists to
prevent, and directly contradicts [ADR 0005](0005-jax-as-orthogonal-subsystem.md)'s
point that `jax/`'s model representation is *not* a maintenance burden separate from
`Model` because it's regenerated, not hand-synchronized.

**The parts of `jax/models.py` with no mechanistic analog don't fit the pattern
either.** `Node`, `Ude`, `FluxUde`, `Anode`, `FluxAnode` (MLP/latent-space/UDE
components) aren't expressible as `add_reaction`/`add_variable` calls — there's no
stoichiometry or rate law to register, just network shape (`width`, `depth`, `key`,
`n_hidden`). These are already about as "fluent" as they need to be via plain
`__init__`/equinox composition (`Ude(ode=Ode.from_mxlpy(...), nn=Node(...), op="+")`).
A mutation-based builder would need a fundamentally different API for this half of
`jax/models.py` anyway, undermining the "mirrors `OdeModelBuilder`/`KineticModelBuilder`"
premise.

**Relation to [ADR 0005](0005-jax-as-orthogonal-subsystem.md) §4.** That ADR's "Future
Directions" section names "additional `JaxModelBuilder` classes" as plausible future
growth. That note refers to more *model-shape* classes in the `Ode`/`FluxOde`/`Anode`
family (new latent-mapper variants, new UDE compositions) — incremental growth of the
existing `eqx.Module` + `from_mxlpy` pattern — not a fluent mutation-based builder
mirroring `OdeModelBuilder`/`KineticModelBuilder`. This ADR resolves that ambiguity: the
mutation-based builder shape was considered and declined; new `eqx.Module` model-shape
classes remain open, unaffected by this decision.

## 4. Consequences

- Users who want a JAX/UDE model from a mechanistic spec continue to: build with the
  ordinary fluent `Model`/`OdeModelBuilder` API, then call `Ode.from_mxlpy(...)` /
  `FluxOde.from_mxlpy(...)`.
- If a real ergonomic gap surfaces in that bridge (e.g. no single call that combines a
  mechanistic `FluxOde.from_mxlpy` with a fresh `FluxNode`/`Node` into a `Ude`/`FluxUde`
  in one step), the fix is an additive convenience classmethod on the relevant
  `jax/models.py` class, not a new builder class.
- If `jax/models.py` model shapes are later found to need incremental, notebook-style
  construction (mirroring how `Model` is typically built cell-by-cell), revisit this
  decision — but note that JAX's tracing requirements (whole-graph, ahead-of-time) sit
  in tension with incremental mutation in a way plain Python `Model` construction does
  not, so this is not a simple port even if revisited.
