# mxlpy: Architecture Context

This document is the entry point for understanding *why* mxlpy is shaped the way it is —
not what each module does (see `.claude/specs/*.md` for detailed, per-module behavior
specs: public API contracts, invariants, edge cases), but the reasoning behind the
architectural decisions, written down before institutional knowledge is lost to a
maintainer handoff.

Each numbered ADR below covers one decision in Status/Context/Decision/Rationale/
Consequences form. Read this page first for the map, then follow links for depth.

## The Core Model

mxlpy's center of gravity is `Model` (`_kinetic_builder.py`, exported as `Model`): a
fluent builder for mechanistic ODE models, where every mutation returns `Self`, and a
`_cache` is invalidated wholesale (not incrementally) on any structural change.

→ [ADR 0002 — Fluent builder + wholesale cache invalidation](0002-fluent-builder-and-cache-invalidation.md)

Two library-wide conventions follow directly from valuing introspectability and
type-checkable failure modes over convenience:

→ [ADR 0003 — `Result`/`Option` for expected failures, exceptions for the rest](0003-result-option-vs-exceptions.md)
→ [ADR 0004 — Named functions only for reactions/derived quantities, never lambdas](0004-named-functions-no-lambdas.md)

## One Model, Many Representations

A recurring theme: a model is authored **once**, and everything else — training-ready
JAX code, LaTeX equations, Julia/Rust snippets, a MxlWeb browser AST, a re-imported SBML
model — is *mechanically derived*, never hand-transcribed. This is the single biggest
idea to internalize before touching `meta/`, `jax/`, `sbml/`, or `symbolic/`.

→ [ADR 0008 — `meta/` codegen: one model, mechanically projected to every target](0008-meta-codegen-single-source-of-truth.md)
→ [ADR 0005 — `jax/` as an orthogonal subsystem, not an integrator plugin](0005-jax-as-orthogonal-subsystem.md)
→ [ADR 0009 — `sbml/` as mxlpy's consumer of the (now-independent) `pysbml`](0009-sbml-dual-use-bridge.md)
→ [ADR 0010 — `symbolic/` scope, and its relationship to `meta/`](0010-symbolic-module-scope.md)

## Subsystem Boundaries

Several packages that look like they could be merged are deliberately split along
scientific-task lines, not implementation-layer lines:

→ [ADR 0006 — `nn/` / `surrogates/` / `npe/` split (forward substitution vs. inverse inference)](0006-nn-surrogates-npe-split.md)
→ [ADR 0007 — `fit/` (problem framing) vs. `minimizers/` (generic optimization)](0007-fit-vs-minimizers-split.md)
→ [ADR 0001 — Parallel execution pool backend (pebble → loky)](0001-parallel-execution-pool-backend.md)

## Project Posture

→ [ADR 0011 — `Ruff ALL` + Bandit + Pyright strict: deliberate rigor for a library other scientists build on](0011-strict-tooling-for-downstream-scientists.md)
→ [ADR 0012 — Dual `uv`/`pixi` toolchains (assimulo is conda-forge-only)](0012-dual-uv-pixi-toolchains.md)

## Threads That Cross Multiple ADRs

- **Introspectability over convenience.** The no-lambdas rule (0004), codegen-as-text
  rather than in-memory construction (0008, 0009), and the pickling/metaprogramming
  constraints in `parallel.py` all trace back to one value: if the system can't inspect
  a function's source or reconstruct it cleanly across a process/language boundary,
  something downstream (codegen, symbolic analysis, process-pool serialization) will
  break in a hard-to-diagnose way. Prefer named, inspectable, source-backed functions
  everywhere a model-facing function is registered.
- **Scars from prior projects inform current trade-offs.** Wholesale (not incremental)
  cache invalidation (0002) is a direct reaction to bugs seen in earlier, related
  projects — not a default anyone would derive from first principles alone.
- **Orthogonal, not layered, growth.** `jax/` (0005) began as "just another integrator
  backend" and grew into a fully separate subsystem once JAX's tracing requirements
  demanded a different model representation. Watch for this pattern elsewhere: a new
  requirement that looks like "just another backend" may in fact demand its own
  abstraction if it has fundamentally different structural requirements (traceability,
  differentiability, etc.).

## Deliberately Out of Scope (For Now)

Conservation-law elimination and symbolic steady-state solving were evaluated and
declined — not as a poor fit, but because conservation laws are currently treated as an
explicit modeller decision, and current model sizes don't yet justify automatic
reduction. See [ADR 0010, §5](0010-symbolic-module-scope.md#5-future-directions-currently-out-of-scope-not-rejected-on-principle).
A broader inventory of considered-and-declined features lives in the separate
`mxlpy-research` repo (`features/rejected/`) but was not incorporated into ADRs here —
most of those are backlog/priority calls, not philosophy-driven rejections.
